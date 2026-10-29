"""Menu bar app: an eye in the menu bar while the monitor is running, plus a native dashboard window."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request

import rumps

from AppKit import (NSAlert, NSAlertFirstButtonReturn, NSApp, NSBackingStoreBuffered, NSClosableWindowMask,
                    NSMakeRect, NSMiniaturizableWindowMask, NSResizableWindowMask, NSSecureTextField,
                    NSTitledWindowMask, NSWindow)
from Foundation import NSProcessInfo, NSURL, NSURLRequest, NSUserDefaults
from PyObjCTools import AppHelper
from WebKit import WKWebView, WKWebViewConfiguration

from . import db, stats
from .classify import ClassifierService
from .collector import Collector
from .config import DATA_DIR, REOPEN_FILE, RESTART_EXIT_CODE, Config, keychain_set, load_config, logical_date
from .icons import menu_icons

log = logging.getLogger("menubar")


def _fmt_min(m: float) -> str:
    return f"{int(m // 60)}h {int(m % 60)}m" if m >= 60 else f"{int(m)}m"


def _stay_awake():
    """Opt out of App Nap and automatic termination.

    2026-09-04: every thread of this process stalled for three days (zero CPU, menu
    unresponsive, no log lines, lldb could not even pause it) while the machine was awake.
    The system log had expired before it was investigated, so the trigger is unproven; the
    leading suspect is macOS deciding an invisible, windowless app is idle. Declaring
    continuous background activity tells the system this process is doing real work while
    invisible. The launcher's heartbeat watchdog (build_app.py) is the backstop if it
    happens anyway. The returned token must stay referenced.
    """
    NSUserDefaults.standardUserDefaults().registerDefaults_({"NSAppSleepDisabled": True})
    background = 0x000000FF  # NSActivityBackground
    no_auto_terminate = 1 << 15  # NSActivityAutomaticTerminationDisabled
    no_sudden_terminate = 1 << 14  # NSActivitySuddenTerminationDisabled
    return NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        background | no_auto_terminate | no_sudden_terminate,
        "Focus Monitor records the active window continuously")


class DashboardWindow:
    """A native window hosting the local dashboard in a WKWebView."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.window = None
        self.webview = None

    def _install_edit_menu(self):
        # A menu-bar-only app has no Edit menu, so Cmd+C/V/X/A/Z do nothing inside the
        # dashboard's text fields - key equivalents are resolved through the main menu.
        from AppKit import NSMenu, NSMenuItem
        main = NSMenu.alloc().init()
        app_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Focus Monitor", None, "")
        main.addItem_(app_item)
        edit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Edit", None, "")
        edit = NSMenu.alloc().initWithTitle_("Edit")
        for title, sel, key in [("Undo", "undo:", "z"), ("Redo", "redo:", "Z"),
                                ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                                ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")]:
            edit.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, sel, key))
        main.addItem_(edit_item)
        main.setSubmenu_forItem_(edit, edit_item)
        NSApp.setMainMenu_(main)

    def show(self, fragment: str = "day"):
        if self.window is None:
            self._install_edit_menu()
            rect = NSMakeRect(0, 0, 1180, 820)
            style = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | NSResizableWindowMask
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, style, NSBackingStoreBuffered, False)
            self.window.setTitle_("Focus Monitor")
            self.window.setReleasedWhenClosed_(False)
            # Open on the Space the user is on right now, instead of switching desktops
            # to wherever the window last lived (NSWindowCollectionBehaviorMoveToActiveSpace).
            self.window.setCollectionBehavior_(1 << 1)
            self.window.center()
            self.webview = WKWebView.alloc().initWithFrame_configuration_(rect, WKWebViewConfiguration.alloc().init())
            self.webview.setAutoresizingMask_(18)  # width + height sizable
            self.window.setContentView_(self.webview)
        url = NSURL.URLWithString_(f"{self.base_url}/#{fragment}")
        self.webview.loadRequest_(NSURLRequest.requestWithURL_(url))
        NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def is_open(self) -> bool:
        return self.window is not None and bool(self.window.isVisible())

    def fragment(self) -> str:
        """The tab currently shown ('day', 'review', ...), read from the webview's URL."""
        try:
            url = self.webview.URL() if self.webview is not None else None
            frag = str(url.fragment()) if url is not None and url.fragment() else ""
        except Exception:
            frag = ""
        return frag or "day"


class FocusMenuBar(rumps.App):
    def __init__(self, cfg: Config, use_claude: bool = True):
        self._activity = _stay_awake()
        self._refreshing = threading.Lock()
        self.icons = menu_icons()
        super().__init__("Focus Monitor", icon=self.icons["watching"], template=True, quit_button=None)
        self.cfg = cfg
        self.base_url = f"http://{cfg.web_host}:{cfg.web_port}"
        self.dashboard = DashboardWindow(self.base_url)
        self.collector = Collector(cfg)
        self.classifier = ClassifierService(cfg, use_claude=use_claude)
        self.threads = {
            "collector": threading.Thread(target=self.collector.run, daemon=True, name="collector"),
            "classifier": threading.Thread(target=self.classifier.run, daemon=True, name="classifier"),
            "web": threading.Thread(target=self._serve, daemon=True, name="web"),
        }
        self.status_item = rumps.MenuItem("Starting…")
        self.focus_item = rumps.MenuItem("")
        self.events_item = rumps.MenuItem("")
        self.queue_item = rumps.MenuItem("")
        self.spend_item = rumps.MenuItem("")
        self.pause_item = rumps.MenuItem("Pause monitoring", callback=self.toggle_pause)
        self.menu = [
            rumps.MenuItem("Open Dashboard", callback=lambda _: self.dashboard.show("day")),
            rumps.MenuItem("Review Uncertain Switches", callback=lambda _: self.dashboard.show("review")),
            rumps.MenuItem("Effort Comparison", callback=lambda _: self.dashboard.show("efforts")),
            None,
            self.status_item, self.focus_item, self.events_item, self.queue_item, self.spend_item, None,
            self.pause_item,
            rumps.MenuItem("Set Anthropic API Key…", callback=self.set_api_key),
            rumps.MenuItem("Set Toggl API Token…", callback=self.set_toggl_token),
            rumps.MenuItem("Open Data Folder", callback=lambda _: subprocess.run(["open", str(DATA_DIR)])),
            None,
            rumps.MenuItem("Quit Focus Monitor", callback=lambda _: rumps.quit_application()),
        ]
        for t in self.threads.values():
            t.start()
        self.refresh(None)
        rumps.Timer(self.refresh, 30).start()
        self._reopen_dashboard_after_restart()

    def _serve(self):
        import uvicorn
        from . import web
        web.restart_hook = self.request_restart
        uvicorn.run(web.app, host=self.cfg.web_host, port=self.cfg.web_port, log_level="warning")

    # -- dev restart: POST /api/restart -> exit with RESTART_EXIT_CODE -> the .app launcher relaunches us
    def request_restart(self) -> dict:
        """Called on the web thread. Remembers an open dashboard so the relaunch can bring it back,
        then exits shortly after (so the HTTP response still gets out)."""
        reopen = self.dashboard.is_open()
        if reopen:
            REOPEN_FILE.write_text(json.dumps({"fragment": self.dashboard.fragment(), "ts": time.time()}))
        else:
            REOPEN_FILE.unlink(missing_ok=True)
        log.info("restart requested (dashboard open=%s)", reopen)
        threading.Timer(0.3, lambda: os._exit(RESTART_EXIT_CODE)).start()
        return {"reopen": reopen}

    def _reopen_dashboard_after_restart(self):
        try:
            info = json.loads(REOPEN_FILE.read_text())
            REOPEN_FILE.unlink()
        except (FileNotFoundError, ValueError):
            return
        if time.time() - float(info.get("ts", 0)) > 300:
            return  # stale marker from some earlier run
        fragment = str(info.get("fragment") or "day")
        deadline = time.time() + 30

        def poll(timer):
            # Wait for uvicorn to accept connections; a load before that shows an error page.
            try:
                urllib.request.urlopen(f"{self.base_url}/api/labels", timeout=1).close()
            except Exception:
                if time.time() < deadline:
                    return
                log.warning("dashboard not up 30 s after restart; not reopening")
            else:
                log.info("reopening dashboard at #%s after restart", fragment)
                self.dashboard.show(fragment)
            timer.stop()

        rumps.Timer(poll, 0.5).start()

    def _ask_secret(self, title: str, info: str, placeholder: str) -> str | None:
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(info)
        field = NSSecureTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 360, 24))
        field.setPlaceholderString_(placeholder)
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        alert.window().setInitialFirstResponder_(field)
        NSApp.activateIgnoringOtherApps_(True)
        if alert.runModal() != NSAlertFirstButtonReturn:
            return None
        return str(field.stringValue()).strip() or None

    def set_toggl_token(self, _):
        tok = self._ask_secret("Toggl Track API token",
                               "Profile → Profile settings → API Token on track.toggl.com. Your time entries "
                               "are used as a statement of what you intended to be doing.", "32 hex characters")
        if not tok:
            return
        from . import toggl
        try:
            toggl._get("/me", tok)
        except Exception as e:
            rumps.alert("Token rejected", f"Toggl did not accept this token:\n{e}")
            return
        keychain_set("toggl", tok)
        os.environ["TOGGL_API_TOKEN"] = tok
        self.classifier.toggl.last_sync = 0.0
        rumps.alert("Toggl connected", "Time entries will sync every couple of minutes.")

    def set_api_key(self, _):
        key = self._ask_secret("Anthropic API key",
                               "Stored in your login Keychain (service \"FocusMonitor\"). Claude Opus 5 will "
                               "classify switches once a valid key is set.", "sk-ant-…")
        if not key:
            return
        try:
            import anthropic
            anthropic.Anthropic(api_key=key).models.retrieve(self.cfg.claude_model)
        except Exception as e:
            rumps.alert("Key rejected", f"The API did not accept this key:\n{e}")
            return
        keychain_set("anthropic", key)
        os.environ["ANTHROPIC_API_KEY"] = key
        try:
            from .classifiers import ClaudeVisionClassifier
            self.classifier.claude = ClaudeVisionClassifier(self.cfg)
            self.classifier._review_count = -1  # force few-shot refresh on next batch
        except Exception as e:
            rumps.alert("Saved, but Claude failed to start", str(e))
            return
        self.refresh(None)
        try:
            rumps.notification("Focus Monitor", "API key saved", "Claude classification is on.")
        except Exception:
            rumps.alert("API key saved", "Claude classification is on.")

    def toggle_pause(self, _):
        self.collector.paused = not self.collector.paused
        self.pause_item.title = "Resume monitoring" if self.collector.paused else "Pause monitoring"
        self.refresh(None)

    def refresh(self, _):
        # The database work runs off the main thread so the menu always opens instantly;
        # only the resulting titles are applied on the main thread.
        if self._refreshing.locked():
            return
        threading.Thread(target=self._refresh_worker, daemon=True, name="menu-refresh").start()

    def _refresh_worker(self):
        with self._refreshing:
            try:
                icon, titles = self._menu_state()
            except Exception:
                log.exception("menu refresh failed")
                return
        AppHelper.callAfter(self._apply_menu_state, icon, titles)

    def _menu_state(self) -> tuple[str, dict[str, str]]:
        alive = {k: t.is_alive() for k, t in self.threads.items()}
        stale = time.time() - self.collector.last_tick > 30
        if self.collector.paused:
            icon, status = "paused", "Paused"
        elif not alive["collector"] or stale:
            icon, status = "trouble", "Collector not responding"
        else:
            icon, status = "watching", "Watching"
        if not alive["classifier"]:
            status += " · classifier stopped"
        if not self.classifier.claude:
            status += " · Claude off (no API key)"
        conn = db.connect()
        try:
            m = stats.daily_metrics(conn, logical_date())
            pending = conn.execute("SELECT COUNT(*) FROM switches WHERE status='pending'").fetchone()[0]
            uncertain = conn.execute("""SELECT COUNT(*) FROM ensemble e LEFT JOIN reviews r ON r.switch_id=e.switch_id
                                        WHERE e.uncertain=1 AND r.switch_id IS NULL""").fetchone()[0]
            if self.cfg.count_events:
                events = (f"{m['interruption_count']} interruptions · {m['distraction_count']} distractions "
                          f"({_fmt_min(m['distracted_min'])})")
            else:
                spans = stats.spans_and_events(stats.labelled_segments(conn, *stats.day_bounds(logical_date())))
                events = f"{len(spans['focus_spans'])} focus spans today"
        finally:
            conn.close()
        return icon, {
            "status": status,
            "focus": f"Today: longest focus {_fmt_min(m['longest_focus_min'])}, total {_fmt_min(m['total_focus_min'])}",
            "events": events,
            "queue": f"{uncertain} awaiting review · {pending} unclassified",
            "spend": f"Claude spend today ${m['spend_usd']:.2f} / ${self.cfg.daily_budget_usd:.0f}",
        }

    def _apply_menu_state(self, icon: str, titles: dict[str, str]):
        if self.icon != self.icons[icon]:
            self.icon = self.icons[icon]
        self.status_item.title = titles["status"]
        self.focus_item.title = titles["focus"]
        self.events_item.title = titles["events"]
        self.queue_item.title = titles["queue"]
        self.spend_item.title = titles["spend"]


def run_app(use_claude: bool = True):
    FocusMenuBar(load_config(), use_claude=use_claude).run()

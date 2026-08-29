"""Menu bar app: an eye in the menu bar while the monitor is running, plus a native dashboard window."""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
import threading
import time

import rumps
import os

from AppKit import (NSAlert, NSAlertFirstButtonReturn, NSApp, NSBackingStoreBuffered, NSClosableWindowMask,
                    NSMakeRect, NSMiniaturizableWindowMask, NSResizableWindowMask, NSSecureTextField,
                    NSTitledWindowMask, NSWindow)
from Foundation import NSURL, NSURLRequest
from WebKit import WKWebView, WKWebViewConfiguration

from . import db, stats
from .classify import ClassifierService
from .collector import Collector
from .config import DATA_DIR, Config, keychain_set, load_config
from .icons import menu_icons

log = logging.getLogger("menubar")


def _fmt_min(m: float) -> str:
    return f"{int(m // 60)}h {int(m % 60)}m" if m >= 60 else f"{int(m)}m"


class DashboardWindow:
    """A native window hosting the local dashboard in a WKWebView."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.window = None
        self.webview = None

    def show(self, fragment: str = "day"):
        if self.window is None:
            rect = NSMakeRect(0, 0, 1180, 820)
            style = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | NSResizableWindowMask
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, style, NSBackingStoreBuffered, False)
            self.window.setTitle_("Focus Monitor")
            self.window.setReleasedWhenClosed_(False)
            self.window.center()
            self.webview = WKWebView.alloc().initWithFrame_configuration_(rect, WKWebViewConfiguration.alloc().init())
            self.webview.setAutoresizingMask_(18)  # width + height sizable
            self.window.setContentView_(self.webview)
        url = NSURL.URLWithString_(f"{self.base_url}/#{fragment}")
        self.webview.loadRequest_(NSURLRequest.requestWithURL_(url))
        NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)


class FocusMenuBar(rumps.App):
    def __init__(self, cfg: Config, use_claude: bool = True):
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

    def _serve(self):
        import uvicorn
        from .web import app
        uvicorn.run(app, host=self.cfg.web_host, port=self.cfg.web_port, log_level="warning")

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
        try:
            alive = {k: t.is_alive() for k, t in self.threads.items()}
            stale = time.time() - self.collector.last_tick > 30
            if self.collector.paused:
                icon, status = "paused", "Paused"
            elif not alive["collector"] or stale:
                icon, status = "trouble", "Collector not responding"
            else:
                icon, status = "watching", "Watching"
            if self.icon != self.icons[icon]:
                self.icon = self.icons[icon]
            if not alive["classifier"]:
                status += " · classifier stopped"
            if not self.classifier.claude:
                status += " · Claude off (no API key)"
            self.status_item.title = status
            conn = db.connect()
            m = stats.daily_metrics(conn, dt.date.today())
            pending = conn.execute("SELECT COUNT(*) FROM switches WHERE status='pending'").fetchone()[0]
            uncertain = conn.execute("""SELECT COUNT(*) FROM ensemble e LEFT JOIN reviews r ON r.switch_id=e.switch_id
                                        WHERE e.uncertain=1 AND r.switch_id IS NULL""").fetchone()[0]
            self.focus_item.title = f"Today: longest focus {_fmt_min(m['longest_focus_min'])}, total {_fmt_min(m['total_focus_min'])}"
            if self.cfg.count_events:
                self.events_item.title = (f"{m['interruption_count']} interruptions · {m['distraction_count']} distractions "
                                          f"({_fmt_min(m['distracted_min'])})")
            else:
                self.events_item.title = f"{len(stats.spans_and_events(stats.labelled_segments(conn, *stats.day_bounds(dt.date.today())))['focus_spans'])} focus spans today"
            self.queue_item.title = f"{uncertain} awaiting review · {pending} unclassified"
            self.spend_item.title = f"Claude spend today ${m['spend_usd']:.2f} / ${self.cfg.daily_budget_usd:.0f}"
            conn.close()
        except Exception:
            log.exception("menu refresh failed")


def run_app(use_claude: bool = True):
    FocusMenuBar(load_config(), use_claude=use_claude).run()

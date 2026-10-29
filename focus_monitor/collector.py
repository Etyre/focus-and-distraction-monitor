"""Watches the active window, records segments/switches, and takes screenshots.

Requires macOS permissions for the host terminal/python:
  * Screen Recording  (screencapture + window titles)
  * Accessibility / Automation for System Events and your browser (AppleScript)
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from . import db
from .config import (ARCHIVE_DIR, CHROMIUM_APPS, DATA_DIR, HEARTBEAT_PATH, PRIVATE_MARK, SAFARI_APPS,
                     SCREENSHOT_DIR, Config, load_config)

log = logging.getLogger("collector")

FRONTMOST_SCRIPT = '''
tell application "System Events"
    set p to first application process whose frontmost is true
    set appName to name of p
    set winTitle to ""
    try
        set winTitle to name of front window of p
    end try
end tell
return appName & linefeed & winTitle
'''

CHROMIUM_URL_SCRIPT = '''
tell application "{app}"
    if (count of windows) > 0 then
        set t to active tab of front window
        return (URL of t) & linefeed & (title of t)
    end if
end tell
return ""
'''

SAFARI_URL_SCRIPT = '''
tell application "Safari"
    if (count of windows) > 0 then
        set d to current tab of front window
        return (URL of d) & linefeed & (name of d)
    end if
end tell
return ""
'''


def osascript(script: str, timeout: float = 3.0) -> str:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            log.debug("osascript error: %s", r.stderr.strip())
            return ""
        return r.stdout.rstrip("\n")
    except subprocess.TimeoutExpired:
        return ""


def idle_seconds() -> float:
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem", "-d", "4"], capture_output=True,
                             text=True, timeout=3).stdout
        m = re.search(r'"HIDIdleTime" = (\d+)', out)
        return int(m.group(1)) / 1e9 if m else 0.0
    except Exception:
        return 0.0


def screen_locked() -> bool:
    try:
        import Quartz
        d = Quartz.CGSessionCopyCurrentDictionary() or {}
        return bool(d.get("CGSSessionScreenIsLocked", 0))
    except Exception:
        return False


@dataclass(frozen=True)
class WindowState:
    app: str
    title: str
    url: str

    @property
    def domain(self) -> str:
        if not self.url:
            return ""
        host = urlparse(self.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host


def current_window() -> WindowState | None:
    out = osascript(FRONTMOST_SCRIPT)
    if not out:
        return None
    app, _, title = out.partition("\n")
    url = ""
    if app in CHROMIUM_APPS:
        r = osascript(CHROMIUM_URL_SCRIPT.format(app=app))
        if r:
            url, _, t2 = r.partition("\n")
            title = t2 or title
    elif app in SAFARI_APPS:
        r = osascript(SAFARI_URL_SCRIPT)
        if r:
            url, _, t2 = r.partition("\n")
            title = t2 or title
    return WindowState(app=app.strip(), title=title.strip()[:300], url=url.strip()[:1000])


def private_displays(cfg: Config) -> set[int] | None:
    """Indices of displays (CGGetActiveDisplayList order, matching screencapture's file order)
    that currently show a window of a private app, e.g. Signal. None means we could not tell -
    callers must then treat every display as private. Occluded windows count too: that costs an
    archive copy at worst, while missing one would keep a disappearing chat for good."""
    if not cfg.private_apps:
        return set()
    try:
        import Quartz
        wins = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID)
        rects = []
        for w in wins:
            b = w.get("kCGWindowBounds")
            if w.get("kCGWindowOwnerName") in cfg.private_apps and b and b["Width"] * b["Height"] >= 10000:
                rects.append((b["X"], b["Y"], b["X"] + b["Width"], b["Y"] + b["Height"]))
        if not rects:
            return set()
        err, ids, cnt = Quartz.CGGetActiveDisplayList(8, None, None)
        if err:
            return None
        hit: set[int] = set()
        for i in range(cnt or 0):
            d = Quartz.CGDisplayBounds(ids[i])
            dx0, dy0 = d.origin.x, d.origin.y
            dx1, dy1 = dx0 + d.size.width, dy0 + d.size.height
            if any(x0 < dx1 and x1 > dx0 and y0 < dy1 and y1 > dy0 for x0, y0, x1, y1 in rects):
                hit.add(i)
        return hit
    except Exception as e:
        log.warning("private window check failed (treating every display as private): %s", e)
        return None


def take_screenshots(cfg: Config, tag: str) -> list[str]:
    """Capture EVERY display (main monitor first, then externals). Returns saved file names -
    empty on failure. screencapture writes one file per attached display and ignores the
    extra paths, so passing four covers any realistic setup.

    Each capture is written twice: full size under SCREENSHOT_DIR (pruned after the retention
    period) and a small copy under ARCHIVE_DIR (kept for good). A display showing a private app
    gets PRIVATE_MARK in its filename, and cleanup removes that capture from BOTH tiers."""
    ts_ms = int(time.time() * 1000)
    priv = private_displays(cfg)
    tmps = [SCREENSHOT_DIR / f"{ts_ms}_{tag}_d{i}.png" for i in range(4)]
    out: list[str] = []
    try:
        subprocess.run(["screencapture", "-x"] + [str(t) for t in tmps], capture_output=True, timeout=10)
        for i, tmp in enumerate(tmps):
            if not tmp.exists():
                continue
            name = tmp.stem + (PRIVATE_MARK if priv is None or i in priv else "") + ".jpg"
            with Image.open(tmp) as im:
                im = im.convert("RGB")
                for d, width, quality in ((SCREENSHOT_DIR, cfg.screenshot_max_width, cfg.screenshot_quality),
                                          (ARCHIVE_DIR, cfg.archive_max_width, cfg.archive_quality)):
                    small = im
                    if im.width > width:
                        small = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
                    small.save(d / name, "JPEG", quality=quality, optimize=True)
            out.append(name)
    except Exception as e:
        log.warning("screenshot failed: %s", e)
    finally:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
    return out


def take_screenshot(cfg: Config, tag: str) -> str | None:
    shots = take_screenshots(cfg, tag)
    return shots[0] if shots else None


def active_display_index(app_name: str) -> int:
    """Which display holds the frontmost app's window (0 = main/laptop). The primary
    screenshot must be of the display the person is actually working on - with an external
    monitor attached, that is often NOT the laptop screen."""
    try:
        import Quartz
        wins = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID)
        b = None
        for w in wins:
            if w.get("kCGWindowLayer", 1) == 0 and w.get("kCGWindowOwnerName") == app_name:
                b = w.get("kCGWindowBounds")
                break
        if not b:
            return 0
        cx, cy = b["X"] + b["Width"] / 2, b["Y"] + b["Height"] / 2
        err, ids, cnt = Quartz.CGGetActiveDisplayList(8, None, None)
        for i in range(cnt or 0):
            db_ = Quartz.CGDisplayBounds(ids[i])
            if (db_.origin.x <= cx <= db_.origin.x + db_.size.width
                    and db_.origin.y <= cy <= db_.origin.y + db_.size.height):
                return i
        return 0
    except Exception:
        return 0


def cleanup_screenshots(cfg: Config) -> int:
    """Delete full-size captures (and any PNG left behind by a crash mid-conversion) older than
    the retention period, plus the archive copies of private captures. Other archive copies are
    permanent."""
    cutoff = time.time() - cfg.screenshot_retention_days * 86400
    n = 0
    for pattern, d in (("*.jpg", SCREENSHOT_DIR), ("*.png", SCREENSHOT_DIR), (f"*{PRIVATE_MARK}.jpg", ARCHIVE_DIR)):
        for p in d.glob(pattern):
            if p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
    return n


class Collector:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.conn = db.connect()
        self.current_id: int | None = None
        self.current_state: WindowState | None = None
        self.current_idle = False
        self.candidate: tuple[WindowState, float] | None = None
        self.last_shot_time = 0.0
        self.paused = False
        self.last_tick = 0.0
        self.last_real_id: int | None = None  # last non-idle, non-neutral segment
        self.last_real_state: WindowState | None = None
        self.short_idle_start: float | None = None  # start of a current pause in input, if any
        self.current_start: float = 0.0
        self._recover()

    def _recover(self):
        # Close any segment left open by a previous crash.
        self.conn.execute("UPDATE segments SET end = start + 1 WHERE end IS NULL")

    # -- segment management ---------------------------------------------------------------
    def _close_current(self, at: float):
        if self.current_id is not None:
            self.conn.execute("UPDATE segments SET end=? WHERE id=?", (at, self.current_id))

    def _open_segment(self, state: WindowState | None, at: float, idle: bool, neutral: bool = False) -> int:
        shots = [] if (idle or neutral) else take_screenshots(self.cfg, "first")
        di = active_display_index(state.app) if (state and shots) else 0
        shot = (shots[di] if di < len(shots) else shots[0]) if shots else None
        cur = self.conn.execute(
            "INSERT INTO segments(start, app, title, url, domain, first_screenshot, last_screenshot, idle, neutral)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (at, state.app if state else "__idle__", state.title if state else "",
             state.url if state else "", state.domain if state else "", shot, shot, int(idle), int(neutral)),
        )
        seg_id = cur.lastrowid
        for i, name in enumerate(shots):
            self.conn.execute("INSERT INTO segment_shots(segment_id, ts, path, display) VALUES (?,?,?,?)",
                              (seg_id, at, name, i))
        self.last_shot_time = at
        return seg_id

    def _switch_to(self, state: WindowState | None, at: float, idle: bool):
        neutral = bool(state) and not idle and self.cfg.is_neutral(state.app, state.title, state.url)
        with db.tx(self.conn):
            self._close_current(at)
            new_id = self._open_segment(state, at, idle, neutral)
            # Only real window switches need classification: not idle transitions, not neutral windows
            # (presence checks etc.), and a switch out of a neutral window is attributed to the last
            # real segment before it - so the neutral window is transparent.
            same_as_before = (self.last_real_state is not None and state == self.last_real_state)
            if (not idle and not neutral and self.last_real_id is not None and not self.current_idle
                    and not same_as_before):
                self.conn.execute(
                    "INSERT INTO switches(ts, from_segment, to_segment) VALUES (?,?,?)",
                    (at, self.last_real_id, new_id))
        if idle:
            self.last_real_id, self.last_real_state = None, None
        elif not neutral:
            self.last_real_id, self.last_real_state = new_id, state
        self.current_id, self.current_state, self.current_idle = new_id, state, idle
        self.current_start = at
        self.candidate = None
        log.info("segment %s: %s | %s | %s%s", new_id, "IDLE" if idle else state.app,
                 state.title[:60] if state else "", state.domain if state else "", " [neutral]" if neutral else "")

    def tick(self):
        now = time.time()
        gap = now - self.last_tick if self.last_tick else 0.0  # large gap => the process was suspended (system sleep)
        self.last_tick = now
        try:
            HEARTBEAT_PATH.touch()  # liveness signal for the launcher's watchdog
        except OSError:
            pass
        if self.current_id is not None and gap < self.cfg.idle_seconds:
            # Flush the open segment's end every tick: an unclean kill (dev restart, crash)
            # must cost at most one poll interval - not the whole segment. (2026-09-02: seven
            # dev restarts during long Terminal stretches each truncated an open ~25-minute
            # segment to a 1-second sliver via _recover.)
            self.conn.execute("UPDATE segments SET end=? WHERE id=?", (now, self.current_id))
        if self.paused:
            if not self.current_idle:  # record the pause as away time so spans break cleanly
                self._switch_to(None, now, True)
            return
        idle_for = idle_seconds()
        locked = screen_locked()
        # Time with no activity, accounting for a suspended process during sleep.
        inactive_for = max(idle_for, gap)
        if inactive_for >= self.cfg.idle_seconds or locked:
            if not self.current_idle:
                # Away: the span ends when the inactivity began, not when we noticed it.
                start = now if (locked and inactive_for < self.cfg.idle_seconds) else now - inactive_for
                self._switch_to(None, max(start, self.current_start), True)
            self.short_idle_start = None
            return
        # Track short pauses in keyboard/mouse input (logged, but they don't break spans).
        if idle_for >= self.cfg.short_inactivity_seconds:
            if self.short_idle_start is None:
                self.short_idle_start = now - idle_for
        elif self.short_idle_start is not None:
            self.conn.execute("INSERT INTO inactivity(start, end) VALUES (?,?)",
                              (self.short_idle_start, now - idle_for))
            log.info("short break: %.0fs", now - idle_for - self.short_idle_start)
            self.short_idle_start = None
        state = current_window()
        if state is None or state.app in self.cfg.ignore_apps:
            return
        if self.current_idle or self.current_state is None:
            self._switch_to(state, now, False)
            return
        if state == self.current_state:
            self.candidate = None
            if now - self.last_shot_time >= self.cfg.refresh_screenshot_seconds:
                shots = take_screenshots(self.cfg, "last")
                if shots:
                    di = active_display_index(self.current_state.app) if self.current_state else 0
                    self.conn.execute("UPDATE segments SET last_screenshot=? WHERE id=?",
                                      (shots[di] if di < len(shots) else shots[0], self.current_id))
                    for i, name in enumerate(shots):
                        self.conn.execute("INSERT INTO segment_shots(segment_id, ts, path, display) VALUES (?,?,?,?)",
                                          (self.current_id, now, name, i))
                self.last_shot_time = now
            return
        # Different window: debounce
        if self.candidate is None or self.candidate[0] != state:
            self.candidate = (state, now)
        elif now - self.candidate[1] >= self.cfg.min_segment_seconds:
            self._switch_to(state, self.candidate[1], False)

    def run(self):
        log.info("collector started; data dir %s", DATA_DIR)
        last_cleanup = 0.0
        while True:
            try:
                self.tick()
                if time.time() - last_cleanup > 3600:
                    n = cleanup_screenshots(self.cfg)
                    # Archived captures stay referenced; private ones are forgotten with their files.
                    cutoff = time.time() - self.cfg.screenshot_retention_days * 86400
                    like = f"%{PRIVATE_MARK}.jpg"
                    with db.tx(self.conn):
                        self.conn.execute("DELETE FROM segment_shots WHERE ts < ? AND path LIKE ?", (cutoff, like))
                        for col in ("first_screenshot", "last_screenshot"):
                            self.conn.execute(f"UPDATE segments SET {col}=NULL WHERE start < ? AND {col} LIKE ?",
                                              (cutoff, like))
                    if n:
                        log.info("deleted %d old screenshots", n)
                    last_cleanup = time.time()
            except Exception:
                log.exception("collector tick failed")
            time.sleep(self.cfg.poll_interval)

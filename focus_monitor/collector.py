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

# A stretch without input is only held open as a call (see Collector._call_until) up to this
# long. Nobody sits four hours in a call without touching the Mac once; a camera left running
# overnight by a stuck app would otherwise never become away.
CALL_MAX_S = 4 * 3600.0

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

# Every window of a browser in one call: the window's title, its showing tab's title and address,
# and its bounds, as four lists in the browser's own window order (`front window` is the first).
# See browser_tab for why. Measured 2026-11-21 on Chrome with 83 windows open: 0.11 s, the same
# as the one-window script above (a `repeat` over the windows took 5.6 s).
_SEP_ITEM, _SEP_LIST = "\x1f", "\x1e"
_WINDOWS_TAIL = '''
set text item delimiters to (ASCII character 31)
return (a as text) & (ASCII character 30) & (b as text) & (ASCII character 30) & (c as text) & (ASCII character 30) & (d as text)
'''
CHROMIUM_WINDOWS_SCRIPT = '''
tell application "{app}"
    set a to title of windows
    set b to title of active tab of windows
    set c to URL of active tab of windows
    set d to bounds of windows
end tell''' + _WINDOWS_TAIL
SAFARI_WINDOWS_SCRIPT = '''
tell application "Safari"
    set a to name of windows
    set b to name of current tab of windows
    set c to URL of current tab of windows
    set d to bounds of windows
end tell''' + _WINDOWS_TAIL


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


# A tab title's leading unread counter or status mark: "(1) Notifications / X", "(20+) Home",
# "● Draft", "🔴 Live". See same_window.
_TITLE_COUNTER = re.compile(r"^(?:\s*(?:\(\d{1,4}\+?\)|[●•◦○◉⚫⚪🔴🟠🟡🟢🔵🟣🔔🔕]\ufe0f?))+\s*")


def _title_key(title: str) -> str:
    return _TITLE_COUNTER.sub("", title) or title   # a title that is nothing but a counter stays as it is


def same_window(a: WindowState | None, b: WindowState | None) -> bool:
    """Whether two readings are the same window for the purpose of segments and switches: the
    same app, address and title, with a leading unread counter or status mark left out of the
    title. Sites count notifications in the tab title, so "(1) Notifications / X" turning into
    "Notifications / X" started a new segment and a switch with nobody touching anything (294
    such pairs by 2026-11-20, on x.com and tinder.com). The stored title is still the one seen
    when the segment started."""
    if a is None or b is None:
        return a is b
    return a.app == b.app and a.url == b.url and _title_key(a.title) == _title_key(b.title)


def _window_list_title(app: str) -> str:
    """Title of the app's frontmost normal window, from the window list (needs only Screen
    Recording, which the app already holds for screenshots). The list is front to back."""
    try:
        import Quartz
        for w in Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID):
            if w.get("kCGWindowOwnerName") == app and w.get("kCGWindowLayer") == 0 and w.get("kCGWindowName"):
                return str(w["kCGWindowName"])
    except Exception:
        log.exception("window-list title lookup failed")
    return ""


# ---- which of a browser's windows is the one on screen (2026-11-21) -------------------------
# Chrome's `front window` is the first window in Chrome's OWN ordering, which is not always the
# window on screen. On 2026-11-21, 11 segments from 13:01:49 to 15:13:29 (37381 ... 37436) were
# stored as Chrome on a food-delivery page last loaded the day before, while the captures show
# other pages, and tab changes in the window really in front opened no new segment (37421 is one
# 510-second segment over several pages). So: take the browser's frontmost window from the
# SYSTEM's on-screen window list, find it among the browser's own windows by title and bounds,
# and read that window's tab.

def _squash(t: str) -> str:
    return " ".join((t or "").split())


def _title_rank(shown: str, win_title: str, tab_title: str) -> tuple[int, int]:
    """How well a title from the system's window list fits one browser window: (kind, length of
    the title that fitted); (0, 0) is no fit. The system's title is the tab's title, sometimes
    followed by " - Google Chrome", a profile name or " - Audio playing", so the tab's title
    being a prefix of it counts; the longest such title wins ("Inbox" also begins "Inbox (2) -
    Mail"). Chrome's own `title of window` shortens a long title to its first and last ~30
    characters around a "…" (25 of 83 windows on 2026-11-21): both ends must then be present."""
    s, best = _squash(shown), (0, 0)
    if not s:
        return best
    for t in (_squash(win_title), _squash(tab_title)):
        if not t:
            continue
        head, cut, tail = t.partition("…")
        if s == t:
            r = (3, len(t))
        elif s.startswith(t):
            r = (2, len(t))
        elif cut and head and tail and s.startswith(head) and tail in s[len(head):]:
            r = (1, len(t))
        else:
            continue
        best = max(best, r)
    return best


def _same_bounds(a, b) -> bool:
    return a is not None and b is not None and all(abs(x - y) <= 2 for x, y in zip(a, b))


def pick_window(onscreen: list, wins: list) -> int | None:
    """Which of the browser's windows is the one on screen.
    `onscreen`: the browser's windows from the system's on-screen list, front to back, as
    (title, (left, top, right, bottom)); the title is '' without Screen Recording.
    `wins`: the browser's own list, in its order, as (window title, tab title, url, bounds).
    Returns an index into `wins`, or None when no on-screen window can be told apart (the caller
    then reads the browser's `front window`, as before 2026-11-21).
    The title decides; bounds decide between windows with the same title, and stand in when no
    title fits (it changed between the two readings, or there is none). Windows left equal on
    both are nearly always the same page open twice: the browser's own order picks. An on-screen
    window that fits nothing by title or bounds is not one of the listed windows (a popup): the
    next one back is tried."""
    for shown, bounds in onscreen:
        ranks = [_title_rank(shown, w[0], w[1]) for w in wins]
        top = max(ranks, default=(0, 0))
        cands = [i for i, r in enumerate(ranks) if r == top] if top > (0, 0) else list(range(len(wins)))
        placed = [i for i in cands if _same_bounds(wins[i][3], bounds)]
        if placed:
            return placed[0]
        if top > (0, 0):
            return cands[0]
    return None


def _parse_windows(out: str) -> list | None:
    """The four lists of a *_WINDOWS_SCRIPT as [(window title, tab title, url, bounds)]; None
    when the output is not four lists of matching length (no windows, a window with no tab, a
    browser without these terms): the caller falls back to the one-window script."""
    parts = [p.split(_SEP_ITEM) for p in out.split(_SEP_LIST)]
    if len(parts) != 4:
        return None
    a, b, c, d = parts
    try:
        nums = [int(x) for x in d]
    except ValueError:
        return None
    if not a or len(nums) != 4 * len(a) or len(b) != len(a) or len(c) != len(a):
        return None
    return [(a[i], b[i], c[i], tuple(nums[4 * i:4 * i + 4])) for i in range(len(a))]


def _onscreen_windows(app: str, limit: int = 4) -> list:
    """The app's normal windows on screen now, front to back: [(title, (left, top, right, bottom))].
    The window list is front to back; titles need Screen Recording, bounds do not."""
    out = []
    try:
        import Quartz
        for w in Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID) or []:
            b = w.get("kCGWindowBounds")
            if w.get("kCGWindowOwnerName") != app or w.get("kCGWindowLayer") != 0 or not b or not w.get("kCGWindowAlpha", 1):
                continue
            x, y = float(b.get("X", 0)), float(b.get("Y", 0))
            out.append((str(w.get("kCGWindowName") or ""),
                        (x, y, x + float(b.get("Width", 0)), y + float(b.get("Height", 0)))))
            if len(out) >= limit:
                break
    except Exception:
        log.exception("on-screen window list unavailable")
        return []
    return out


_pick_noted: dict = {}


def browser_tab(app: str, windows_script: str, front_script: str) -> tuple[str, str]:
    """(address, title) of the tab showing in the browser window that is on screen; ('', '')
    when it cannot be read. One osascript call, as before; a second (the old `front window`
    script) only when the all-windows one gave nothing usable. Never raises."""
    try:
        wins = _parse_windows(osascript(windows_script))
        if wins is not None:
            i = pick_window(_onscreen_windows(app), wins)
            # One log line when the answer's kind changes, not one per poll: how often the
            # on-screen window is not the browser's "front window" is what this fix rests on.
            kind = "none" if i is None else "front" if i == 0 else "other"
            if _pick_noted.get(app) != kind:
                _pick_noted[app] = kind
                log.info("%s: %s", app,
                         "no on-screen window matched; reading its front window" if i is None else
                         "the window on screen is its front window" if i == 0 else
                         f"the window on screen is its window {i + 1} of {len(wins)}, not its front window")
            _, tab_title, url, _ = wins[i or 0]   # wins[0] is `front window`
            return url, tab_title
    except Exception:
        log.exception("could not match %s's on-screen window; reading its front window", app)
    r = osascript(front_script)
    url, _, title = r.partition("\n")
    return url, title


def current_window(title_apps=()) -> WindowState | None:
    out = osascript(FRONTMOST_SCRIPT)
    if not out:
        return None
    app, _, title = out.partition("\n")
    app = app.strip()
    # System Events has never returned a window name here (every non-browser segment up to
    # 2026-11-17 has an empty title), so for apps where the title carries what the person is
    # working on - Agent Coding puts the active project in it - read it from the window list.
    # Limited to `title_apps`: a new title starts a new segment, and an app whose title ticks
    # (a running timer) would shred the timeline.
    if not title.strip() and app in title_apps:
        title = _window_list_title(app)
    url = ""
    if app in CHROMIUM_APPS:
        url, t2 = browser_tab(app, CHROMIUM_WINDOWS_SCRIPT.format(app=app), CHROMIUM_URL_SCRIPT.format(app=app))
        title = t2 or title
    elif app in SAFARI_APPS:
        url, t2 = browser_tab(app, SAFARI_WINDOWS_SCRIPT, SAFARI_URL_SCRIPT)
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
    """Delete private captures from both tiers after private_retention_days, and, when
    screenshot_retention_days is non-zero, full-size captures (and any PNG left behind by a crash
    mid-conversion) older than that. Other archive copies are permanent."""
    now = time.time()
    jobs = [(f"*{PRIVATE_MARK}.jpg", d, now - cfg.private_retention_days * 86400) for d in (SCREENSHOT_DIR, ARCHIVE_DIR)]
    if cfg.screenshot_retention_days > 0:
        cutoff = now - cfg.screenshot_retention_days * 86400
        jobs += [("*.jpg", SCREENSHOT_DIR, cutoff), ("*.png", SCREENSHOT_DIR, cutoff)]
    n = 0
    for pattern, d, cutoff in jobs:
        for p in d.glob(pattern):
            if p.exists() and p.stat().st_mtime < cutoff:
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
        # Kept across restarts: a queued restart or a crash relaunch must not quietly resume
        # recording (and taking captures) after the person paused it. (Until 2026-11-20 it did.)
        self.paused = db.kv_get(self.conn, "collector.paused") == "1"
        self.last_tick = 0.0
        self.last_real_id: int | None = None  # last non-idle, non-neutral segment
        self.last_real_state: WindowState | None = None
        self.short_idle_start: float | None = None  # start of a current pause in input, if any
        self.current_start: float = 0.0
        self.first_id: int | None = None   # the first segment this process opened (see _absorb_restart)
        # While the screen is locked: when input stopped before the lock (None when not locked).
        # Input on the lock screen itself (a key that wakes the display) does not move it.
        self.lock_idle_start: float | None = None
        # The first locked tick of a process that started locked and has opened no segment yet:
        # where its first segment begins if the person comes back before the away threshold.
        self.started_locked_at: float | None = None
        self.signals = None   # extra signals beside the captures (signals.py); started in run()
        # When the screen was first seen locked (None when not locked): microphone readings
        # after it do not count as a call (see _call_until).
        self.locked_since: float | None = None
        # The last input of a stretch that is being held open as a call instead of going away
        # (None when there is none); only so that the log says it once.
        self.call_hold_from: float | None = None
        self._recover()

    def _recover(self):
        # Close any segment left open by a previous crash.
        self.conn.execute("UPDATE segments SET end = start + 1 WHERE end IS NULL")

    # -- segment management ---------------------------------------------------------------
    def _close_current(self, at: float):
        if self.current_id is not None:
            self.conn.execute("UPDATE segments SET end=? WHERE id=?", (at, self.current_id))

    def set_paused(self, paused: bool):
        """Called from the menu (another thread), so it writes through a connection of its own."""
        self.paused = paused
        c = db.connect()
        try:
            db.kv_set(c, "collector.paused", "1" if paused else "0")
        finally:
            c.close()
        log.info("monitoring %s", "paused" if paused else "resumed")

    def _open_segment(self, state: WindowState | None, at: float, idle: bool, neutral: bool, shots: list) -> int:
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
        if self.signals is not None and not idle:
            self.signals.note_switch(seg_id, at)   # the click / key press / banner just before; never raises
        return seg_id

    def _switch_to(self, state: WindowState | None, at: float, idle: bool):
        neutral = bool(state) and not idle and self.cfg.is_neutral(state.app, state.title, state.url)
        # Captured BEFORE the transaction: screencapture and the JPEG encoding take a few hundred
        # ms, and holding the write lock that long made other threads' writes fail at once with
        # "database is locked" (question scoring, 2026-11-18).
        shots = [] if (idle or neutral) else take_screenshots(self.cfg, "first")
        prev_id = self.current_id
        with db.tx(self.conn):
            self._close_current(at)
            new_id = self._open_segment(state, at, idle, neutral, shots)
            if idle and prev_id is not None:
                db.move_late_shots(self.conn, prev_id, at, new_id)
            # Only real window switches need classification: not idle transitions, not neutral windows
            # (presence checks etc.), and a switch out of a neutral window is attributed to the last
            # real segment before it - so the neutral window is transparent.
            same_as_before = (self.last_real_state is not None and same_window(state, self.last_real_state))
            if (not idle and not neutral and self.last_real_id is not None and not self.current_idle
                    and not same_as_before):
                self.conn.execute(
                    "INSERT INTO switches(ts, from_segment, to_segment) VALUES (?,?,?)",
                    (at, self.last_real_id, new_id))
        if idle:
            self.last_real_id, self.last_real_state = None, None
        elif not neutral:
            self.last_real_id, self.last_real_state = new_id, state
        if self.first_id is None:
            self.first_id = new_id
        self.current_id, self.current_state, self.current_idle = new_id, state, idle
        self.current_start = at
        self.candidate = None
        self.started_locked_at = None
        log.info("segment %s: %s | %s | %s%s", new_id, "IDLE" if idle else state.app,
                 state.title[:60] if state else "", state.domain if state else "", " [neutral]" if neutral else "")

    def _go_away(self, at: float):
        self._switch_to(None, at, True)

    def _absorb_restart(self, start: float) -> list[int] | None:
        """The app was restarted while the person was already away: the segment it opened on
        starting lies wholly inside the absence (which began at `start`, before that segment).
        Without this the away time was clamped to begin at the restart, and the minutes between
        the last input and the restart stayed credited to the segment before (2 minutes on
        2026-11-19 21:01; a `restart --when-idle` does this every time the person stays away).
        Only handles the plain cases - the one segment opened since this process started, or no
        segment yet (a start with the screen locked opens none; 2026-11-20); anything else keeps
        the old clamp and returns None. That segment is kept as a zero-length row at the absence's
        start (nothing is deleted); its captures move to the away segment with the rest.

        Returns the ids of the EARLIER segments it changed, for the caller to pass to
        db.move_late_shots once the away segment exists: pulling a segment's end back left the
        captures taken after it on that segment (segment 36636 on 2026-11-20: 18 s long, holding
        4 captures from after its end). Two kinds: the segment that was open when input stopped
        (its end is pulled back to `start`), and any that began after input stopped and ended
        before the restart - a window that came forward by itself, or an away segment the old
        process had already opened. Those would stay as activity inside the away time, so they
        too become zero-length rows at `start`, like the restart segment."""
        if self.current_idle:
            return None
        if self.current_id is not None and (start >= self.current_start or self.first_id != self.current_id):
            return None
        below = self.current_id if self.current_id is not None else 2 ** 62
        with db.tx(self.conn):
            cut = [r[0] for r in self.conn.execute(
                "SELECT id FROM segments WHERE end>? AND start<=? AND id<?", (start, start, below))]
            inside = [r[0] for r in self.conn.execute(
                "SELECT id FROM segments WHERE start>? AND id<?", (start, below))]
            self.conn.execute("UPDATE segments SET end=? WHERE end>? AND start<=? AND id<?", (start, start, start, below))
            self.conn.execute("UPDATE segments SET start=?, end=? WHERE start>? AND id<?", (start, start, start, below))
            if self.current_id is not None:
                self.conn.execute("UPDATE segments SET start=?, end=? WHERE id=?", (start, start, self.current_id))
        self.current_start = start
        return cut + inside

    def _call_until(self, start: float, now: float) -> float | None:
        """When the call that ran through the stretch without input [start, now] was last on,
        or None when there was none: then the stretch is an absence from `start`, as it always
        was. A call with no key press or pointer move is not time away. On 2026-11-20 and 21 four
        stretches (idle segments 36874, 36948, 37428, 37435; 113.9 minutes) with the camera and
        the microphone on throughout were recorded as away, and the meeting spans around them cut
        short. The rule is signals.call_until: camera on for 80% of the stretch, or the
        microphone; read from `activity_samples`, which survive a restart in the middle of a
        call. Time nobody sampled (the Mac asleep) counts as off, so a stale reading cannot hold
        a segment open. With the screen locked only the camera counts from the lock on: a camera
        that is running is a call, while some apps keep the microphone open for hours.
        None too when signals are off, the query fails, or the stretch is over CALL_MAX_S."""
        if self.signals is None:
            return None
        if now - start > CALL_MAX_S:
            if self.call_hold_from is not None:
                log.warning("camera or microphone on for over %.0f h without any input: no longer treated as a call",
                            CALL_MAX_S / 3600)
            return None
        try:
            from .signals import call_until
            return call_until(self.conn, start, now, mic_until=self.locked_since)
        except Exception:
            log.exception("could not read camera / microphone samples; the idle rule runs without them")
            return None

    def tick(self):
        now = time.time()
        gap = now - self.last_tick if self.last_tick else 0.0  # large gap => the process was suspended (system sleep)
        self.last_tick = now
        try:
            HEARTBEAT_PATH.touch()  # liveness signal for the launcher's watchdog
            if time.time() - getattr(self, "_tz_checked", 0.0) > 60:
                # Days and the times shown to models follow the zone the Mac was in at the
                # time (config.local_dt); a change of zone is recorded here, once a minute.
                self._tz_checked = time.time()
                try:
                    from .config import note_timezone
                    z = note_timezone()
                    if z:
                        log.info("time zone is now %s", z)
                except Exception:
                    log.exception("could not record the time zone")
        except OSError:
            pass
        if self.current_id is not None and gap < self.cfg.idle_seconds:
            # Flush the open segment's end every tick: an unclean kill (dev restart, crash)
            # must cost at most one poll interval - not the whole segment. (2026-09-02: seven
            # dev restarts during long Terminal stretches each truncated an open ~25-minute
            # segment to a 1-second sliver via _recover.)
            self.conn.execute("UPDATE segments SET end=? WHERE id=?", (now, self.current_id))
        if self.signals is not None and not self.paused:
            self.signals.sample(now)   # input counts, audio/mic/camera, clicks, banners, page visits; never raises
        if self.paused:
            if not self.current_idle:  # record the pause as away time so spans break cleanly
                log.info("away: tracking is paused")
                self._switch_to(None, now, True)
            self.lock_idle_start = self.locked_since = self.call_hold_from = None
            return
        idle_for = idle_seconds()
        locked = screen_locked()
        # Time with no activity, accounting for a suspended process during sleep.
        inactive_for = max(idle_for, gap)
        # A locked screen is not "away" by itself (until 2026-11-20 it was, at once and from the
        # moment of the lock). On battery the display sleeps 2 minutes after the last input and
        # the session then reports itself locked, so every 2-to-10-minute pause became an away
        # segment and ended the focus span: 122 of the 190 away segments opened by the lock
        # branch in the logged period lasted under 10 minutes (median 322 s). Locked time is
        # time without input: it counts from the last input BEFORE the lock, and becomes away
        # only at idle_seconds, backdated like any other absence. Shorter, it is a short break.
        if locked:
            if self.lock_idle_start is None:
                self.lock_idle_start = now - inactive_for
                self.locked_since = now
                if self.current_id is None:
                    self.started_locked_at = now
            inactive_for = max(inactive_for, now - self.lock_idle_start)
        else:
            self.locked_since = None
        if not locked and self.lock_idle_start is not None:
            # First tick after the unlock. If the process slept through the lock (no locked tick
            # reached the threshold), the absence is still measured from before the lock.
            if not self.current_idle and now - idle_for - self.lock_idle_start >= self.cfg.idle_seconds:
                inactive_for = max(inactive_for, now - self.lock_idle_start)
            self.lock_idle_start = None
        ended_call = False
        if inactive_for < self.cfg.idle_seconds:
            self.call_hold_from = None   # input came back (or never stopped for long)
        elif not self.current_idle:
            # No input for the away threshold - but not away while a call is evidently going on
            # (see _call_until). The stretch then counts from when the camera and microphone
            # went off: the segment stays open, captures and window changes go on as usual
            # below, and ten minutes after the call ended with still no input this becomes an
            # absence that began when the call ended, not at the last input.
            call_end = self._call_until(now - inactive_for, now)
            if call_end is not None:
                if self.call_hold_from is None:
                    self.call_hold_from = now - inactive_for
                    log.info("not away: %.0fs since last input, but camera or microphone on (a call); screen %s",
                             inactive_for, "locked" if locked else "not locked")
                ended_call = True
                inactive_for = max(0.0, now - call_end)
        if inactive_for >= self.cfg.idle_seconds:
            if not self.current_idle:
                # Away: the span ends when the inactivity began, not when we noticed it.
                start = now - inactive_for
                if ended_call:
                    log.info("away: the call ended %.0fs ago and there has been no input since; screen %s",
                             inactive_for, "locked" if locked else "not locked")
                else:
                    log.info("away: %.0fs since last input, screen %s", inactive_for, "locked" if locked else "not locked")
                self.call_hold_from = None
                absorbed = self._absorb_restart(start)
                self._go_away(start if absorbed is not None else max(start, self.current_start))
                for sid in absorbed or ():
                    db.move_late_shots(self.conn, sid, start, self.current_id)
            self.short_idle_start = None
            return
        if locked:
            # Nothing to look at and nothing to capture while locked. The pause is carried as a
            # pending short break from when input stopped; the open segment continues.
            if not self.current_idle and self.short_idle_start is None:
                self.short_idle_start = self.lock_idle_start
            return
        # Track short pauses in keyboard/mouse input (logged, but they don't break spans).
        if idle_for >= self.cfg.short_inactivity_seconds:
            if self.short_idle_start is None:
                self.short_idle_start = now - idle_for
        elif self.short_idle_start is not None:
            # A pause that began with a lock can be shorter than a short break (locked for 20 s):
            # that one is not recorded.
            if now - idle_for - self.short_idle_start >= self.cfg.short_inactivity_seconds:
                self.conn.execute("INSERT INTO inactivity(start, end) VALUES (?,?)",
                                  (self.short_idle_start, now - idle_for))
                log.info("short break: %.0fs", now - idle_for - self.short_idle_start)
            self.short_idle_start = None
        state = current_window(self.cfg.title_apps)
        if state is None or state.app in self.cfg.ignore_apps:
            return
        if self.current_idle or self.current_state is None:
            # A process that started with the screen locked (a `restart --when-idle` fires after
            # 120 s without input, just when the display sleeps on battery) opened nothing while
            # locked. Back before the away threshold, its first segment starts where the process
            # began watching, so the pause sits inside a segment like any other short break
            # instead of leaving a hole in the record. The capture is still dated now.
            at = self.started_locked_at if (self.started_locked_at is not None and not self.current_idle) else now
            self._switch_to(state, at, False)
            if at != now:
                self.conn.execute("UPDATE segment_shots SET ts=? WHERE segment_id=?", (now, self.current_id))
                self.last_shot_time = now
            return
        if same_window(state, self.current_state):
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
        if self.candidate is None or not same_window(self.candidate[0], state):
            self.candidate = (state, now)
        elif now - self.candidate[1] >= self.cfg.min_segment_seconds:
            self._switch_to(state, self.candidate[1], False)

    def run(self):
        log.info("collector started; data dir %s", DATA_DIR)
        if self.cfg.collect_signals:
            try:
                from .signals import Sampler
                self.signals = Sampler(self.conn)
            except Exception:
                log.exception("extra signals could not be started; collection continues without them")
        last_cleanup = 0.0
        while True:
            try:
                self.tick()
                if time.time() - last_cleanup > 3600:
                    n = cleanup_screenshots(self.cfg)
                    # Archived captures stay referenced; private ones are forgotten with their files.
                    cutoff = time.time() - self.cfg.private_retention_days * 86400
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

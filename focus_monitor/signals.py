"""Extra signals about what is happening on the Mac, beside the window title and the captures
(Eli, 2026-11-19: "does macOS give additional information we can use to supplement the
screenshots?"). macOS never says WHY an app came forward, so the cause is reconstructed from
what can be observed:

  * input rates     - counts of key presses, clicks, scroll and pointer-move events, from the
                      system's own per-type event counters. Counts only: no key is ever read.
  * audio / mic / camera - whether any output device is playing, any input device is recording,
                      any camera is running (CoreAudio / CoreMediaIO "is running somewhere").
  * clicks          - when the click counter goes up, which app's window the pointer is over
                      (the Dock, a notification banner, another app...). With "was a key pressed
                      just before", this tells a Dock click from a keyboard switch from a link
                      clicked in another app. Stored per segment in `segment_arrivals`.
  * banners         - a Notification Center window above the desktop means a banner is showing.
                      It says THAT a notification appeared, not which app sent it.
  * page visits     - Chrome's History file records how each page was reached (typed, link,
                      link from another app, new tab from a page, reload, back/forward...).

None of this needs a permission beyond what the app already holds. Not collected, because each
needs Full Disk Access and could not be tested without it: which app sent a notification, and
whether a Focus mode (Do Not Disturb) is on.

`Sampler.sample()` is called from the collector's tick (every poll_interval) and
`Sampler.note_switch()` as it opens a segment; neither may raise. Storage: `activity_samples`
(10-second buckets), `ui_events` (clicks, banners), `segment_arrivals`, `page_visits`. `describe()` turns them into the short notes the evaluator reads on each line of
its activity log and the Captures card shows."""
from __future__ import annotations

import collections
import ctypes
import glob
import json
import logging
import os
import sqlite3
import struct
import threading
import time
from urllib.parse import urlparse

from . import db

log = logging.getLogger("signals")

BUCKET_S = 10.0
ARRIVE_BEFORE_S = 6.0     # a click this long before a segment's recorded start may be what caused it
ARRIVE_AFTER_S = 0.5      #  (the collector notices a switch up to one poll after it happened)
BANNER_BEFORE_S = 20.0    # a banner that showed this long before a switch is worth mentioning
VISITS_EVERY_S = 30.0
CHROME_EPOCH = 11644473600.0   # Chrome stores microseconds since 1601-01-01

# ---- input counters ----------------------------------------------------------------------


def _counters() -> dict | None:
    """Events of each type since login. No permission needed, and no event content is seen."""
    try:
        import Quartz
        s = Quartz.kCGEventSourceStateCombinedSessionState
        c = lambda t: int(Quartz.CGEventSourceCounterForEventType(s, t))
        return {"keys": c(Quartz.kCGEventKeyDown),
                "clicks": c(Quartz.kCGEventLeftMouseDown) + c(Quartz.kCGEventRightMouseDown),
                "scrolls": c(Quartz.kCGEventScrollWheel), "moves": c(Quartz.kCGEventMouseMoved)}
    except Exception:
        log.exception("event counters unavailable")
        return None


# ---- audio / microphone / camera ---------------------------------------------------------

class _Addr(ctypes.Structure):
    _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]


def _fcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


_ca = _cm = None


def _libs():
    global _ca, _cm
    if _ca is None:
        _ca = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        _cm = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreMediaIO.framework/CoreMediaIO")
    return _ca, _cm


def _audio_devices(ca) -> list[int]:
    a = _Addr(_fcc("dev#"), _fcc("glob"), 0)
    sz = ctypes.c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(ctypes.c_uint32(1), ctypes.byref(a), 0, None, ctypes.byref(sz)):
        return []
    arr = (ctypes.c_uint32 * (sz.value // 4))()
    if ca.AudioObjectGetPropertyData(ctypes.c_uint32(1), ctypes.byref(a), 0, None, ctypes.byref(sz), arr):
        return []
    return list(arr)


def _audio_u32(ca, dev: int, sel: str, scope: str = "glob") -> int:
    a = _Addr(_fcc(sel), _fcc(scope), 0)
    v, sz = ctypes.c_uint32(0), ctypes.c_uint32(4)
    r = ca.AudioObjectGetPropertyData(ctypes.c_uint32(dev), ctypes.byref(a), 0, None, ctypes.byref(sz), ctypes.byref(v))
    return v.value if r == 0 else 0


def _has_streams(ca, dev: int, scope: str) -> bool:
    a = _Addr(_fcc("stm#"), _fcc(scope), 0)
    sz = ctypes.c_uint32(0)
    return ca.AudioObjectGetPropertyDataSize(ctypes.c_uint32(dev), ctypes.byref(a), 0, None, ctypes.byref(sz)) == 0 and sz.value > 0


def media_state() -> dict:
    """{'audio': bool, 'mic': bool, 'camera': bool}: is any output device playing, any input
    device recording, any camera running - in any app. 'gone' is kAudioDevicePropertyDeviceIsRunningSomewhere."""
    out = {"audio": False, "mic": False, "camera": False}
    try:
        ca, cm = _libs()
        for d in _audio_devices(ca):
            if not _audio_u32(ca, d, "gone"):
                continue
            # A device running "somewhere" is running for all its streams; a headset has both
            # directions, so tell them apart by which side has a client where that can be read.
            has_in, has_out = _has_streams(ca, d, "inpt"), _has_streams(ca, d, "outp")
            if has_in and not has_out:
                out["mic"] = True
            elif has_out and not has_in:
                out["audio"] = True
            else:   # both directions on one device: cannot tell which; the commoner is playback
                out["audio"] = True
        a = _Addr(_fcc("dev#"), _fcc("glob"), 0)
        sz = ctypes.c_uint32(0)
        if cm.CMIOObjectGetPropertyDataSize(ctypes.c_uint32(1), ctypes.byref(a), 0, None, ctypes.byref(sz)) == 0 and sz.value:
            arr = (ctypes.c_uint32 * (sz.value // 4))()
            used = ctypes.c_uint32(0)
            if cm.CMIOObjectGetPropertyData(ctypes.c_uint32(1), ctypes.byref(a), 0, None, sz, ctypes.byref(used), arr) == 0:
                for d in arr:
                    a2 = _Addr(_fcc("gone"), _fcc("glob"), 0)
                    v, u = ctypes.c_uint32(0), ctypes.c_uint32(0)
                    if cm.CMIOObjectGetPropertyData(ctypes.c_uint32(d), ctypes.byref(a2), 0, None, ctypes.c_uint32(4),
                                                    ctypes.byref(u), ctypes.byref(v)) == 0 and v.value:
                        out["camera"] = True
    except Exception:
        log.exception("audio / camera state unavailable")
    return out


# ---- windows: what a click landed on, and whether a banner is showing ---------------------

NC_NAMES = ("Notification Center", "NotificationCenter")
SYSTEM_UI = ("Dock", "Window Server", "Control Center", "SystemUIServer", "Spotlight")   # Window Server = the menu bar


def _windows() -> list:
    import Quartz
    return Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID) or []


def _display_sizes() -> set:
    try:
        import Quartz
        _, ids, _ = Quartz.CGGetActiveDisplayList(16, None, None)
        return {(float(Quartz.CGDisplayPixelsWide(d)), float(Quartz.CGDisplayPixelsHigh(d))) for d in ids}
    except Exception:
        return set()


def window_at(x: float, y: float, windows=None) -> tuple[str, int]:
    """(owner app, window layer) of the topmost window at a point; ('desktop', 0) if none. The
    list is front to back. Skipped: desktop-level windows (wallpaper, widgets), and anything
    above the normal layer that is not the system's own (dictation tools float invisible windows there)."""
    full = _display_sizes()
    for w in (windows if windows is not None else _windows()):
        layer = int(w.get("kCGWindowLayer", 0))
        if layer < 0 or not w.get("kCGWindowAlpha", 1):
            continue
        b = w.get("kCGWindowBounds") or {}
        bx, by, bw, bh = (float(b.get(k, 0)) for k in ("X", "Y", "Width", "Height"))
        if layer > 0 and w.get("kCGWindowOwnerName") in NC_NAMES:
            # While a banner shows, Notification Center's window covers the whole display; the
            # banners themselves stack in its top right corner.
            if bx + bw - 420 <= x < bx + bw and by <= y < by + 260:
                return "Notification Center", layer
            continue
        if layer > 0 and ((bw, bh) in full or w.get("kCGWindowOwnerName") not in SYSTEM_UI):
            # Above the normal layer only the system's own surfaces count. Other apps keep
            # floating windows there that clicks pass straight through: Superwhisper's 240 x 160
            # indicator sits mid-screen at layer 27 even when nothing shows (2026-11-20).
            continue
        if bx <= x < bx + bw and by <= y < by + bh:
            return str(w.get("kCGWindowOwnerName") or "?"), layer
    return "desktop", 0


def place_click(x: float, y: float, before, after) -> tuple[str, int]:
    """Where a click landed: the window under the pointer just before the click; if nothing was
    there (a window that only just appeared), the window there now."""
    owner, layer = window_at(x, y, before)
    return window_at(x, y, after) if owner == "desktop" else (owner, layer)


def banner_showing(windows=None) -> bool:
    """A Notification Center window above the desktop layer = a banner (or the opened panel).
    Its desktop widgets sit on a negative layer and do not count."""
    try:
        for w in (windows if windows is not None else _windows()):
            if w.get("kCGWindowOwnerName") in NC_NAMES and int(w.get("kCGWindowLayer", 0)) > 0:
                return True
    except Exception:
        log.exception("window list unavailable")
    return False


def _pointer() -> tuple[float, float]:
    import Quartz
    p = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return p.x, p.y


class Watcher:
    """Notices clicks and key presses as they happen, without reading them: four times a second
    it looks at the system's click and key COUNTERS, and when the click count has gone up it
    notes which app's window the pointer is over. (A listen-only event tap would be exact, but
    macOS delivers it nothing without Input Monitoring - tried 2026-11-19 - and the counters and
    the pointer position need no permission at all.)

    A click often changes what is on screen (it is why the next app came forward), so the
    pointer is tested against the window list from the poll BEFORE the one that noticed the
    click: the latest picture that is certain to predate it. (Until 2026-11-20 the picture was
    1-2 s old, and after a swipe to another Space it still showed the Space just left: a click
    in the app just arrived at was reported as a click in the app left behind.) Recent clicks and key presses stay in memory so the collector can ask, the moment
    it opens a segment, what the person did just before (`before()`)."""
    POLL_S = 0.25

    def __init__(self):
        self.alive = False
        self.clicks: collections.deque = collections.deque(maxlen=400)   # (ts, owner, layer)
        self.keys: collections.deque = collections.deque(maxlen=400)     # ts of polls that saw key presses
        self.banners: collections.deque = collections.deque(maxlen=50)   # [start, end or None]
        self.pending: collections.deque = collections.deque(maxlen=5000) # rows for ui_events, drained by the sampler

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="signal-watcher").start()

    def _run(self):
        try:
            import Quartz
            s = Quartz.kCGEventSourceStateCombinedSessionState
            count = lambda *ts: sum(int(Quartz.CGEventSourceCounterForEventType(s, t)) for t in ts)
            last_c = count(Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown)
            last_k = count(Quartz.kCGEventKeyDown)
            before = _windows()
            self.alive = True
            log.info("signal watcher running")
            while True:
                time.sleep(self.POLL_S)
                now = time.time()
                wins = _windows()   # read before the counters, so `before` predates any click counted next time
                c = count(Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown)
                k = count(Quartz.kCGEventKeyDown)
                if c > last_c:
                    try:
                        x, y = _pointer()
                        owner, layer = place_click(x, y, before, wins)
                        self.clicks.append((now, owner, layer))
                        self.pending.append((now, "click", json.dumps({"owner": owner, "layer": layer})))
                    except Exception:
                        log.exception("could not place a click")
                if k > last_k:
                    self.keys.append(now)
                last_c, last_k = c, k   # (a counter that went DOWN restarted at login: just follow it)
                before = wins
                showing = banner_showing(wins)
                open_b = self.banners[-1] if self.banners and self.banners[-1][1] is None else None
                if showing and open_b is None:
                    self.banners.append([now, None])
                elif not showing and open_b is not None:
                    open_b[1] = now
                    self.pending.append((open_b[0], "banner", json.dumps({"until": now})))
        except Exception:
            log.exception("signal watcher died")
        self.alive = False

    def before(self, at: float) -> dict:
        """What happened just before time `at` (a segment's start): the last click and where it
        landed, how long since a key was pressed, how long since a banner appeared."""
        # Copies: the watcher thread appends while this (the collector's thread) reads, and
        # iterating a deque that changes underneath raises.
        clicks, keys, banners = list(self.clicks), list(self.keys), list(self.banners)
        click = next(((ts, o, l) for ts, o, l in reversed(clicks)
                      if at - ARRIVE_BEFORE_S <= ts <= at + ARRIVE_AFTER_S), None)
        key = next((ts for ts in reversed(keys) if ts <= at + ARRIVE_AFTER_S), None)
        banner = next((b for b in reversed(banners)
                       if b[0] <= at and (b[1] is None or b[1] >= at - BANNER_BEFORE_S)), None)
        return {"click_owner": click[1] if click else None, "click_layer": click[2] if click else None,
                "click_age": (at - click[0]) if click else None,
                "key_age": (at - key) if key is not None and at - key <= 60 else None,
                "banner_age": (at - banner[0]) if banner else None}


# ---- Chrome: how each page was reached -----------------------------------------------------

_CORE = {0: "link", 1: "typed", 2: "bookmark", 3: "subframe", 4: "subframe", 5: "search",
         6: "start", 7: "form", 8: "reload", 9: "search", 10: "search"}
_FWD_BACK, _ADDRESS_BAR, _REDIRECT = 0x01000000, 0x02000000, 0xC0000000


def chrome_history_files() -> list[tuple[str, str]]:
    base = os.path.expanduser("~/Library/Application Support/Google/Chrome")
    return [(os.path.basename(os.path.dirname(p)), p) for p in sorted(glob.glob(base + "/*/History"))]


def _how(tr: int, has_from: bool, has_opener: bool) -> str:
    """One word for how a visit started, from Chrome's transition value. Checked against a week
    of this Mac's history on 2026-11-19:
      * "typed" WITH the address-bar flag is a typed or autocompleted address;
      * "typed" WITHOUT it is a link handed to Chrome by another app (Slack, Calendar, a terminal
        `open`): 28 of 39 such visits began while another app was in front;
      * "reload" is mostly a single-page site moving between its own pages (X records every
        in-site navigation this way), so it says nothing about how the person got there;
      * "link" with the from-API flag comes from extensions as often as from outside."""
    if tr & _FWD_BACK:
        return "back_forward"
    core = _CORE.get(tr & 0xFF, "other")
    if core == "typed":
        return "typed" if tr & _ADDRESS_BAR else "external"
    if core == "reload":
        return "in_page"
    if core == "link":
        if has_opener:
            return "new_tab"       # opened in a new tab from a page
        return "link" if has_from else "link_unknown"
    return core


def sync_visits(conn, since_s: float = 3 * 86400) -> int:
    """Copy new rows from every Chrome profile's History into `page_visits`. Chrome holds the
    file locked while it runs; `immutable=1` reads it anyway (a read that races a write can fail
    or see a half-written page, so any error just waits for the next round)."""
    n, all_read, started = 0, True, time.time()
    for profile, path in chrome_history_files():
        h = None
        try:
            last = conn.execute("SELECT COALESCE(MAX(visit_id), 0) FROM page_visits WHERE profile=?", (profile,)).fetchone()[0]
            h = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, timeout=1)
            h.row_factory = sqlite3.Row
            floor = int((time.time() - since_s + CHROME_EPOCH) * 1e6)
            rows = h.execute("""SELECT v.id, v.visit_time, v.transition, v.from_visit, v.opener_visit, u.url, u.title
                                FROM visits v JOIN urls u ON u.id = v.url
                                WHERE v.id > ? AND v.visit_time >= ? ORDER BY v.id LIMIT 2000""", (last, floor)).fetchall()

            def visit(vid):
                return h.execute("""SELECT v.id, v.transition, v.from_visit, v.opener_visit, u.url
                                    FROM visits v JOIN urls u ON u.id = v.url WHERE v.id=?""", (vid,)).fetchone() if vid else None
            out = []
            for r in rows:
                # A redirect's visit carries the flags of the hop, not of what started the chain:
                # walk back to the visit the person actually made.
                origin = r
                for _ in range(8):
                    if not (origin["transition"] & _REDIRECT) or not origin["from_visit"]:
                        break
                    prev = visit(origin["from_visit"])
                    if prev is None:
                        break
                    origin = prev
                src = visit(origin["opener_visit"]) or visit(origin["from_visit"])
                out.append((profile, r["id"], r["visit_time"] / 1e6 - CHROME_EPOCH, r["url"], r["title"] or "",
                            r["transition"], _how(origin["transition"], bool(origin["from_visit"]), bool(origin["opener_visit"])),
                            src["url"] if src else None))
            if out:
                with db.tx(conn):
                    conn.executemany("INSERT OR IGNORE INTO page_visits(profile, visit_id, ts, url, title, transition, how, from_url)"
                                     " VALUES (?,?,?,?,?,?,?,?)", out)
                n += len(out)
        except sqlite3.Error as e:
            all_read = False
            log.debug("chrome history (%s) not read this round: %s", profile, e)
        except Exception:
            all_read = False
            log.exception("chrome history sync failed (%s)", profile)
        finally:
            if h is not None:
                h.close()
    if all_read:
        # The stretch History has been read for, so "no visit recorded" can be told from "not read yet".
        if db.kv_get(conn, "signals.visits_from") is None:
            db.kv_set(conn, "signals.visits_from", str(started - since_s))
        db.kv_set(conn, "signals.visits_to", str(started))
    return n


# ---- the sampler the collector drives ------------------------------------------------------

class Sampler:
    def __init__(self, conn):
        self.conn = conn
        self.watcher = Watcher()
        self.prev: dict | None = None
        self.prev_ts = 0.0
        self.last_visits = 0.0
        # Read the counters once before the watcher thread starts: pyobjc resolves Quartz names
        # lazily, and two threads doing that at the same moment made the first sample fail.
        self.prev = _counters()
        self.prev_ts = time.time()
        self.watcher.start()

    def sample(self, now: float | None = None):
        try:
            self._sample(now or time.time())
        except Exception:
            log.exception("signal sample failed")

    def note_switch(self, segment_id: int, at: float):
        """Called by the collector as it opens a segment: store what the person did just before."""
        try:
            if not self.watcher.alive:
                return
            b = self.watcher.before(at)
            self.conn.execute("INSERT OR REPLACE INTO segment_arrivals(segment_id, click_owner, click_layer, click_age, key_age, banner_age)"
                              " VALUES (?,?,?,?,?,?)",
                              (segment_id, b["click_owner"], b["click_layer"], b["click_age"], b["key_age"], b["banner_age"]))
        except Exception:
            log.exception("could not record how segment %s was arrived at", segment_id)

    def _sample(self, now: float):
        dt_s = now - self.prev_ts if self.prev_ts else 0.0
        cur = _counters()
        delta = {"keys": 0, "clicks": 0, "scrolls": 0, "moves": 0}
        # After a gap (sleep, a stalled tick) the difference spans time nobody sampled: drop it.
        if cur and self.prev and 0 < dt_s <= 30:
            delta = {k: max(0, cur[k] - self.prev[k]) for k in delta}   # counters restart at login
        secs = dt_s if 0 < dt_s <= 30 else 0.0
        self.prev, self.prev_ts = cur, now
        m = media_state()
        bucket = now - (now % BUCKET_S)
        with db.tx(self.conn):
            self.conn.execute(
                """INSERT INTO activity_samples(ts, secs, keys, clicks, scrolls, moves, audio, mic, camera)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(ts) DO UPDATE SET secs=secs+excluded.secs, keys=keys+excluded.keys,
                     clicks=clicks+excluded.clicks, scrolls=scrolls+excluded.scrolls, moves=moves+excluded.moves,
                     audio=audio+excluded.audio, mic=mic+excluded.mic, camera=camera+excluded.camera""",
                (bucket, secs, delta["keys"], delta["clicks"], delta["scrolls"], delta["moves"],
                 secs * m["audio"], secs * m["mic"], secs * m["camera"]))
            while self.watcher.pending:
                ts, kind, detail = self.watcher.pending.popleft()
                self.conn.execute("INSERT INTO ui_events(ts, kind, detail) VALUES (?,?,?)", (ts, kind, detail))
        if now - self.last_visits >= VISITS_EVERY_S:
            self.last_visits = now
            sync_visits(self.conn)


# ---- reading the signals back ---------------------------------------------------------------

def _domain(url: str | None) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def activity(conn, t0: float, t1: float) -> dict | None:
    """Input and media totals over [t0, t1), from the 10-second buckets (a bucket that straddles
    an edge is counted in proportion). None when nothing was sampled in that stretch."""
    rows = conn.execute("SELECT * FROM activity_samples WHERE ts > ? AND ts < ?", (t0 - BUCKET_S, t1)).fetchall()
    tot = dict.fromkeys(("secs", "keys", "clicks", "scrolls", "moves", "audio", "mic", "camera"), 0.0)
    for r in rows:
        share = max(0.0, min(t1, r["ts"] + BUCKET_S) - max(t0, r["ts"])) / BUCKET_S
        for k in tot:
            tot[k] += (r[k] or 0) * share
    return tot if tot["secs"] >= 2 else None


def arrival(conn, segment_id, app: str, prev_app: str | None) -> dict | None:
    """What the person did just before a segment began (stored by Sampler.note_switch): the last
    click and where it landed, or that there was none. None for segments from before 2026-11-19."""
    r = conn.execute("SELECT * FROM segment_arrivals WHERE segment_id=?", (segment_id,)).fetchone() if segment_id else None
    if r is None:
        return None
    owner = r["click_owner"]
    click = (None if owner is None else
             "dock" if owner == "Dock" else
             "banner" if owner in NC_NAMES else
             "menu_bar" if owner in ("Window Server", "Control Center", "SystemUIServer") else
             "own_window" if owner == app else
             "previous_app" if prev_app and owner == prev_app else "other")
    return {"click": click, "click_owner": owner, "banner_s": r["banner_age"],
            "keys": r["key_age"] is not None and r["key_age"] <= 4.0}


def page_reached(conn, url: str, start: float, end: float, full: bool = False) -> dict | None:
    """How Chrome reached this segment's page: the visit to that address nearest the segment's
    start. None when there is none (a tab that was already open, or History not yet read)."""
    if not url or not url.startswith("http"):
        return None
    bare = url.split("#")[0]
    r = conn.execute("""SELECT how, from_url, ts FROM page_visits
                        WHERE ts >= ? AND ts <= ? AND (url = ? OR url = ? OR url LIKE ? ESCAPE '\\')
                        ORDER BY ABS(ts - ?) LIMIT 1""",
                     (start - 15, max(end, start + 5), url, bare,
                      bare.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "#%", start)).fetchone()
    if r is None:
        # "No visit" only means "already open" if History was read for this stretch (Chrome
        # writes a visit to disk up to ~10 s late, hence the margin).
        lo, hi = (float(db.kv_get(conn, k) or 0) for k in ("signals.visits_from", "signals.visits_to"))
        return {"how": "already_open"} if lo and lo <= start - 15 and hi >= max(end, start + 5) + 20 else None
    out = {"how": r["how"], "from": _domain(r["from_url"])}
    if full:   # the whole address of the page it came from (the evaluator; other callers get the site)
        out["from_url"] = r["from_url"] or ""
    return out


_PAGE_TEXT = {"typed": "typed the address", "bookmark": "opened from a bookmark", "search": "searched from the address bar",
              "back_forward": "went back / forward", "form": "submitted a form",
              "start": "opened with the window (start page or restored tab)",
              "external": "link opened from another app",
              "already_open": "no new page load (tab already open, or a move inside the page)",
              "link_unknown": "a link, source page not recorded"}
_PAGE_SILENT = ("in_page", "subframe", "other")   # say nothing: these do not tell how the person got there


def describe(conn, seg: dict, prev_app: str | None = None, full: bool = False) -> list[str]:
    """Short notes about one computer segment, for the evaluator's log and the Captures card.
    `seg` needs start/end (or clip_start/clip_end), app, url. Empty when nothing was recorded.
    `full` (the evaluator, since 2026-11-20) names the page a link came from by its whole
    address, not its site, adds pointer moves to the input rates, and says when a segment is
    shorter than the 10-second counters it is measured with."""
    t0 = seg.get("clip_start", seg.get("start"))
    t1 = seg.get("clip_end", seg.get("end")) or t0
    notes: list[str] = []
    try:
        if seg.get("idle") or seg.get("state") == "idle" or t0 is None:
            return notes
        arr = arrival(conn, seg.get("id"), seg.get("app") or "", prev_app)
        if arr is not None and prev_app is not None and prev_app != seg.get("app"):
            k = arr["click"]
            notes.append("arrived by: " + (
                "a click on the Dock" if k == "dock" else
                "a click on a notification banner" if k == "banner" else
                "a click on the menu bar" if k == "menu_bar" else
                "a click on this app's own window" if k == "own_window" else
                f"a click inside {prev_app} (something clicked there brought this forward, e.g. a link)" if k == "previous_app" else
                "a click where no window was (the desktop)" if k == "other" and arr.get("click_owner") == "desktop" else
                f"a click in a third app's window ({arr.get('click_owner')})" if k == "other" else
                "the keyboard, no click (Cmd-Tab, Spotlight or a shortcut)" if arr["keys"] else
                "no click and no key press (something else opened it)"))
        if arr is not None and arr["banner_s"] is not None and arr["banner_s"] <= BANNER_BEFORE_S and arr.get("click") != "banner":
            notes.append(f"a notification banner appeared {arr['banner_s']:.0f} s before")
        # only Chrome's History is read; another browser's pages would all look "already open"
        pg = page_reached(conn, seg.get("url") or "", t0, t1, full) if seg.get("app") == "Google Chrome" else None
        if pg and pg["how"] not in _PAGE_SILENT:
            h = pg["how"]
            src = (pg.get("from_url") if full else None) or pg.get("from")
            notes.append("page reached by: " + (
                f"a link on {src}" if h == "link" and src else
                f"a new tab opened from {src}" if h == "new_tab" and src else
                "a new tab opened from a page" if h == "new_tab" else
                "a link" if h == "link" else _PAGE_TEXT.get(h, h)))
        act = activity(conn, t0, t1)
        if act:
            mins = max(act["secs"], 1.0) / 60.0
            notes.append(f"input: {act['keys'] / mins:.0f} keys/min, {act['clicks'] / mins:.0f} clicks/min, "
                         f"{act['scrolls'] / mins:.0f} scroll events/min"
                         + (f", {act['moves'] / mins:.0f} pointer-move events/min" if full else "")
                         # Counts come in 10-second buckets split pro rata: a 6-second segment between
                         # two typing segments read "148 keys/min" (segment 36705, 2026-11-20).
                         + (" (measured in 10-second counts, so for a segment this short these are"
                            " partly its neighbours' input)" if full and t1 - t0 < 2 * BUCKET_S else ""))
            for k, label in (("audio", "audio playing"), ("mic", "microphone on"), ("camera", "camera on")):
                share = act[k] / max(act["secs"], 1.0)
                if share >= 0.15:
                    notes.append(f"{label} {min(share, 1.0) * 100:.0f}% of the time")
    except Exception:
        log.exception("could not describe signals for segment %s", seg.get("id"))
    return notes


# ---- more for the evaluator's log (2026-11-20) ----------------------------------------------------
# One rate per segment hides when the typing happened: 605 segments over 5 minutes carried 101 of
# 273 active hours in 30 days, each described by a single "N keys/min". And a Chrome segment is
# recorded under the title it opened with: segment 36414 was 24 minutes on one food-delivery
# title while History held 45 page loads on two social sites. These give the evaluator the
# minute-by-minute counts, the recorded pauses, and every page load inside a segment.

def _hm(ts: float, secs: bool = True) -> str:
    from .config import local_dt
    return local_dt(ts).strftime("%H:%M:%S" if secs else "%H:%M")


def minute_strip(conn, t0: float, t1: float) -> str:
    """Input and media for each clock minute of [t0, t1), as one line; '' when nothing was
    sampled. Each entry: HH:MM keys/clicks/scroll events/pointer moves, then a (audio playing),
    m (microphone on), c (camera on) when that was so for half the sampled time or more."""
    rows = conn.execute("SELECT * FROM activity_samples WHERE ts > ? AND ts < ? ORDER BY ts", (t0 - BUCKET_S, t1)).fetchall()
    if not rows:
        return ""
    mins: dict[float, dict] = {}
    for r in rows:
        share = max(0.0, min(t1, r["ts"] + BUCKET_S) - max(t0, r["ts"])) / BUCKET_S
        if share <= 0:
            continue
        m = mins.setdefault(max(t0, r["ts"]) - (max(t0, r["ts"]) % 60), dict.fromkeys(("secs", "keys", "clicks", "scrolls", "moves", "audio", "mic", "camera"), 0.0))
        for k in m:
            m[k] += (r[k] or 0) * share
    out = []
    first = min(mins) if mins else t0
    k = first
    while k < t1:
        m = mins.get(k)
        if m is None or m["secs"] < 1:
            out.append(f"{_hm(k, False)} not sampled")
        else:
            flags = "".join(f for f, key in (("a", "audio"), ("m", "mic"), ("c", "camera")) if m[key] >= 0.5 * m["secs"])
            out.append(f"{_hm(k, False)} {m['keys']:.0f}/{m['clicks']:.0f}/{m['scrolls']:.0f}/{m['moves']:.0f}" + (" " + flags if flags else ""))
        k += 60
    return " | ".join(out)


def pauses(conn, t0: float, t1: float) -> list[tuple[float, float]]:
    """Recorded stretches of no keyboard or mouse input inside [t0, t1) (the `inactivity` table:
    pauses of a minute or more that did not become an away segment), clipped to it."""
    return [(max(r["start"], t0), min(r["end"], t1)) for r in conn.execute(
        "SELECT start, end FROM inactivity WHERE start < ? AND end > ? ORDER BY start", (t1, t0))]


def page_loads(conn, t0: float, t1: float) -> list[dict]:
    """Every page Chrome loaded in [t0, t1], oldest first: {ts, how, url, title, from_url}.
    Frames inside a page are left out. Empty when none, or when History was not read then."""
    return [dict(r) for r in conn.execute(
        "SELECT ts, how, url, title, from_url FROM page_visits WHERE ts >= ? AND ts <= ? AND COALESCE(how,'') != 'subframe' ORDER BY ts",
        (t0, t1))]


_HOW_TEXT = dict(_PAGE_TEXT, link="a link", new_tab="a new tab opened from a page", in_page="a move inside the site (or a reload)",
                 link_unknown="a link, source not recorded")


def page_load_lines(conn, t0: float, t1: float) -> list[str]:
    """page_loads() as text lines: time, how, address, title, and the page it came from."""
    out = []
    for v in page_loads(conn, t0, t1):
        title = " ".join((v["title"] or "").split())
        out.append(f"{_hm(v['ts'])}  {_HOW_TEXT.get(v['how'] or '', v['how'] or 'unknown')}  {v['url']}"
                   + (f"  “{title}”" if title else "") + (f"  from {v['from_url']}" if v["from_url"] else ""))
    return out


def input_around(conn, t0: float, t1: float, look_s: float = 900.0) -> str:
    """Mac input during [t0, t1) and how long before / after it the nearest input was, as a note
    for a PHONE line; '' when input was not being sampled then (before 2026-11-20). Whether the
    Mac's keyboard was in use is the one direct sign of whether the phone was in hand."""
    act = activity(conn, t0, t1)
    if act is None:
        return ""
    used = lambda r: (r["keys"] or 0) + (r["clicks"] or 0) + (r["scrolls"] or 0) > 0
    prev = next((r for r in conn.execute("SELECT * FROM activity_samples WHERE ts < ? AND ts >= ? ORDER BY ts DESC",
                                         (t0 - BUCKET_S, t0 - look_s)) if used(r)), None)
    nxt = next((r for r in conn.execute("SELECT * FROM activity_samples WHERE ts >= ? AND ts <= ? ORDER BY ts",
                                        (t1, t1 + look_s)) if used(r)), None)
    n = act["keys"] + act["clicks"] + act["scrolls"]
    during = (f"{act['keys']:.0f} keys, {act['clicks']:.0f} clicks, {act['scrolls']:.0f} scroll events" if n >= 1 else "none")
    before = f"about {max(0.0, t0 - (prev['ts'] + BUCKET_S)):.0f} s before" if prev is not None else f"more than {look_s / 60:.0f} min before (or not sampled)"
    after = f"about {max(0.0, nxt['ts'] - t1):.0f} s after" if nxt is not None else f"more than {look_s / 60:.0f} min after (or not sampled yet)"
    return f"Mac input during this: {during}; last Mac input {before}, next {after} (10-second counts)"

"""The iPhone's foreground log, read from the phone-unlock-logger's database (read-only).

That project (~/Documents/agent-coding/phone-unlock-logger) collects the phone's App.InFocus
stream, which Screen Time syncs to this Mac, into per-app segments and unlock events. The
phone pushes the stream on its own schedule, so the data usually runs hours behind the Mac
lane; `through()` says how far it reaches, and every payload carries it so the dashboard can
say "phone data ends 12:12" instead of showing an empty lane.

The stream says which app was on screen, not whether anyone was touching it. The one proxy is
iOS Auto-Lock: an untouched (and, with attention awareness, unwatched) phone locks itself after
the Auto-Lock interval, which ends the segment. So a segment that ends in a lock at almost exactly
that interval was put down with the app open and barely used (`put_down`), one that locks earlier
was in use to the end, and a longer one is ambiguous. The interval is read off the data
(`autolock_s`: the sharp spike in lock-ended durations at one of iOS's Auto-Lock options), so a
change in Settings (5 min until 2026-11-11, 2 min after) is picked up on its own.

Besides the lane under each timeline, the evaluator is shown PICKUPS (`pickups()`): one per
stretch of use between locks, each judged as a switch of attention (table phone_evals; see
evaluator.py and stats._carve_phone).
"""
from __future__ import annotations

import bisect
import logging
import os
import sqlite3
import time
from pathlib import Path

log = logging.getLogger("phone")

DEFAULT_DB = Path.home() / "Library" / "Application Support" / "PhoneUnlockLogger" / "phone.db"

# Lock-screen surfaces and overlays that are not the person using an app.
NOT_USE = {"com.apple.ClockAngel", "com.apple.springboard.today-view", "com.apple.control-center",
           "com.apple.PassbookUIService", "com.apple.ContinuityCaptureShieldUI",
           "com.apple.ScreenshotServicesService", "com.apple.springboard"}

NAMES = {
    "org.whispersystems.signal": "Signal", "com.ouraring.oura": "Oura", "com.apple.InCallService": "Phone call",
    "com.3nder.threender": "Feeld", "com.apple.mobilesafari": "Safari", "com.elityre.emaApp": "EMA",
    "com.anthropic.claude": "Claude", "com.moxco.bumble": "Bumble", "com.apple.MobileSMS": "Messages",
    "com.elityre.meditationTimer": "Meditation Timer", "com.google.ios.youtube": "YouTube",
    "com.google.Docs": "Google Docs", "com.ubercab.UberClient": "Uber", "com.elityre.exerciseTracker": "Exercise Tracker",
    "com.toggl.daneel": "Toggl", "com.google.Gmail": "Gmail", "com.deezer.Deezer": "Deezer",
    "com.cardify.tinder": "Tinder", "net.whatsapp.WhatsApp": "WhatsApp", "com.ubercab.UberEats": "Uber Eats",
    "com.apple.camera": "Camera", "com.apple.Preferences": "Settings", "com.amazon.Amazon": "Amazon",
    "com.aisense.otter": "Otter", "com.leshko.cloudplayer": "Cloud Player", "com.google.Sheets": "Google Sheets",
    "com.google.Maps": "Google Maps", "com.apple.mobilephone": "Phone", "com.airbnb.app": "Airbnb",
    "com.tabatasongs.musicapp": "Tabata Songs", "com.google.photos": "Google Photos", "com.apple.mobiletimer": "Clock",
    "com.apple.mobileslideshow": "Photos", "com.apple.mobilecal": "Calendar", "net.ichi2.anki": "Anki",
    "com.fourthframe.Intervals": "Intervals", "com.apple.reminders": "Reminders", "com.apple.WebSheet": "Wi-Fi sign-in",
    "com.apple.mobilenotes": "Notes", "com.apple.Music": "Music", "com.apple.podcasts": "Podcasts",
    "com.apple.mobilemail": "Mail", "com.apple.weather": "Weather", "com.apple.AppStore": "App Store",
    "com.apple.Maps": "Apple Maps", "com.apple.Health": "Health", "com.burbn.instagram": "Instagram",
    "com.atebits.Tweetie2": "X", "com.reddit.Reddit": "Reddit", "com.spotify.client": "Spotify",
    "com.apple.findmy": "Find My", "com.apple.shortcuts": "Shortcuts", "com.apple.Passbook": "Wallet",
}


# iOS Settings > Display & Brightness > Auto-Lock offers these, in seconds.
AUTOLOCK_OPTIONS = (30, 60, 120, 180, 240, 300)
LOCK_GAP_S = 60.0          # no app on screen for this long after a segment = the phone was locked
AUTOLOCK_HALF_WINDOW_S = 2 * 86400


def _lock_ended_durations(c, t0: float, t1: float) -> list[float]:
    """Durations of segments in [t0, t1) that ended by the phone locking (nothing on screen afterwards)."""
    marks = ",".join("?" * len(NOT_USE))
    rows = c.execute(f"""SELECT s.duration_s,
                           (SELECT MIN(n.start_ts) FROM segments n
                             WHERE n.start_ts >= s.end_ts AND n.bundle_id NOT IN ({marks})) - s.end_ts AS gap
                         FROM segments s
                         WHERE s.end_kind = 'explicit' AND s.end_ts IS NOT NULL AND s.duration_s IS NOT NULL
                           AND s.start_ts >= ? AND s.start_ts < ? AND s.bundle_id NOT IN ({marks})""",
                     (*NOT_USE, t0, t1, *NOT_USE)).fetchall()
    return [r[0] for r in rows if r[1] is None or r[1] >= LOCK_GAP_S]


def autolock_s(c, t: float) -> int | None:
    """The Auto-Lock interval in force around time t, inferred from the data.

    Self-locks pile up in a 15-second band just above the interval (300.0-310 s for a 5-minute
    setting), while deliberate locks spread out. The option whose band holds the most lock-ended
    segments wins, if it has at least 3 and stands clear of the surrounding durations. Two days
    either side of t, so a setting change shows up within a couple of days of self-locks."""
    # Worked out per six hours and kept for ten minutes: the four-day scan takes 0.1 s, and the
    # span review queue asked for it once per span (6 of that page's 7 seconds, 2026-11-20).
    part = int(t // 21600)
    hit = _AUTOLOCK_CACHE.get(part)
    if hit is not None and hit[0] > time.time():
        return hit[1]
    val = _autolock_s(c, part * 21600 + 10800.0)
    _AUTOLOCK_CACHE[part] = (time.time() + 600, val)
    return val


_AUTOLOCK_CACHE: dict = {}


def _autolock_s(c, t: float) -> int | None:
    durs = _lock_ended_durations(c, t - AUTOLOCK_HALF_WINDOW_S, t + AUTOLOCK_HALF_WINDOW_S)
    if len(durs) < 3:
        return None
    best, best_n = None, 0
    for L in AUTOLOCK_OPTIONS:
        n = sum(1 for d in durs if L - 5 <= d < L + 15)
        around = sum(1 for d in durs if (L - 45 <= d < L - 5) or (L + 15 <= d < L + 55)) / 4.0  # per 20 s
        if n >= 3 and n > 2 * around and n > best_n:
            best, best_n = L, n
    return best


def is_put_down(duration_s: float, locked: bool, L: int | None) -> bool:
    return bool(L) and locked and L - 5 <= duration_s < L + 15


def app_name(bundle: str) -> str:
    if bundle in NAMES:
        return NAMES[bundle]
    tail = bundle.rsplit(".", 1)[-1]
    return tail[:1].upper() + tail[1:] if tail else bundle


def db_path(cfg=None) -> Path:
    p = (getattr(cfg, "phone_db", "") or "").strip()
    return Path(os.path.expanduser(p)) if p else DEFAULT_DB


def _connect(cfg=None):
    p = db_path(cfg)
    if not p.exists():
        return None
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
    c.row_factory = sqlite3.Row
    return c


def through(cfg=None) -> float | None:
    """Timestamp of the newest phone event the Mac has - where the lane's knowledge ends."""
    c = _connect(cfg)
    if c is None:
        return None
    try:
        r = c.execute("SELECT MAX(COALESCE(end_ts, start_ts)) FROM segments").fetchone()
        return float(r[0]) if r and r[0] else None
    except sqlite3.Error as e:
        log.warning("phone db: %s", e)
        return None
    finally:
        c.close()


def window(cfg, t0: float, t1: float, open_cap_s: float = 900.0, shots: bool = True) -> dict:
    """Phone segments and unlocks overlapping [t0, t1): {segments, unlocks, through, autolock_s}.

    A segment carries `put_down` when the phone locked itself at the Auto-Lock interval: the app
    was left open on an untouched phone, so the real use was the first seconds at most."""
    c = _connect(cfg)
    if c is None:
        return {"segments": [], "unlocks": [], "through": None, "available": False, "autolock_s": None}
    try:
        now = time.time()
        L = autolock_s(c, (t0 + t1) / 2)
        marks = ",".join("?" * len(NOT_USE))
        segs = []
        for r in c.execute(f"""SELECT start_ts, end_ts, duration_s, bundle_id, transition_reason, end_kind,
                                 (SELECT MIN(n.start_ts) FROM segments n
                                   WHERE n.start_ts >= s.end_ts AND n.bundle_id NOT IN ({marks})) AS next_start
                               FROM segments s WHERE start_ts < ? AND COALESCE(end_ts, start_ts + ?) > ?
                               ORDER BY start_ts""", (*NOT_USE, t1, open_cap_s, t0)):
            if r["bundle_id"] in NOT_USE:
                continue
            end = r["end_ts"] if r["end_ts"] is not None else min(r["start_ts"] + open_cap_s, now)
            if end - r["start_ts"] < 1.0:
                continue
            reason = (r["transition_reason"] or "")
            how = ("notification" if "notification" in reason else "home screen" if "homescreen" in reason
                   else "app switcher" if "appswitcher" in reason else "another app" if "externalrequest" in reason else "")
            locked = (r["end_kind"] == "explicit" and r["end_ts"] is not None
                      and (r["next_start"] is None or r["next_start"] - r["end_ts"] >= LOCK_GAP_S))
            segs.append({"start": max(r["start_ts"], t0), "end": min(end, t1), "bundle": r["bundle_id"],
                         "app": app_name(r["bundle_id"]), "how": how, "open": r["end_kind"] == "open",
                         "put_down": is_put_down(end - r["start_ts"], locked, L)})
        unlocks = [{"ts": r["ts"], "app": app_name(r["first_bundle_id"]), "notification": r["notification"],
                    "gap_min": round((r["gap_s"] or 0) / 60, 1)}
                   for r in c.execute("""SELECT ts, first_bundle_id, gap_s, notification FROM unlocks
                                         WHERE ts >= ? AND ts < ? AND excluded_reason IS NULL ORDER BY ts""", (t0, t1))]
        thr = c.execute("SELECT MAX(COALESCE(end_ts, start_ts)) FROM segments").fetchone()[0]
        if shots:           # the lane's tooltip shows what was on the phone's screen, when a picture exists
            caps = captures(cfg, t0 - 60, t1 + 60)
            for sg in segs:
                mine = captures_of(caps, sg["start"], sg["end"], sg["app"]) if caps else []
                if len(mine) > 12:
                    mine = [mine[round(j * (len(mine) - 1) / 11)] for j in range(12)]
                sg["shots"] = [[k["name"], round(k["ts"], 1)] for k in mine]
        return {"segments": segs, "unlocks": unlocks, "through": float(thr) if thr else None, "available": True,
                "autolock_s": L}
    except sqlite3.Error as e:
        log.warning("phone db: %s", e)
        return {"segments": [], "unlocks": [], "through": None, "available": False, "autolock_s": None}
    finally:
        c.close()


CAPTURE_PAD_S = 5.0        # the phone's clock (segments) and the Mac's (captures) differ a little
CAPTURE_LATENCY_S = 3.0    # how long a screenshot took to come back, for rows that did not record it
LOGGER_PRIVATE = ("org.whispersystems.signal",)   # the logger's own default private_bundles
_warned_private: set = set()


def _private_bundles(cfg, c) -> set:
    """Bundle ids whose captures are never handed out: the logger's `private_bundles` (its
    config.json beside the database, else its default) and every bundle that one of this app's
    `private_apps` names. A name matches a bundle id, its display name (NAMES) or any dot-part of
    the id; a name that matches nothing in the phone log is logged once, since it protects nothing."""
    import json
    out = {b.lower() for b in LOGGER_PRIVATE}
    try:
        user = json.loads((db_path(cfg).parent / "config.json").read_text())
        pb = user.get("private_bundles") if isinstance(user, dict) else None
        out |= {str(b).lower() for b in ([pb] if isinstance(pb, str) else pb or [])}
    except (OSError, ValueError):
        pass
    names = {str(a).strip().lower() for a in (getattr(cfg, "private_apps", None) or []) if str(a).strip()}
    if names:
        hit = set()
        for (bundle,) in c.execute("SELECT DISTINCT bundle_id FROM segments"):
            keys = {bundle.lower(), app_name(bundle).lower(), *bundle.lower().split(".")}
            if keys & names:
                out.add(bundle.lower())
                hit |= keys & names
        for n in names - hit - _warned_private:
            _warned_private.add(n)
            log.warning("private_apps entry %r matches no app in the phone log: it withholds no phone capture", n)
    return out


def captures(cfg, t0: float, t1: float) -> list[dict]:
    """Pictures of the iPhone's screen taken in [t0, t1) that may be shown:
    [{ts, last_seen, name, bundle, app}].

    The phone-unlock-logger's capture job (since 2026-11-23) photographs the screen while the
    phone is on the cable or the Mac's Wi-Fi. A capture does not know which app it shows until
    the phone's log syncs, and these pictures go to a model, so one is handed out only when ALL
    of these hold; anything else is withheld:
      - the logger has not flagged it private;
      - it was taken wholly inside one recorded app segment (asked for after the segment's start
        and back before its end, with a margin for the two clocks), and that segment has a real
        end record. So nothing from a gap is shown - lock screen, notification centre, app
        switcher, home screen - and nothing from the edge of an app change;
      - that app is not private (_private_bundles), and no private app's segment comes within
        the margin of it. A private segment whose end is missing or estimated counts as running
        until the next segment starts.
    What this cannot see: a notification banner of a private app drawn over another app.
    `name` is what config.screenshot_file and /shot/ resolve (they vet through here again)."""
    from .config import PHONE_SHOT_PREFIX, PRIVATE_MARK
    c = _connect(cfg)
    if c is None:
        return []
    try:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name='captures'").fetchone():
            return []
        has_ret = "returned_ts" in [r[1] for r in c.execute("PRAGMA table_info(captures)")]
        rows = c.execute(f"SELECT ts, {'returned_ts' if has_ret else 'NULL'} AS ret, last_seen_ts, file FROM captures"
                         " WHERE private = 0 AND ts >= ? AND ts < ? ORDER BY ts", (t0, t1)).fetchall()
        if not rows:
            return []
        private = _private_bundles(cfg, c)
        segs = c.execute("""SELECT s.start_ts, s.end_ts, s.end_kind, s.bundle_id,
                              (SELECT MIN(n.start_ts) FROM segments n WHERE n.start_ts > s.start_ts) AS next_start
                            FROM segments s WHERE s.start_ts < ? AND COALESCE(s.end_ts, s.start_ts + 86400) > ?
                            ORDER BY s.start_ts""", (t1 + 3600, t0 - 86400)).fetchall()
        spans = []   # where a private app was, or may have been, in front
        for g in segs:
            if g["bundle_id"].lower() in private:
                sure = g["end_ts"] is not None and g["end_kind"] in ("explicit", "implicit")
                end = g["end_ts"] if sure else (g["next_start"] if g["next_start"] is not None else float("inf"))
                spans.append((g["start_ts"] - CAPTURE_PAD_S, end + CAPTURE_PAD_S))
        ok = [g for g in segs if g["end_ts"] is not None and g["end_kind"] == "explicit"
              and g["bundle_id"] not in NOT_USE and g["bundle_id"].lower() not in private]
        starts = [g["start_ts"] for g in ok]
        out = []
        for r in rows:
            day, _, fn = r["file"].partition("/")
            if not fn or "/" in fn or PRIVATE_MARK in fn.lower():
                continue
            a = r["ts"]
            b = r["ret"] if r["ret"] is not None else a + CAPTURE_LATENCY_S
            i = bisect.bisect_right(starts, a - 1.0) - 1      # the latest segment that had begun by then
            g = ok[i] if i >= 0 and b + 1.0 <= ok[i]["end_ts"] else None
            if g is None or any(lo <= b and hi >= a for lo, hi in spans):
                continue
            out.append({"ts": a, "last_seen": r["last_seen_ts"], "name": f"{PHONE_SHOT_PREFIX}{day}~{fn}",
                        "bundle": g["bundle_id"], "app": app_name(g["bundle_id"])})
        return out
    except sqlite3.Error as e:
        log.warning("phone db captures: %s", e)
        return []
    finally:
        c.close()


def capture_allowed(cfg, name: str) -> bool:
    """Whether the capture with this `iphone~<day>~<file>` name may be shown NOW. Asked on every
    resolve, so a name stored when it was allowed stops working if its app is made private."""
    from .config import PHONE_SHOT_PREFIX
    parts = name[len(PHONE_SHOT_PREFIX):].split("~")
    c = _connect(cfg)
    if c is None or len(parts) != 2:
        return False
    try:
        r = c.execute("SELECT ts FROM captures WHERE file = ? AND private = 0", (parts[0] + "/" + parts[1],)).fetchone()
    except sqlite3.Error:
        return False
    finally:
        c.close()
    return r is not None and any(k["name"] == name for k in captures(cfg, r[0] - 1.0, r[0] + 1.0))


def captures_of(caps: list[dict], start: float, end: float, app: str | None = None) -> list[dict]:
    """Those of `caps` taken during [start, end]; with `app`, only that app's (a pickup's entry
    can span a short gap between two stretches of one app, and nothing else belongs to it)."""
    return [k for k in caps if start <= k["ts"] <= end and (app is None or k["app"] == app)]


PICKUP_PAD_S = 1800.0      # look this far either side of a window so a pickup's key is stable
PUT_DOWN_USE_S = 10.0      # a put-down segment counts as use for its first seconds only
PICKUP_MIN_S = 3.0         # shorter than this is a pocket wake, not a look
# The one lock-screen view the stream records: swiping to the widgets page without unlocking.
# A plain glance at the lock screen (time, notifications) leaves no record at all.
LOCK_SCREEN_BUNDLE = "com.apple.springboard.today-view"


def lock_glances(cfg, t0: float, t1: float) -> list[tuple[float, float]]:
    """(start, end) of lock-screen widget views starting in [t0, t1): the phone was looked at
    and not unlocked. Rare (20 in 40 days, 6 of them not followed by an unlock); it is all the
    data shows of the lock screen."""
    c = _connect(cfg)
    if c is None:
        return []
    try:
        return [(r[0], r[1]) for r in c.execute(
            "SELECT start_ts, COALESCE(end_ts, start_ts + 1) FROM segments WHERE bundle_id = ?"
            " AND (transition_reason LIKE '%lock-screen%' OR transition_reason LIKE '%DashBoard%')"
            " AND start_ts >= ? AND start_ts < ? ORDER BY start_ts", (LOCK_SCREEN_BUNDLE, t0, t1))]
    except sqlite3.Error as e:
        log.warning("phone db: %s", e)
        return []
    finally:
        c.close()


def _first_opens(cfg, ts: float) -> dict:
    """{bundle: start of its first segment} for the logical day `ts` falls in - so a line can say
    "first Oura open today" (Eli: the first Oura open of the day is an intentional sync)."""
    from .config import day_bounds, logical_date
    d0, d1 = day_bounds(logical_date(ts))
    c = _connect(cfg)
    if c is None:
        return {}
    try:
        return {r[0]: r[1] for r in c.execute(
            "SELECT bundle_id, MIN(start_ts) FROM segments WHERE start_ts >= ? AND start_ts < ?"
            " AND (duration_s IS NULL OR duration_s >= 1.0) GROUP BY bundle_id", (d0, d1))}
    except sqlite3.Error:
        return {}
    finally:
        c.close()


def pickups(cfg, t0: float, t1: float) -> list[dict]:
    """Pickups STARTING in [t0, t1): one per stretch of phone use between locks.

    {key, start, end, use_s, apps: [(name, seconds, how)], segs, put_down, lock_screen}. `segs`
    is the unit the evaluator judges: one entry per app in the order used, {key, start, end, app,
    how, first_today} (back-to-back records of one app are one entry; key = its start in whole
    seconds, unique within the pickup; first_today = that app's first open of the logical day).
    A pickup with
    lock_screen=True is a look at the lock screen with no unlock (lock_glances) that is not part
    of a pickup of the ordinary kind; it has no minimum length. Segments less than
    LOCK_GAP_S apart are one pickup. Time on a put-down segment (the phone locked itself at the
    Auto-Lock interval: left open, untouched) is not use, beyond its first seconds. `key` is the
    pickup's start in whole seconds; it is computed from unclipped segments, so the same pickup
    gets the same key whichever window asks."""
    from .config import logical_date
    w = window(cfg, t0 - PICKUP_PAD_S, t1 + PICKUP_PAD_S, shots=False)
    out, cur, firsts = [], None, {}
    for sg in w["segments"]:
        end = min(sg["end"], sg["start"] + PUT_DOWN_USE_S) if sg["put_down"] else sg["end"]
        if cur is not None and sg["start"] - cur["end"] < LOCK_GAP_S and not cur["put_down"]:
            cur["end"] = max(cur["end"], end)
        else:
            cur = {"start": sg["start"], "end": end, "apps": [], "segs": [], "put_down": False}
            out.append(cur)
        cur["put_down"] = sg["put_down"]
        if cur["apps"] and cur["apps"][-1][0] == sg["app"]:
            n, d, how = cur["apps"][-1]
            cur["apps"][-1] = (n, d + (end - sg["start"]), how)
            cur["segs"][-1]["end"] = end
        else:
            cur["apps"].append((sg["app"], end - sg["start"], sg["how"]))
            day = logical_date(sg["start"])
            if day not in firsts:
                firsts[day] = _first_opens(cfg, sg["start"])
            f0 = firsts[day].get(sg["bundle"])
            key = int(sg["start"])
            if cur["segs"] and key <= cur["segs"][-1]["key"]:
                key = cur["segs"][-1]["key"] + 1
            cur["segs"].append({"key": key, "start": sg["start"], "end": end, "app": sg["app"], "how": sg["how"],
                                "first_today": f0 is not None and abs(f0 - sg["start"]) < 1.0})
    res = []
    for p in out:
        p["use_s"] = sum(d for _, d, _ in p["apps"])
        p["key"] = int(p["start"])
        p["lock_screen"] = False
        if t0 <= p["start"] < t1 and p["use_s"] >= PICKUP_MIN_S:
            res.append(p)
    glance = None
    for a, b in lock_glances(cfg, t0, t1):
        if any(q["start"] - LOCK_GAP_S < a < q["end"] + LOCK_GAP_S for q in out):
            continue          # on the way into, or out of, an unlocked pickup: part of that pickup
        if glance is not None and a - glance["end"] < LOCK_GAP_S:
            glance["end"] = max(glance["end"], b)
        else:
            glance = {"start": a, "end": max(b, a + 1.0), "key": int(a), "put_down": False, "lock_screen": True}
            res.append(glance)
        glance["use_s"] = glance["end"] - glance["start"]
        glance["apps"] = [("Lock screen", glance["use_s"], "")]
        glance["segs"] = [{"key": glance["key"], "start": glance["start"], "end": glance["end"], "app": "Lock screen",
                           "how": "", "first_today": False}]
    res.sort(key=lambda p: p["start"])
    return res

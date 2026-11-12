"""The iPhone's foreground log, read from the phone-unlock-logger's database (read-only).

That project (~/Documents/agent-coding/phone-unlock-logger) collects the phone's App.InFocus
stream, which Screen Time syncs to this Mac, into per-app segments and unlock events. The
phone pushes the stream on its own schedule, so the data usually runs hours behind the Mac
lane; `through()` says how far it reaches, and every payload carries it so the dashboard can
say "phone data ends 12:12" instead of showing an empty lane.

The monitor only displays this for now (a lane under each timeline). Feeding it to the
evaluator is a separate step.
"""
from __future__ import annotations

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


def window(cfg, t0: float, t1: float, open_cap_s: float = 900.0) -> dict:
    """Phone segments and unlocks overlapping [t0, t1): {segments, unlocks, through}."""
    c = _connect(cfg)
    if c is None:
        return {"segments": [], "unlocks": [], "through": None, "available": False}
    try:
        now = time.time()
        segs = []
        for r in c.execute("""SELECT start_ts, end_ts, duration_s, bundle_id, transition_reason, end_kind
                              FROM segments WHERE start_ts < ? AND COALESCE(end_ts, start_ts + ?) > ?
                              ORDER BY start_ts""", (t1, open_cap_s, t0)):
            if r["bundle_id"] in NOT_USE:
                continue
            end = r["end_ts"] if r["end_ts"] is not None else min(r["start_ts"] + open_cap_s, now)
            if end - r["start_ts"] < 1.0:
                continue
            reason = (r["transition_reason"] or "")
            how = ("notification" if "notification" in reason else "home screen" if "homescreen" in reason
                   else "app switcher" if "appswitcher" in reason else "another app" if "externalrequest" in reason else "")
            segs.append({"start": max(r["start_ts"], t0), "end": min(end, t1), "bundle": r["bundle_id"],
                         "app": app_name(r["bundle_id"]), "how": how, "open": r["end_kind"] == "open"})
        unlocks = [{"ts": r["ts"], "app": app_name(r["first_bundle_id"]), "notification": r["notification"],
                    "gap_min": round((r["gap_s"] or 0) / 60, 1)}
                   for r in c.execute("""SELECT ts, first_bundle_id, gap_s, notification FROM unlocks
                                         WHERE ts >= ? AND ts < ? AND excluded_reason IS NULL ORDER BY ts""", (t0, t1))]
        thr = c.execute("SELECT MAX(COALESCE(end_ts, start_ts)) FROM segments").fetchone()[0]
        return {"segments": segs, "unlocks": unlocks, "through": float(thr) if thr else None, "available": True}
    except sqlite3.Error as e:
        log.warning("phone db: %s", e)
        return {"segments": [], "unlocks": [], "through": None, "available": False}
    finally:
        c.close()

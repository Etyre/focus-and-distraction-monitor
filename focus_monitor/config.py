from __future__ import annotations

import datetime as _dt
import os
import time as _time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.environ.get("FOCUS_MONITOR_DATA", Path.home() / "Library/Application Support/FocusMonitor")
)
CONFIG_PATH = Path(os.environ.get("FOCUS_MONITOR_CONFIG", PROJECT_DIR / "config.toml"))
# The app log (timestamped, rotated). Overridable so a test run does not write into the real log.
LOG_PATH = Path(os.environ.get("FOCUS_MONITOR_LOG", Path.home() / "Library/Logs/focus-and-distraction-monitor.log"))

# Dev restart: the Python process exits with this status and the .app launcher (build_app.py)
# relaunches it at once, outside its crash budget. The launcher hard-codes the same number.
RESTART_EXIT_CODE = 75
# Written just before a dev restart when the dashboard window is open; the next launch reads it
# and reopens the same tab.
REOPEN_FILE = DATA_DIR / "reopen_dashboard.json"

CHROMIUM_APPS = {"Google Chrome", "Arc", "Brave Browser", "Microsoft Edge", "Chromium", "Vivaldi"}
SAFARI_APPS = {"Safari"}

# Claude Opus 5 first-party pricing, USD per token
PRICE_INPUT = 4.0 / 1e6      # Claude Opus 5.5 (Opus 5, used until 2026-11-16, was 5 / 25 / 6.25 / 0.5)
PRICE_OUTPUT = 20.0 / 1e6
PRICE_CACHE_WRITE = 5.0 / 1e6
PRICE_CACHE_READ = 0.2 / 1e6


@dataclass
class Config:
    poll_interval: float = 2.0
    min_segment_seconds: float = 3.0
    idle_seconds: float = 600.0            # no keyboard/mouse input this long => away; breaks focus spans
    short_inactivity_seconds: float = 60.0  # shorter pauses (>= this) are logged as short breaks only
    collect_signals: bool = True           # input counts, audio/mic/camera, click targets, banners, Chrome page visits (signals.py)
    refresh_screenshot_seconds: float = 60.0
    screenshot_max_width: int = 1280
    screenshot_quality: int = 60
    # Full-size captures are deleted after this many days; 0 keeps them for good. Kept for good since
    # 2026-11-11: every capture is raw data for re-running the evaluator over the past once it
    # predicts Eli's judgments well; after that rerun the full-size tier can go and the archive stays.
    screenshot_retention_days: int = 0
    # A smaller copy of every capture is kept for good (see ARCHIVE_DIR) - except captures that
    # show a window of one of these apps, which are deleted from both tiers after
    # private_retention_days whatever the setting above. Signal chats with disappearing messages
    # must not outlive the conversation itself.
    archive_max_width: int = 640
    archive_quality: int = 60
    private_apps: list[str] = field(default_factory=lambda: ["Signal"])
    private_retention_days: int = 14

    # classification
    lookahead_seconds: float = 120.0
    transit_seconds: float = 20.0   # windows held shorter than this while moving are 'transit'; the whole
                                    # hunt (desktops -> window -> tab) is judged once as one transition  # wait this long after a switch so we can see what followed
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    daily_budget_usd: float = 40.0     # generous; per-switch Opus is gated against it,
    total_budget_usd: float = 500.0    # cumulative ceiling: flag the user when crossed
    hard_budget_multiple: float = 2.0  # but judgment work only stops at this multiple (safety)
    switch_claude_budget_fraction: float = 0.35  # per-switch Opus stops here; rest goes to the evaluator
    # Deferral: skip the LLM for a segment when the local models (heuristic+learned) are this
    # confident AND their track record on the user's edits has earned it.
    defer_confidence: float = 0.85
    defer_accuracy: float = 0.85
    defer_min_feedback: int = 25
    daily_audit: bool = True           # one end-of-day LLM sweep re-checks the day, flags errors
    # Which account pays for the judgment calls: "api" = the API key, per token; "subscription" =
    # the person's claude.ai login through the Claude Code binary (Agent SDK). On the subscription,
    # a usage-limit error switches to the API key until the limit resets (see llm.py).
    llm_backend: str = "api"
    subscription_retry_min: float = 30.0   # retry the subscription this long after a limit error
                                           # when the error does not name its reset time
    # frugality: skip Claude when local models are this accurate on recent reviews AND this confident
    frugal_accuracy_threshold: float = 0.85
    frugal_confidence_threshold: float = 0.85
    frugal_min_reviews: int = 40
    audit_sample_rate: float = 0.10  # still send this fraction to Claude for calibration
    count_events: bool = False                # phase 2: show interruption/distraction counts. Off for now -
                                              # the goal is accurate focus spans (stable vs. broken).
    break_min: float = 5.0                    # a detour (interrupted/distracted) shorter than this stays INSIDE
                                              # the span as an event; this long or longer breaks the span
    min_unbroken_min: float = 10.0             # a span needs one continuous block this long with no
                                               # contained interruption, or it is not a focus span
    min_focus_span_min: float = 10.0           # attention must be on a task this long to count as a span,
                                              # and only leaving such a span counts as an interruption/distraction
    # Spot checks: this many CONFIDENT judgments a day are drawn at random and put to the person as
    # "is this right?" - the only unbiased measure of how often the app is right (the questions it
    # asks otherwise are about segments it already doubted). 0 = off.
    spot_checks_per_day: int = 3
    question_archive_days: float = 3.0        # an open review question this old is archived (kept, out of the
                                              # queue): past ~3 days it will not get answered. 0 = never
    max_reviews_per_day: int = 8              # per day, review the top-N by rank (uncertainty x stakes)
    working_pair_window_min: float = 30.0     # A<->B bounced >= 3 times in this window => trivial pair
    uncertain_threshold: float = 0.65
    uncertain_threshold_focus: float = 0.75   # stricter for switches that leave a focus span
    low_priority_budget_fraction: float = 0.6  # switches NOT leaving focus only use Claude below this share of budget
    few_shot_reviews: int = 40

    about_me: str = ""
    phone_db: str = ""     # the phone-unlock-logger's database; empty = its default location
                           # (~/Library/Application Support/PhoneUnlockLogger/phone.db)
    distraction_domains: list[str] = field(default_factory=list)
    blocked_page_patterns: list[str] = field(default_factory=list)
    work_apps: list[str] = field(default_factory=list)
    notes_apps: list[str] = field(default_factory=list)
    # Non-browser apps whose window title is recorded, so a change of title (a different project
    # in Agent Coding) starts a new segment. Other apps keep an empty title.
    title_apps: list[str] = field(default_factory=lambda: ["Agent Coding"])
    # Quick-capture: adding a spare thought to your daily notes is never an interruption/distraction.
    daily_notes_apps: list[str] = field(default_factory=lambda: ["Roam Research", "Logseq"])
    daily_notes_domains: list[str] = field(default_factory=lambda: ["roamresearch.com", "logseq.com"])
    daily_notes_regex: str = r"(?i)daily note|^\w+ \d+(st|nd|rd|th)?,? \d{4}|^\d{4}-\d{2}-\d{2}"
    focus_domains: list[str] = field(default_factory=lambda: ["roamresearch.com", "logseq.com", "notion.so",
                                                              "docs.google.com", "obsidian.md"])
    # Span subtype heuristics: authoring surfaces => creative-work; else task-churn under Toggl => focused-task.
    authoring_apps: list[str] = field(default_factory=lambda: ["Typora", "Code", "Cursor", "Visual Studio Code",
                                      "Xcode", "Pages", "Keynote", "Final Cut Pro", "iMovie", "Figma", "Sketch"])
    authoring_domains: list[str] = field(default_factory=lambda: ["docs.google.com", "overleaf.com"])
    creative_frac: float = 0.40        # >= this share of a span's focus time on authoring => creative-work
    task_toggl_frac: float = 0.60      # Toggl running for >= this share of the span
    reading_notes_frac: float = 0.10   # a reading span requires notes surfaces for >= this share of its
                                       # focus time (Rules doc: reading = reading AND taking notes;
                                       # passive essay-hopping without notes is unfocused)
    task_min_apps: int = 5             # and this many distinct apps/sites (task-churn), not one reading surface
    # Passive-consumption domains: watching here is NEVER focused work (categorical rule) -
    # the state machine forces these segments to 'distracted' unless a human review overrides.
    passive_domains: list[str] = field(default_factory=lambda: ["youtube.com"])
    # Correspondence surfaces (chat + email): a prior toward the 'correspondence' content label.
    correspondence_apps: list[str] = field(default_factory=lambda: ["Signal", "Messages", "WhatsApp", "Telegram",
                                                                    "Slack", "Discord", "Mail", "Superhuman", "Spark"])
    correspondence_domains: list[str] = field(default_factory=lambda: ["mail.google.com", "outlook.live.com",
                                                                       "outlook.office.com", "web.whatsapp.com",
                                                                       "web.telegram.org", "app.slack.com",
                                                                       "discord.com", "messenger.com"])
    ignore_apps: list[str] = field(default_factory=list)
    # Windows that are "neutral": recorded, but never a switch/event and transparent to focus spans
    # (e.g. a presence-check form). Case-insensitive substrings matched against app, title and URL.
    neutral_patterns: list[str] = field(default_factory=list)

    def is_daily_notes(self, app: str, title: str, url: str) -> bool:
        """A daily-notes / journal page in Roam or Logseq - where you log a spare thought."""
        import re
        from urllib.parse import urlparse
        dom = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
        in_app = app in self.daily_notes_apps or any(dom == d or dom.endswith("." + d) for d in self.daily_notes_domains)
        if not in_app:
            return False
        return bool(not title or re.search(self.daily_notes_regex, title))

    def is_authoring(self, app: str, title: str, url: str) -> bool:
        from urllib.parse import urlparse
        dom = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
        return app in self.authoring_apps or any(dom == d or dom.endswith("." + d) for d in self.authoring_domains)

    def is_focus_context(self, app: str, title: str, url: str) -> bool:
        """Is this a work/notes/writing surface - somewhere returning to counts as resuming focus?"""
        from urllib.parse import urlparse
        dom = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
        return (app in self.work_apps or app in self.notes_apps
                or any(dom == d or dom.endswith("." + d) for d in self.focus_domains)
                or self.is_daily_notes(app, title, url))

    def is_neutral(self, app: str, title: str, url: str) -> bool:
        hay = f"{app}\n{title}\n{url}".lower()
        return any(pat.lower() in hay for pat in self.neutral_patterns if pat)

    day_start_hour: int = 4    # a "day" runs from this hour to the same hour next day (4 => 4am-4am)

    web_host: str = "127.0.0.1"
    web_port: int = 8790


def load_config() -> Config:
    cfg = Config()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        for section in data.values() if all(isinstance(v, dict) for v in data.values()) else [data]:
            for k, v in section.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    return cfg


def keychain_get(account: str) -> str | None:
    import subprocess
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", "FocusMonitor", "-a", account, "-w"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def keychain_set(account: str, secret: str) -> None:
    import subprocess
    subprocess.run(["security", "add-generic-password", "-U", "-s", "FocusMonitor", "-a", account, "-w", secret],
                   capture_output=True)


def load_api_key() -> bool:
    """Make ANTHROPIC_API_KEY (and TOGGL_API_TOKEN) available when launched from Finder (no shell env).

    Order: existing env var -> macOS Keychain (service "FocusMonitor") -> DATA_DIR/env file.
    """
    if not os.environ.get("TOGGL_API_TOKEN"):
        t = keychain_get("toggl")
        if t:
            os.environ["TOGGL_API_TOKEN"] = t
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    k = keychain_get("anthropic")
    if k:
        os.environ["ANTHROPIC_API_KEY"] = k
        return True
    env_file = DATA_DIR / "env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


SCREENSHOT_DIR = DATA_DIR / "screenshots"        # full size, pruned after screenshot_retention_days (0 = never)
ARCHIVE_DIR = SCREENSHOT_DIR / "archive"         # archive_max_width copies, kept for good
PRIVATE_MARK = "_priv"                           # filename suffix: shows a private app; never archived past retention


def ensure_dirs() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def screenshot_file(name: str | None) -> Path | None:
    """Where a capture lives now: the full-size file while it is retained, then the archive copy."""
    if not name:
        return None
    base = Path(name).name
    for d in (SCREENSHOT_DIR, ARCHIVE_DIR):
        p = d / base
        if p.exists():
            return p
    return None


# Touched by the collector every poll. The app bundle's native launcher watches its mtime and
# restarts the Python process when it stops ticking (see build_app.py).
HEARTBEAT_PATH = DATA_DIR / "heartbeat"


_DAY_HOUR = None


def day_start_hour() -> int:
    """Cached day-start hour (loaded once; restart to change)."""
    global _DAY_HOUR
    if _DAY_HOUR is None:
        _DAY_HOUR = int(load_config().day_start_hour)
    return _DAY_HOUR


# ---- time zones --------------------------------------------------------------------------------
# Days and clock times follow the zone the Mac was in WHEN the activity happened, not the zone it
# is in now. Until 2026-11-20 everything used the process's current zone: after the Mac moved
# from Pacific to Eastern time on 2026-11-08, every earlier day was cut at 1am as lived instead
# of 4am (16 of 75 days off by 10+ focus minutes), and re-judgments showed old activity three
# hours late. The history is a small file, one entry per change, appended by the collector
# (note_timezone). Without the file (tests, a fresh install) the system zone applies throughout.
TZ_PATH = DATA_DIR / "timezones.json"
_TZ = (None, [])   # (file mtime, [(since, ZoneInfo)])


def _tz_history() -> list:
    global _TZ
    try:
        m = TZ_PATH.stat().st_mtime
    except OSError:
        return []
    if _TZ[0] != m:
        import json
        from zoneinfo import ZoneInfo
        try:
            rows = sorted((float(r["since"]), ZoneInfo(r["tz"])) for r in json.loads(TZ_PATH.read_text()))
        except Exception:   # an unreadable file must not take the app down: fall back to the system zone
            rows = []
        _TZ = (m, rows)
    return _TZ[1]


def zone_at(ts: float):
    """The time zone the Mac was in at `ts` (a ZoneInfo), or None for the system zone."""
    z = None
    for since, zone in _tz_history():
        if since > ts:
            break
        z = zone
    return z


def local_dt(ts: float) -> _dt.datetime:
    """`ts` as the wall-clock time where the Mac was then. Use this, not datetime.fromtimestamp,
    for any time shown to a model or used to decide which day something belongs to."""
    z = zone_at(ts)
    return _dt.datetime.fromtimestamp(ts, z).replace(tzinfo=None) if z is not None else _dt.datetime.fromtimestamp(ts)


def system_zone_name() -> str | None:
    try:
        p = os.path.realpath("/etc/localtime")
        return p.split("zoneinfo/", 1)[1] if "zoneinfo/" in p else None
    except OSError:
        return None


def note_timezone(now: float | None = None) -> str | None:
    """Record the system time zone if it differs from the last one recorded (called by the
    collector every minute or so). Returns the new zone's name when it changed."""
    import json
    name = system_zone_name()
    if not name:
        return None
    try:
        rows = json.loads(TZ_PATH.read_text()) if TZ_PATH.exists() else []
    except Exception:
        return None     # never overwrite a file we could not read
    if rows and sorted(rows, key=lambda r: r["since"])[-1]["tz"] == name:
        return None
    rows.append({"since": 0.0 if not rows else float(now if now is not None else _time.time()), "tz": name})
    tmp = TZ_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(rows, key=lambda r: r["since"]), indent=1))
    os.replace(tmp, TZ_PATH)
    return name


def logical_date(ts: float | None = None) -> _dt.date:
    """The logical day a timestamp belongs to, where days run from day_start_hour to the same hour
    next day. 1am on May 12 (with a 4am start) belongs to May 11. In the zone of the time."""
    ts = _time.time() if ts is None else ts
    return (local_dt(ts) - _dt.timedelta(hours=day_start_hour())).date()


def _day_start(day: _dt.date) -> float:
    t = _dt.datetime.combine(day, _dt.time(hour=day_start_hour()))
    guess = t.timestamp()                      # in the system zone
    for _ in range(2):                         # settle on the zone in force at that moment
        z = zone_at(guess)
        if z is None:
            return t.timestamp()
        guess = t.replace(tzinfo=z).timestamp()
    return guess


def day_bounds(day: _dt.date) -> tuple[float, float]:
    """[day_start_hour on `day`, day_start_hour on the next day), each in the zone the Mac was in
    then. Not start + 24 h: a clock-change day is 23 or 25 hours, and a day the Mac changed zone
    is as long as it was lived (2026-11-08, Pacific to Eastern: 21 hours)."""
    return _day_start(day), _day_start(day + _dt.timedelta(days=1))

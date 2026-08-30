from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.environ.get("FOCUS_MONITOR_DATA", Path.home() / "Library/Application Support/FocusMonitor")
)
CONFIG_PATH = Path(os.environ.get("FOCUS_MONITOR_CONFIG", PROJECT_DIR / "config.toml"))

CHROMIUM_APPS = {"Google Chrome", "Arc", "Brave Browser", "Microsoft Edge", "Chromium", "Vivaldi"}
SAFARI_APPS = {"Safari"}

# Claude Opus 5 first-party pricing, USD per token
PRICE_INPUT = 5.0 / 1e6
PRICE_OUTPUT = 25.0 / 1e6
PRICE_CACHE_WRITE = 6.25 / 1e6
PRICE_CACHE_READ = 0.5 / 1e6


@dataclass
class Config:
    poll_interval: float = 2.0
    min_segment_seconds: float = 3.0
    idle_seconds: float = 600.0            # no keyboard/mouse input this long => away; breaks focus spans
    short_inactivity_seconds: float = 60.0  # shorter pauses (>= this) are logged as short breaks only
    refresh_screenshot_seconds: float = 60.0
    screenshot_max_width: int = 1280
    screenshot_quality: int = 60
    screenshot_retention_days: int = 14

    # classification
    lookahead_seconds: float = 120.0
    transit_seconds: float = 20.0   # windows held shorter than this while moving are 'transit'; the whole
                                    # hunt (desktops -> window -> tab) is judged once as one transition  # wait this long after a switch so we can see what followed
    claude_model: str = "claude-opus-5"
    claude_effort: str = "medium"
    daily_budget_usd: float = 15.0
    # frugality: skip Claude when local models are this accurate on recent reviews AND this confident
    frugal_accuracy_threshold: float = 0.85
    frugal_confidence_threshold: float = 0.85
    frugal_min_reviews: int = 40
    audit_sample_rate: float = 0.10  # still send this fraction to Claude for calibration
    count_events: bool = False                # phase 2: show interruption/distraction counts. Off for now -
                                              # the goal is accurate focus spans (stable vs. broken).
    break_min: float = 5.0                    # a detour (interrupted/distracted) shorter than this stays INSIDE
                                              # the span as an event; this long or longer breaks the span
    min_focus_span_min: float = 10.0           # attention must be on a task this long to count as a span,
                                              # and only leaving such a span counts as an interruption/distraction
    max_reviews_per_day: int = 8              # per day, review the top-N by rank (uncertainty x stakes)
    working_pair_window_min: float = 30.0     # A<->B bounced >= 3 times in this window => trivial pair
    uncertain_threshold: float = 0.65
    uncertain_threshold_focus: float = 0.75   # stricter for switches that leave a focus span
    low_priority_budget_fraction: float = 0.6  # switches NOT leaving focus only use Claude below this share of budget
    few_shot_reviews: int = 40

    about_me: str = ""
    distraction_domains: list[str] = field(default_factory=list)
    blocked_page_patterns: list[str] = field(default_factory=list)
    work_apps: list[str] = field(default_factory=list)
    notes_apps: list[str] = field(default_factory=list)
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
    task_min_apps: int = 5             # and this many distinct apps/sites (task-churn), not one reading surface
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


def ensure_dirs() -> None:
    (DATA_DIR / "screenshots").mkdir(parents=True, exist_ok=True)

"""Focus & Distraction Monitor - entry point (`python -m focus_monitor <command>`).

Watches the active window on this Mac, screenshots on every window switch, and has an ensemble of
models (Claude vision + rules + a learned model trained on the person's reviews) judge whether each
switch continued focus, interrupted it, was a self-distraction, or started a new task. A menu bar
eye (`app`) runs the collector, the classifier and the dashboard (http://127.0.0.1:8790) in one
process; `run` is the headless equivalent; `collect` / `classify` / `web` run the pieces alone.

Config:  ./config.toml (next to this package; FOCUS_MONITOR_CONFIG overrides)
Secrets: Keychain service "FocusMonitor" (accounts "anthropic", "toggl"), else
         ~/Library/Application Support/FocusMonitor/env
Data:    ~/Library/Application Support/FocusMonitor/ (focus.db, screenshots/, launcher.log;
         FOCUS_MONITOR_DATA overrides)
Log:     ~/Library/Logs/focus-and-distraction-monitor.log (FOCUS_MONITOR_LOG overrides)
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import threading

from .config import DATA_DIR, LOG_PATH, load_api_key, load_config, screenshot_file


def main():
    ap = argparse.ArgumentParser(prog="focus_monitor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect", help="watch active window and record segments/switches")
    c = sub.add_parser("classify", help="classify pending switches (loops)")
    c.add_argument("--once", action="store_true")
    c.add_argument("--no-claude", action="store_true")
    sub.add_parser("web", help="serve the dashboard")
    a = sub.add_parser("app", help="menu bar app (eye icon) running collector + classifier + web")
    a.add_argument("--no-claude", action="store_true")
    r = sub.add_parser("run", help="headless: collector + classifier + web in one process")
    r.add_argument("--no-claude", action="store_true")
    s = sub.add_parser("stats", help="print daily metrics")
    s.add_argument("--days", type=int, default=7)
    sub.add_parser("check", help="check permissions / API access")
    sub.add_parser("rejudge-wrong-display", help="queue stretches judged from the wrong monitor's screenshots; "
                                                 "the running app re-judges them in the background")
    rh = sub.add_parser("rejudge-history", help="status of the re-judgment of everything judged at low effort "
                                                "(before 2026-11-20 03:34); --start / --pause / --resume / --every SECONDS")
    rh.add_argument("--start", action="store_true", help="queue the days (once); the running app works through them")
    rh.add_argument("--pause", action="store_true")
    rh.add_argument("--resume", action="store_true")
    rh.add_argument("--every", type=float, help="least seconds between two history calls (default 60)")
    rs = sub.add_parser("restart", help="ask the running menu bar app to relaunch itself (picks up source edits)")
    rs.add_argument("--if-stale", action="store_true", help="only when a source file is newer than the running process")
    rs.add_argument("--hook", action="store_true", help="print Claude Code hook JSON (silent when nothing happened)")
    rs.add_argument("--when-idle", action="store_true",
                    help="wait until there has been no keyboard or mouse input for 2 minutes, then restart "
                         "if the source is newer than the running process (run it in the background)")
    args = ap.parse_args()

    if args.cmd == "restart":
        if args.when_idle:
            wait_for_idle()
        return restart_app(if_stale=args.if_stale or args.when_idle, hook=args.hook)

    from .config import ensure_dirs
    ensure_dirs()
    from logging.handlers import RotatingFileHandler
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=2)])
    cfg = load_config()
    load_api_key()

    if args.cmd == "collect":
        from .collector import Collector
        Collector(cfg).run()
    elif args.cmd == "classify":
        from .classify import ClassifierService
        svc = ClassifierService(cfg, use_claude=not args.no_claude)
        print(f"classified {svc.run_once()}") if args.once else svc.run()
    elif args.cmd == "web":
        from .web import serve
        print(f"dashboard: http://{cfg.web_host}:{cfg.web_port}")
        serve()
    elif args.cmd == "app":
        from .menubar import run_app
        print(f"dashboard: http://{cfg.web_host}:{cfg.web_port}   data: {DATA_DIR}")
        run_app(use_claude=not args.no_claude)
    elif args.cmd == "run":
        from .classify import ClassifierService
        from .collector import Collector
        from .web import serve
        threading.Thread(target=Collector(cfg).run, daemon=True, name="collector").start()
        threading.Thread(target=ClassifierService(cfg, use_claude=not args.no_claude).run, daemon=True,
                         name="classifier").start()
        print(f"dashboard: http://{cfg.web_host}:{cfg.web_port}   data: {DATA_DIR}")
        serve()
    elif args.cmd == "stats":
        from . import db, stats
        conn = db.connect()
        print(f"{'day':10} {'longest':>8} {'total':>8} {'intr':>5} {'distr':>5} {'d-min':>6} {'active':>7} {'$':>6}")
        for m in stats.metrics_range(conn, args.days):
            print(f"{m['day']:10} {m['longest_focus_min']:8.0f} {m['total_focus_min']:8.0f} {m['interruption_count']:5d} "
                  f"{m['distraction_count']:5d} {m['distracted_min']:6.0f} {m['active_min']:7.0f} {m['spend_usd']:6.2f}")
    elif args.cmd == "rejudge-history":
        from . import db, evaluator
        c = db.connect()
        if args.start:
            print("queued", evaluator.queue_history_rejudge(c), "day(s)")
        if args.pause or args.resume:
            db.kv_set(c, "rejudge.history.paused", "1" if args.pause else "0")
        if args.every is not None:
            db.kv_set(c, "rejudge.history.every_s", str(max(args.every, 0.0)))
        c.commit()
        st = evaluator.history_status(c)
        print(f"{st['days'] - st['days_left']} of {st['days']} days done, {st['calls']} calls so far"
              + (f"; now at {st['working_on']}" if st["working_on"] else "")
              + ("; PAUSED" if st["paused"] else ""))
    elif args.cmd == "rejudge-wrong-display":
        from . import db, evaluator
        print("queued", evaluator.queue_wrong_display_rejudge(db.connect()), "stretch(es); the running app works through them")
    elif args.cmd == "check":
        from .collector import current_window, idle_seconds, take_screenshot
        w = current_window(cfg.title_apps)
        print("active window:", w or "FAILED - grant Accessibility/Automation permission to your terminal")
        print("idle seconds:", idle_seconds())
        shot = take_screenshot(cfg, "check")
        print("screenshot:", screenshot_file(shot) if shot else "FAILED - grant Screen Recording permission")
        try:
            import anthropic
            client = anthropic.Anthropic()
            m = client.models.retrieve(cfg.claude_model)
            print("claude:", m.id, "ok")
        except Exception as e:
            print("claude: FAILED -", e)




RESTART_IDLE_S = 120.0   # no input this long = nobody is mid-sentence in the dashboard


def wait_for_idle() -> None:
    """Block until the keyboard and mouse have been untouched for RESTART_IDLE_S. A restart only
    resets which page is open (typed text is saved as drafts), but doing it while the person is
    away means they never see even that. Several waiters are harmless: --when-idle restarts only
    if the source is newer than the process, so the later ones find it up to date."""
    import time
    from .collector import idle_seconds
    while idle_seconds() < RESTART_IDLE_S:
        time.sleep(15)


def restart_app(if_stale: bool, hook: bool) -> None:
    """POST /api/restart on the running app and report. With --hook, emit Claude Code hook output:
    a systemMessage when the app restarted or refused to, nothing when there was nothing to do."""
    import json
    import urllib.error
    import urllib.request

    cfg = load_config()
    url = f"http://{cfg.web_host}:{cfg.web_port}/api/restart"
    body = json.dumps({"only_if_stale": if_stale}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        res = json.load(e)
    except (urllib.error.URLError, OSError):
        res = {"restarted": False, "reason": "app not running"}

    reason = res.get("reason") or res.get("detail") or "unexpected response"
    if res.get("restarted"):
        msg = "Focus Monitor restarted to pick up source changes" + \
              (" (dashboard reopened)" if res.get("reopen") else "")
    elif reason in ("up to date", "app not running"):
        msg = None if hook else f"not restarted: {reason}"
    else:
        msg = f"Focus Monitor NOT restarted: {reason}"
    if msg is None:
        return
    print(json.dumps({"systemMessage": msg}) if hook else msg)


if __name__ == "__main__":
    main()

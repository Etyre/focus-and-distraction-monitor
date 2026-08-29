from __future__ import annotations

import argparse
import datetime as dt
import logging
import threading

from .config import DATA_DIR, load_api_key, load_config


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
    args = ap.parse_args()

    from .config import ensure_dirs
    ensure_dirs()
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  RotatingFileHandler(DATA_DIR / "app.log", maxBytes=5_000_000, backupCount=2)])
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
    elif args.cmd == "check":
        from .collector import current_window, idle_seconds, take_screenshot
        w = current_window()
        print("active window:", w or "FAILED - grant Accessibility/Automation permission to your terminal")
        print("idle seconds:", idle_seconds())
        shot = take_screenshot(cfg, "check")
        print("screenshot:", (DATA_DIR / "screenshots" / shot) if shot else "FAILED - grant Screen Recording permission")
        try:
            import anthropic
            client = anthropic.Anthropic()
            m = client.models.retrieve(cfg.claude_model)
            print("claude:", m.id, "ok")
        except Exception as e:
            print("claude: FAILED -", e)


if __name__ == "__main__":
    main()

"""Self-test of the iPhone screen captures as this app reads them (phone.captures and what is
built on it), on a throwaway phone database and picture folder. No phone, no model call.

    .venv/bin/python -m focus_monitor.phone_selftest
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

from . import config, evaluator, phone


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "phone.db"
        c = sqlite3.connect(db)
        c.executescript("""
            CREATE TABLE segments (start_ts REAL, end_ts REAL, duration_s REAL, bundle_id TEXT,
                                   transition_reason TEXT, end_kind TEXT);
            CREATE TABLE unlocks (ts REAL, first_bundle_id TEXT, gap_s REAL, notification TEXT, excluded_reason TEXT);
            CREATE TABLE captures (id INTEGER PRIMARY KEY, ts REAL, ts_local TEXT, last_seen_ts REAL,
                                   file TEXT, private INTEGER DEFAULT 0);""")
        T = time.time() - 7200
        segs = [(T, T + 100, "com.apple.mobilesafari", "explicit"), (T + 100, T + 160, "org.whispersystems.signal", "explicit"),
                (T + 160, T + 300, "com.google.Gmail", "explicit"),
                (T + 330, T + 400, "com.apple.mobilesafari", "explicit"),      # Safari again after a 30 s gap: one pickup entry
                (T + 400, T + 460, "ph.telegra.Telegraph", "explicit"),        # private by name ("Telegraph" below)
                (T + 460, T + 520, "org.whispersystems.signal", "lost"),       # estimated end: really in front until T+600
                (T + 600, T + 700, "com.apple.mobilesafari", "explicit"),
                (T + 700, T + 800, "com.apple.mobilenotes", "implicit")]       # no real end record
        for a, b, bundle, kind in segs:
            c.execute("INSERT INTO segments VALUES (?,?,?,?,?,?)", (a, b, b - a, bundle, "", kind))
        caps = [(T + 10, "d/a.jpg", 0),          # Safari
                (T + 50, "d/b.jpg", 0),          # Safari
                (T + 93, "d/b2.jpg", 0),         # Safari, but within the margin of Signal coming forward
                (T + 120, "d/c.jpg", 0),         # Signal, which the logger has NOT flagged yet
                (T + 130, "d/e_priv.jpg", 1),    # Signal, flagged
                (T + 200, "d/f.jpg", 0),         # Gmail
                (T + 298, "d/f2.jpg", 0),        # asked for in Gmail, back after Gmail's end
                (T + 315, "d/gap.jpg", 0),       # between two Safari stretches: lock screen, switcher, anything
                (T + 350, "d/h.jpg", 0),         # Safari
                (T + 430, "d/tg.jpg", 0),        # the app private by name
                (T + 560, "d/lost.jpg", 0),      # after Signal's ESTIMATED end
                (T + 650, "d/i.jpg", 0),         # Safari
                (T + 750, "d/n.jpg", 0),         # an app segment with no real end
                (T + 900, "d/late.jpg", 0)]      # after the phone log ends: app unknown
        (Path(d) / "captures" / "d").mkdir(parents=True)
        for ts, f, priv in caps:
            c.execute("INSERT INTO captures (ts, ts_local, last_seen_ts, file, private) VALUES (?,?,?,?,?)", (ts, "", ts + 2, f, priv))
            (Path(d) / "captures" / f).write_bytes(b"\xff\xd8\xff\xd9")
        c.commit()
        c.close()

        cfg = config.Config()
        cfg.phone_db = str(db)
        cfg.private_apps = ["Signal", "telegraph", "NoSuchApp"]
        config.load_config = lambda: cfg      # screenshot_file looks the folder up through the config

        want = ["iphone~d~a.jpg", "iphone~d~b.jpg", "iphone~d~f.jpg", "iphone~d~h.jpg", "iphone~d~i.jpg"]
        got = [k["name"] for k in phone.captures(cfg, T - 60, T + 2000)]
        assert got == want, got
        for n in got:
            assert config.screenshot_file(n) is not None, n
        withheld = [f"iphone~d~{f.split('/')[1]}" for _, f, _ in caps if f"iphone~d~{f.split('/')[1]}" not in want]
        for bad in withheld + ["iphone~d~e_PRIV.jpg", "iphone~..~phone.db", "iphone~d~missing.jpg", "iphone~d", "iphone~d~x~y"]:
            assert config.screenshot_file(bad) is None, bad
        cfg.private_apps = ["Signal", "telegraph", "Gmail"]   # an app made private later: its stored names stop resolving
        assert config.screenshot_file("iphone~d~f.jpg") is None
        cfg.private_apps = ["Signal", "telegraph"]

        w = phone.window(cfg, T - 60, T + 2000)
        shots = [(s["app"], [k[0] for k in s["shots"]]) for s in w["segments"] if s["shots"]]
        assert shots == [("Safari", want[:2]), ("Gmail", want[2:3]), ("Safari", want[3:4]), ("Safari", want[4:])], shots

        ims = [i for p in phone.pickups(cfg, T - 60, T + 2000) for i in evaluator._phone_images(cfg, p)[0]]
        assert [i["path"] for i in ims] == want, [i["path"] for i in ims]   # the gap capture is not in the merged Safari entry
        assert all("the iPhone's screen during PHONE id=P" in i["label"] for i in ims)
        content, meta = evaluator._image_content(ims)
        assert len(meta) == 5 and sum(1 for b in content if b["type"] == "image") == 5
        (Path(d) / "captures" / "d" / "a.jpg").unlink()       # a file gone since: not attached, so not counted
        assert len(evaluator._image_content(ims)[1]) == 4

        # ---- a pickup judged without its pictures is queued for one re-judgment; others are not
        (Path(d) / "captures" / "d" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        app = sqlite3.connect(":memory:", isolation_level=None)
        app.row_factory = sqlite3.Row
        app.executescript("""
            CREATE TABLE phone_evals (seg_key INTEGER PRIMARY KEY, pickup_key INTEGER, start REAL, end REAL, app TEXT,
                                      switch_label TEXT, run_id INTEGER);
            CREATE TABLE phone_reviews (seg_key INTEGER PRIMARY KEY);
            CREATE TABLE eval_runs (id INTEGER PRIMARY KEY, shots TEXT);
            CREATE TABLE reeval_queue (id INTEGER PRIMARY KEY, t0 REAL, t1 REAL, reason TEXT, created REAL);
            CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO eval_runs VALUES (1, '[{"path": "20261123_d0.jpg", "label": "x"}]');""")
        picks = phone.pickups(cfg, T - 60, T + 2000)
        assert len(picks) >= 2, len(picks)
        for p in picks:
            for g in p["segs"]:
                app.execute("INSERT INTO phone_evals VALUES (?,?,?,?,?,?,?)",
                            (g["key"], p["key"], g["start"], g["end"], g["app"], "interruption", 1))
        have = lambda: {r["seg_key"]: r for r in app.execute("SELECT * FROM phone_evals")}
        with_pics = [p for p in picks if evaluator._phone_images(cfg, p)[0]]
        assert with_pics and all(evaluator._phone_unseen(app, cfg, p, have()) for p in with_pics)
        assert not any(evaluator._phone_unseen(app, cfg, p, have()) for p in picks if p not in with_pics)
        evaluator._phone_shots_checked = 0.0
        n = evaluator.queue_phone_capture_rejudge(app, cfg)
        q = app.execute("SELECT reason FROM reeval_queue").fetchall()
        assert n == len(q) >= 1 and all(r[0] == evaluator.PHONE_SHOT_REASON for r in q), (n, q)
        evaluator._phone_shots_checked = 0.0
        assert evaluator.queue_phone_capture_rejudge(app, cfg) == 0            # each pickup once
        k0 = with_pics[0]["segs"][0]["key"]
        app.execute("UPDATE eval_runs SET shots = ?", (json.dumps([{"path": "iphone~d~a.jpg",
                    "label": f"Screenshot - the iPhone's screen during PHONE id=P{k0} (taken 10:00:00, iPhone: Safari):"}]),))
        assert not evaluator._phone_unseen(app, cfg, with_pics[0], have())             # its run saw ITS screen
        assert all(evaluator._phone_unseen(app, cfg, p, have()) for p in with_pics[1:])  # another pickup's picture does not count
        app.execute("UPDATE eval_runs SET shots = '[]'")
        app.execute("INSERT INTO phone_reviews SELECT seg_key FROM phone_evals")
        assert not any(evaluator._phone_unseen(app, cfg, p, have()) for p in picks)   # the person has ruled
        app.execute("DELETE FROM phone_reviews")
        app.execute("UPDATE phone_evals SET switch_label = NULL")
        assert not any(evaluator._phone_unseen(app, cfg, p, have()) for p in picks)   # not judged yet: the ordinary path

        # ---- the chat tool hands out the same vetted pictures, and says so when there are none
        import datetime as _dt
        from . import chat_tools
        day = config.logical_date(T)
        hm = lambda t: _dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")
        res = chat_tools._screenshots(app, cfg, {"which": "phone", "start_hhmm": hm(T - 30), "end_hhmm": hm(T + 1000)}, day)
        assert isinstance(res, list) and sum(1 for b in res if b["type"] == "image") == 5, res if isinstance(res, str) else len(res)
        res = chat_tools._screenshots(app, cfg, {"which": "phone", "start_hhmm": hm(T + 100), "end_hhmm": hm(T + 160)}, day)
        assert isinstance(res, str) and res.startswith("no picture"), res

        # a phone database from before the capture job: no table, no pictures, no error
        c = sqlite3.connect(db)
        c.execute("DROP TABLE captures")
        c.commit()
        c.close()
        assert phone.captures(cfg, T - 60, T + 1000) == []
        assert all(not s.get("shots") for s in phone.window(cfg, T - 60, T + 1000)["segments"])
    print("ok: phone captures - only those wholly inside a known, non-private app segment are handed out, resolved and attached")


if __name__ == "__main__":
    main()

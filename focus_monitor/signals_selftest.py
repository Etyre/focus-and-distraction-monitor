"""Checks of signals.py that need no live Mac state: where a click lands, how Chrome's
transition values are read, and the notes built for a segment. Run: python -m focus_monitor.signals_selftest"""
from __future__ import annotations

import json
import os
import sys
import tempfile


def main() -> int:
    os.environ["FOCUS_MONITOR_DATA"] = tempfile.mkdtemp(prefix="fm-signals-")
    from . import db, signals
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f"\n          got  {got!r}\n          want {want!r}"))

    signals._display_sizes = lambda: {(1728.0, 1117.0)}
    W = lambda owner, layer, x, y, w, h: {"kCGWindowOwnerName": owner, "kCGWindowLayer": layer, "kCGWindowAlpha": 1.0,
                                           "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}}
    wins = [W("Superwhisper", 30, 0, 0, 1728, 1117),          # invisible full-screen overlay
            W("Superwhisper", 27, 744, 495, 240, 160),        # small floating indicator, click-through
            W("Notification Center", 21, 0, 0, 1728, 1117),   # the banner host: whole display
            W("Dock", 20, 400, 1040, 900, 77),
            W("Signal", 0, 100, 100, 800, 600), W("Google Chrome", 0, 0, 30, 1728, 1000),
            W("Notification Center", -2147483601, 8, 180, 360, 180)]   # a desktop widget
    check("a click on the Dock", signals.window_at(600, 1060, wins), ("Dock", 20))
    check("a click on the front app, under a full-screen overlay", signals.window_at(300, 300, wins), ("Signal", 0))
    check("a click under another app's small floating window", signals.window_at(800, 550, wins), ("Signal", 0))
    check("a click on the app behind", signals.window_at(1200, 500, wins), ("Google Chrome", 0))
    check("a click in the banner corner while a banner shows", signals.window_at(1600, 80, wins), ("Notification Center", 21))
    check("a click elsewhere while a banner shows is not a banner click", signals.window_at(300, 80, wins), ("Google Chrome", 0))
    check("a banner is showing", signals.banner_showing(wins), True)
    check("desktop widgets alone are not a banner", signals.banner_showing([w for w in wins if w["kCGWindowLayer"] != 21]), False)

    here = [W("Atlas", 0, 0, 30, 1728, 1000)]
    check("a click is placed on the window that was there before it", signals.place_click(300, 300, wins, here), ("Signal", 0))
    check("... or, where nothing was, on the one there now", signals.place_click(300, 300, [], here), ("Atlas", 0))
    w = signals.Watcher(); w.clicks.append((100.0, "Dock", 20)); w.keys.append(99.0); w.banners.append([90.0, 95.0])
    check("what happened before a switch", w.before(101.0),
          {"click_owner": "Dock", "click_layer": 20, "click_age": 1.0, "key_age": 2.0, "banner_age": 11.0})

    T, AB, FB = 1, 0x02000000, 0x01000000
    check("typed in the address bar", signals._how(T | AB, False, False), "typed")
    check("'typed' without the address bar = from another app", signals._how(T, False, False), "external")
    check("link with an opener = new tab from a page", signals._how(0, False, True), "new_tab")
    check("link from a page", signals._how(0, True, False), "link")
    check("'reload' is a single-page site moving about", signals._how(8, False, False), "in_page")
    check("back / forward wins over the core type", signals._how(T | AB | FB, False, False), "back_forward")

    c = db.connect()
    t = 1_800_000_000.0
    for i in range(6):   # one minute: 120 keys, 6 clicks, 60 scrolls; audio on for half
        c.execute("INSERT INTO activity_samples(ts, secs, keys, clicks, scrolls, moves, audio, mic, camera) VALUES (?,?,?,?,?,?,?,?,?)",
                  (t + 10 * i, 10, 20, 1, 10, 50, 10 if i < 3 else 0, 0, 0))
    c.execute("INSERT INTO page_visits(profile, visit_id, ts, url, title, transition, how, from_url) VALUES ('Default',1,?,?,?,?,?,?)",
              (t + 1, "https://x.com/home", "Home", T | AB, "typed", None))
    seg = {"id": 7, "start": t, "end": t + 60, "app": "Google Chrome", "url": "https://x.com/home"}
    put = lambda owner, key_age, banner_age: c.execute(
        "INSERT OR REPLACE INTO segment_arrivals(segment_id, click_owner, click_layer, click_age, key_age, banner_age) VALUES (7,?,?,?,?,?)",
        (owner, 0, 1.0 if owner else None, key_age, banner_age))
    put("Dock", None, None)
    check("notes for a segment", signals.describe(c, seg, "Cursor"),
          ["arrived by: a click on the Dock", "page reached by: typed the address",
           "input: 120 keys/min, 6 clicks/min, 60 scroll events/min", "audio playing 50% of the time"])
    put(None, 1.0, 8.0)
    check("keyboard switch after a banner", signals.describe(c, seg, "Cursor")[:2],
          ["arrived by: the keyboard, no click (Cmd-Tab, Spotlight or a shortcut)", "a notification banner appeared 8 s before"])
    put("Cursor", None, None)
    check("a click in the previous app", signals.describe(c, seg, "Cursor")[0],
          "arrived by: a click inside Cursor (something clicked there brought this forward, e.g. a link)")
    put("Notification Center", None, 3.0)
    check("a click on a banner (not also reported as 'a banner appeared')", signals.describe(c, seg, "Cursor")[:2],
          ["arrived by: a click on a notification banner", "page reached by: typed the address"])
    put(None, None, None)
    check("nothing pressed", signals.describe(c, seg, "Cursor")[0], "arrived by: no click and no key press (something else opened it)")
    put("desktop", None, 50.0)
    check("a click on nothing; a banner from long before is not mentioned", signals.describe(c, seg, "Cursor")[:2],
          ["arrived by: a click where no window was (the desktop)", "page reached by: typed the address"])
    tab = {"id": 8, "start": t + 30, "end": t + 50, "app": "Google Chrome", "url": "https://docs.example/a_b"}
    check("no visit, History not read yet: say nothing", signals.page_reached(c, tab["url"], tab["start"], tab["end"]), None)
    db.kv_set(c, "signals.visits_from", str(t - 1000)); db.kv_set(c, "signals.visits_to", str(t + 100))
    check("no visit, History read since: the tab was already open",
          signals.page_reached(c, tab["url"], tab["start"], tab["end"]), {"how": "already_open"})
    check("another browser's page is never called already open",
          [n for n in signals.describe(c, dict(tab, app="Safari"), "Safari") if n.startswith("page")], [])
    c.execute("INSERT INTO page_visits(profile, visit_id, ts, url, title, transition, how, from_url) VALUES ('Default',2,?,?,?,?,?,?)",
              (t + 31, "https://docs.example/a_b#part", "", 0, "link", "https://mail.example/x"))
    check("an address with _ still matches its #fragment visit",
          signals.page_reached(c, tab["url"], tab["start"], tab["end"]), {"how": "link", "from": "mail.example"})
    check("... and _ is not a wildcard", signals.page_reached(c, "https://docs.example/aXb", t + 30, t + 50), {"how": "already_open"})
    check("same app: no 'arrived by'", signals.describe(c, seg, "Google Chrome")[0], "page reached by: typed the address")
    check("a segment from before signals existed has no notes",
          signals.describe(c, {"id": 99, "start": t - 9000, "end": t - 8900, "app": "Notes", "url": ""}, "Cursor"), [])

    # The away rule's "was this stretch a call" (2026-11-21). 10-second buckets from t2 on.
    t2 = t + 10_000

    def put_media(a, b, cam=0, mic=0, secs=10):
        for k in range(int(a), int(b), 10):
            c.execute("INSERT OR REPLACE INTO activity_samples(ts, secs, keys, clicks, scrolls, moves, audio, mic, camera)"
                      " VALUES (?,?,0,0,0,0,0,?,?)", (t2 + k, secs, mic * secs, cam * secs))
    until = lambda a, b, **kw: (lambda v: None if v is None else round(v - t2))(signals.call_until(c, t2 + a, t2 + b, **kw))
    put_media(0, 1800, cam=1)
    check("camera on throughout: the call runs to now", until(0, 1800), 1800)
    check("... and from a last input in the middle of a bucket", until(605, 1203), 1203)
    put_media(1800, 2600)
    check("camera off since 1800 s: the call ended there", until(0, 2400), 1800)
    c.execute("DELETE FROM activity_samples WHERE ts >= ?", (t2,))
    put_media(0, 720)
    for a in (0, 240, 500):
        put_media(a, a + 20, mic=1)
    check("microphone in three 20-second bursts is dictation, not a call", until(0, 720), None)
    put_media(0, 720, mic=1)
    check("microphone on throughout is a call", until(0, 720), 720)
    check("... but not past the moment the screen locked", until(0, 720, mic_until=t2 + 300), 300)
    check("under two minutes of it does not count", until(0, 100), None)
    c.execute("DELETE FROM activity_samples WHERE ts >= ?", (t2,))
    put_media(0, 600, cam=1)
    check("nothing sampled after 600 s (asleep): the old reading does not carry on", until(0, 3000), 600)
    put_media(600, 720)
    put_media(720, 1800, cam=1)
    check("a two-minute camera break in a call", until(0, 1800), 1800)
    c.execute("DELETE FROM activity_samples WHERE ts >= ?", (t2,))
    put_media(0, 300)
    put_media(300, 720, cam=1)
    check("camera off for the first 5 of 12 minutes: under 80%", until(0, 720), None)
    check("no samples at all", until(5000, 6000), None)
    put_media(0, 720, cam=1, secs=0)
    check("buckets with no sampled seconds are not 'on'", until(0, 720), None)

    # Which browser window is the one on screen (collector.pick_window, 2026-11-21).
    from . import collector
    full, left = (0, 33, 1728, 1117), (0, 33, 864, 1117)
    long_tab = "A very long page title that goes on and on, with a lot of words, until its end - Example Site"
    wins = [("Food order", "Food order", "https://food.example/cart", full),            # Chrome's "front window": stale
            ("Inbox", "Inbox", "https://mail.example/", full),
            ("Inbox (2) - Mail", "Inbox (2) - Mail", "https://mail.example/2", left),
            ("A very long page title that go…til its end - Example Site", long_tab, "https://long.example/", left),
            ("Notes", "Notes", "https://notes.example/a", full),
            ("Notes", "Notes", "https://notes.example/b", left),
            ("blank.example/x", "", "https://blank.example/x", full)]
    pick = lambda *onscreen: collector.pick_window(list(onscreen), wins)
    check("the window on screen is not the browser's front window", pick(("Inbox - Google Chrome", full)), 1)
    check("... and the front window when it is the one on screen", pick(("Food order - Google Chrome", full)), 0)
    check("a bare tab title", pick(("Inbox", full)), 1)
    check("a profile suffix", pick(("Inbox - Google Chrome - Eli (Work)", full)), 1)
    check("the longest title that fits wins", pick(("Inbox (2) - Mail - Google Chrome", left)), 2)
    check("a long title the browser shortens around an ellipsis", pick((long_tab + " - Google Chrome", left)), 3)
    check("the same title twice: bounds decide", pick(("Notes - Google Chrome", left)), 5)
    check("the same title and bounds as nothing else: the browser's order", pick(("Notes - Google Chrome", (5, 5, 50, 50))), 4)
    check("title fits nothing: bounds stand in", pick(("Something else", left)), 2)
    check("no title (no Screen Recording): bounds, in the browser's order", pick(("", full)), 0)
    check("an untitled tab is found by its window title", pick(("blank.example/x - Google Chrome", full)), 6)
    check("no match at all", pick(("Something else", (1, 2, 3, 4))), None)
    check("nothing on screen (another Space)", pick(), None)
    check("a popup in front that fits nothing is skipped", pick(("", (300, 80, 700, 400)), ("Inbox - Google Chrome", full)), 1)
    sep = lambda *xs: "\x1f".join(map(str, xs))
    out = "\x1e".join([sep("A", "B"), sep("A", ""), sep("https://a.example/", "https://b.example/"), sep(0, 33, 864, 1117, 5, 6, 7, 8)])
    check("the all-windows script's output is read", collector._parse_windows(out),
          [("A", "A", "https://a.example/", (0, 33, 864, 1117)), ("B", "", "https://b.example/", (5, 6, 7, 8))])
    check("no windows", collector._parse_windows("\x1e\x1e\x1e"), None)
    check("lists of different lengths", collector._parse_windows("\x1e".join([sep("A", "B"), "A", sep("u", "v"), sep(0, 0, 1, 1, 0, 0, 1, 1)])), None)
    check("a failed script", collector._parse_windows(""), None)
    # browser_tab end to end, with the two readings faked
    scripts = []
    real = collector.osascript, collector._onscreen_windows
    try:
        collector.osascript = lambda script, timeout=3.0: (scripts.append(script), out if script == "ALL" else "https://front.example/\nFront")[1]
        collector._onscreen_windows = lambda app, limit=4: [("B - Google Chrome", (5, 6, 7, 8))]
        check("browser_tab reads the on-screen window's tab", collector.browser_tab("X", "ALL", "FRONT"), ("https://b.example/", ""))
        collector._onscreen_windows = lambda app, limit=4: []
        check("... the front window's when nothing matches, in the same call", (collector.browser_tab("X", "ALL", "FRONT"), scripts), (("https://a.example/", "A"), ["ALL", "ALL"]))
        collector.osascript = lambda script, timeout=3.0: "" if script == "ALL" else "https://front.example/\nFront"
        check("... and through the old script when the new one fails", collector.browser_tab("X", "ALL", "FRONT"), ("https://front.example/", "Front"))
        collector._onscreen_windows = lambda app, limit=4: 1 / 0
        collector.log.disabled = True   # the failure is logged with its traceback: not wanted in this output
        collector.osascript = lambda script, timeout=3.0: out if script == "ALL" else "https://front.example/\nFront"
        check("... or when the window list raises", collector.browser_tab("X", "ALL", "FRONT"), ("https://front.example/", "Front"))
    finally:
        collector.osascript, collector._onscreen_windows = real
        collector.log.disabled = False
    print(f"signals self-test: {'FAILED' if fails else 'passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

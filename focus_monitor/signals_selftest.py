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
            W("Notification Center", 21, 0, 0, 1728, 1117),   # the banner host: whole display
            W("Dock", 20, 400, 1040, 900, 77),
            W("Signal", 0, 100, 100, 800, 600), W("Google Chrome", 0, 0, 30, 1728, 1000),
            W("Notification Center", -2147483601, 8, 180, 360, 180)]   # a desktop widget
    check("a click on the Dock", signals.window_at(600, 1060, wins), ("Dock", 20))
    check("a click on the front app, under a full-screen overlay", signals.window_at(300, 300, wins), ("Signal", 0))
    check("a click on the app behind", signals.window_at(1200, 500, wins), ("Google Chrome", 0))
    check("a click in the banner corner while a banner shows", signals.window_at(1600, 80, wins), ("Notification Center", 21))
    check("a click elsewhere while a banner shows is not a banner click", signals.window_at(300, 80, wins), ("Google Chrome", 0))
    check("a banner is showing", signals.banner_showing(wins), True)
    check("desktop widgets alone are not a banner", signals.banner_showing(wins[2:]), False)

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
    check("same app: no 'arrived by'", signals.describe(c, seg, "Google Chrome")[0], "page reached by: typed the address")
    check("a segment from before signals existed has no notes",
          signals.describe(c, {"id": 99, "start": t - 9000, "end": t - 8900, "app": "Notes", "url": ""}, "Cursor"), [])
    print(f"signals self-test: {'FAILED' if fails else 'passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

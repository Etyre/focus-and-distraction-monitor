"""Self-test of the span rules on hand-built days (no database, no model calls).

    python -m focus_monitor.span_selftest

Each case builds a list of judged segments the way stats.labelled_segments would hand them to
stats.spans_and_events, for segments judged under the 2026-11-19 redesign (a focused/unfocused
field per segment; continuation / shift / return per switch; self-distraction and next-task
flags on a shift), and checks the spans and events against the Rules doc. Run by ./test.sh.
"""
from __future__ import annotations

import sys

from . import stats

T0 = 1_800_000_000.0   # any fixed start; the tests never look at the clock


class Day:
    def __init__(self):
        self.segs: list[dict] = []
        self.t = T0

    def add(self, minutes, what, *, focused=True, switch="continuation", ante=None, sd=False,
            nxt=False, toggl=None, content="creative", user_unfocused=False, domain=None):
        """Append one segment. `what` names the window (app); returns its id."""
        sid = len(self.segs) + 1
        if ante is None and switch == "continuation" and self.segs:
            ante = self.segs[-1]["id"]
        if sd:
            self._sd = True
        elif switch in ("interruption", "return"):
            self._sd = False
        sd_live = getattr(self, "_sd", False)
        state = "focus" if focused else ("distracted" if sd_live else "interrupted")
        if content == "passive":
            state = "distracted" if sd_live else "interrupted"
        label = "continuation"
        if switch == "interruption":
            label = "distraction" if sd else ("task_change" if state == "focus" else "interruption")
        dur = minutes * 60.0
        self.segs.append({
            "id": sid, "app": what, "domain": domain or "", "title": "", "url": "",
            "start": self.t, "end": self.t + dur, "clip_start": self.t, "clip_end": self.t + dur,
            "duration": dur, "state": state, "label": label, "content": content,
            "focused": focused, "switch": switch, "self_distraction": bool(sd and switch in ("interruption", "return")),
            "next_task": bool(nxt and switch == "interruption"), "user_unfocused": user_unfocused,
            "antecedent_id": ante if switch != "interruption" else None, "antecedent_p": 1.0,
            "toggl_id": toggl, "tg_near_a": toggl, "tg_near_b": toggl, "toggl_tags": "", "toggl_desc": "",
            "source": "eval", "neutral": 0, "idle": 0, "hybrid": 0, "switch_id": sid, "uncertain": 0,
        })
        self.t += dur
        return sid

    def run(self):
        return stats.spans_and_events(self.segs, min_focus_min=10, break_min=5)

    def phone(self, at_min, minutes, *, switch="interruption", sd=False, focused=False, apps="Signal 60s"):
        """An iPhone pickup `at_min` minutes into the day, judged as given; carves it out of
        the segments the way stats.labelled_segments does (through an in-memory phone_evals)."""
        import sqlite3
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute("CREATE TABLE phone_evals (pickup_key INTEGER PRIMARY KEY, start REAL, end REAL, apps TEXT,"
                  " switch_label TEXT, content TEXT, focused INTEGER, self_distraction INTEGER, rationale TEXT,"
                  " created REAL, run_id INTEGER)")
        a = T0 + at_min * 60.0
        c.execute("INSERT INTO phone_evals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (int(a), a, a + minutes * 60.0, apps, switch, "correspondence", int(focused), int(sd), "test", 0, None))
        for sg in self.segs:
            sg.setdefault("redesign", True)
        self.segs = stats._carve_phone(c, self.segs, T0, self.t)
        return self


def _mins(sp):
    return round(sp["focus_min"], 1)


def cases():
    out = []

    def case(name):
        def deco(fn):
            out.append((name, fn))
            return fn
        return deco

    @case("a quick focused purchase mid-writing is a diversion, not unfocused time")
    def _():
        d = Day()
        a = d.add(12, "Essay")
        d.add(2, "Amazon", switch="interruption", content="task")
        d.add(10, "Essay", switch="return", ante=a)
        se = d.run()
        assert len(se["focus_spans"]) == 1, se["focus_spans"]
        sp = se["focus_spans"][0]
        assert _mins(sp) == 22 and len(sp["detours"]) == 1, (_mins(sp), sp["detours"])
        assert d.segs[1]["state"] == "focus"            # no "unfocused" mark on the purchase
        assert sp["detours"][0]["state"] == "interrupted"  # a diversion, not a self-distraction

    @case("unfocused time on the span's own thread counts as focus; no question under the thresholds")
    def _():
        d = Day()
        d.add(15, "Essay"); d.add(8, "Essay", focused=False); d.add(20, "Essay")
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 43, [(_mins(x)) for x in sp]
        assert round(sp[0]["unfocused_min"]) == 8 and not sp[0]["unfocused_ask"]
        assert not sp[0]["detours"]

    @case("more than 20% unfocused, or a 10-minute unfocused stretch, raises the question")
    def _():
        d = Day()
        d.add(15, "Essay"); d.add(12, "Essay", focused=False); d.add(20, "Essay")
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and sp[0]["unfocused_ask"] and len(sp[0]["unfocused_seg_ids"]) == 1
        d = Day()
        d.add(12, "Essay"); d.add(4, "Essay", focused=False); d.add(2, "Essay"); d.add(4, "Essay", focused=False)
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and sp[0]["unfocused_ask"], sp   # 8 of 22 minutes = 36%

    @case("the person ruling that time unfocused makes it time away, and 5 minutes of it ends the span")
    def _():
        d = Day()
        d.add(12, "Essay"); d.add(8, "Essay", focused=False, user_unfocused=True); d.add(12, "Essay")
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [12, 12], [_mins(x) for x in sp]

    @case("a span starts with 10 continuous FOCUSED minutes; unfocused time does not start one")
    def _():
        d = Day()
        d.add(4, "Essay"); d.add(7, "Essay", focused=False); d.add(12, "Essay")
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 12, [_mins(x) for x in sp]
        d = Day()
        d.add(30, "Essay", focused=False)
        se = d.run()
        assert not se["focus_spans"] and not se["disqualified"], se

    @case("moving to the next task joins a focused task block only under a running Toggl entry")
    def _():
        d = Day()
        d.add(6, "Mail", content="task", toggl=7)
        d.add(6, "Taxes", content="task", switch="interruption", nxt=True, toggl=7)
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 12 and sp[0]["subtype"] == "focused_task", sp
        d = Day()
        d.add(6, "Mail", content="task")
        d.add(6, "Taxes", content="task", switch="interruption", nxt=True)
        se = d.run()
        assert not se["focus_spans"], se["focus_spans"]

    @case("a glance at a feed between tasks is a diversion from the block")
    def _():
        d = Day()
        d.add(11, "Mail", content="task", toggl=7)
        d.add(1, "Feed", switch="interruption", sd=True, focused=False, content="passive", toggl=7)
        d.add(7, "Taxes", content="task", switch="interruption", nxt=True, toggl=7)
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 18 and len(sp[0]["detours"]) == 1, sp
        assert sp[0]["detours"][0]["state"] == "distracted"   # the glance was a self-distraction
        assert len(se["distractions"]) == 1

    @case("a planned passive break after a span is not a self-distraction")
    def _():
        d = Day()
        d.add(15, "Essay")
        d.add(10, "Video", switch="interruption", focused=False, content="passive")
        se = d.run()
        assert len(se["focus_spans"]) == 1 and not se["distractions"] and len(se["interruptions"]) == 1, se
        assert d.segs[1]["state"] == "interrupted"

    @case("a shift inside the same window is a new thread (the pointer is read before the window)")
    def _():
        d = Day()
        a = d.add(12, "Chrome", domain="docs.google.com")
        d.add(6, "Chrome", domain="docs.google.com", switch="interruption", content="task")
        d.add(11, "Chrome", domain="docs.google.com", switch="return", ante=a)
        sp = d.run()["focus_spans"]
        # 6 minutes away ends the first span where it was left; the essay then starts a new one
        assert [_mins(x) for x in sp] == [12, 11], [_mins(x) for x in sp]

    def flag(d):
        for sg in d.segs:
            sg["redesign"] = True

    @case("a bounce-away is the shift that leaves the span; later self-distractions are events only")
    def _():
        d = Day()   # writing -> feed (self-distraction) -> shop (another) -> writing
        a = d.add(12, "Essay")
        d.add(1, "Feed", switch="interruption", focused=False, sd=True, content="passive")
        d.add(1, "Shop", switch="interruption", focused=False, sd=True, content="passive")
        d.add(10, "Essay", switch="return", ante=a)
        d.add(8, "Video", switch="interruption", focused=False, sd=True, content="passive")   # leaves the span
        d.add(3, "Video2", switch="interruption", focused=False, sd=True, content="passive")  # after it ended
        flag(d)
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and len(sp[0]["detours"]) == 1, sp     # one attention-interruption event
        assert stats.self_distraction_counts(d.segs, se) == (4, 2)  # 4 events; 2 leavings, both bounce-aways

    @case("leaving on a deliberate check and then drifting: no bounce-away, one self-distraction event")
    def _():
        d = Day()   # writing -> email (deliberate) -> Bluesky (self-distraction) -> writing
        a = d.add(12, "Essay")
        d.add(1, "Mail", switch="interruption", content="correspondence")
        d.add(1, "Bluesky", switch="interruption", focused=False, sd=True, content="passive")
        d.add(10, "Essay", switch="return", ante=a)
        flag(d)
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 22, sp
        assert len(sp[0]["detours"]) == 1, sp[0]["detours"]         # one leaving = one interruption
        assert stats.self_distraction_counts(d.segs, se) == (1, 0)

    @case("email was the self-distraction: one bounce-away, and Bluesky after it is a second event")
    def _():
        d = Day()
        a = d.add(12, "Essay")
        d.add(1, "Mail", switch="interruption", focused=False, sd=True, content="correspondence")
        d.add(1, "Bluesky", switch="interruption", focused=False, sd=True, content="passive")
        d.add(10, "Essay", switch="return", ante=a)
        flag(d)
        se = d.run()
        assert len(se["focus_spans"][0]["detours"]) == 1
        assert stats.self_distraction_counts(d.segs, se) == (2, 1)

    @case("creative work that turns into task mode on the same thread stays one creative span, no Toggl needed")
    def _():
        d = Day()
        d.add(15, "Agent"); d.add(25, "Agent", content="task")
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 40 and sp[0]["subtype"] == "creative_work", sp
        d = Day()   # only 6 creative minutes first: it never qualified as creative work
        d.add(6, "Agent"); d.add(25, "Agent", content="task")
        se = d.run()
        assert not se["focus_spans"] and se["disqualified"], se["focus_spans"]
        d = Day()   # task-mode work on OTHER threads is time away, and 5 minutes of it ends the span
        a = d.add(15, "Agent")
        d.add(6, "Mail", content="task", switch="interruption")
        d.add(4, "Agent", content="task", switch="return", ante=a)
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [15], [_mins(x) for x in sp]

    @case("going back to the same feed is a return that counts again, when it is judged a self-distraction")
    def _():
        d = Day()
        a = d.add(12, "Essay")
        b = d.add(1, "Feed", switch="interruption", focused=False, sd=True, content="passive")
        a2 = d.add(3, "Essay", switch="return", ante=a)
        d.add(1, "Feed", switch="return", ante=b, focused=False, sd=True, content="passive")
        a3 = d.add(3, "Essay", switch="return", ante=a2)
        d.add(1, "Feed", switch="return", ante=b, focused=False, content="passive")   # deliberate this time
        d.add(6, "Essay", switch="return", ante=a3)
        flag(d)
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and len(sp[0]["detours"]) == 3, sp
        assert stats.self_distraction_counts(d.segs, se) == (2, 2)

    @case("one slow task inside a brisk task block does not split the block")
    def _():
        d = Day()
        d.add(4, "Mail", content="task", toggl=7)
        d.add(3, "Forms", content="task", switch="interruption", nxt=True, toggl=7)
        d.add(25, "Taxes", content="task", switch="interruption", nxt=True, toggl=7)
        d.add(3, "Returns", content="task", switch="interruption", nxt=True, toggl=7)
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 35 and sp[0]["subtype"] == "focused_task", sp

    @case("a phone pickup judged a shift, inside a span, is one diversion and one interruption")
    def _():
        d = Day()
        d.add(30, "Essay")
        d.phone(14, 2)                     # 2 minutes on the phone, 14 minutes in
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 28 and len(sp[0]["detours"]) == 1, [(_mins(x), x["detours"]) for x in sp]
        assert sp[0]["detours"][0]["segments"][0]["app"] == "iPhone"
        assert stats.self_distraction_counts(d.segs, se) == (0, 0)   # a deliberate look: no self-distraction

    @case("six minutes on the phone ends the span where it was left")
    def _():
        d = Day()
        d.add(40, "Essay")
        d.phone(15, 6)
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [15, 19], [_mins(x) for x in sp]

    @case("a phone pickup judged a self-distraction is an event, and a bounce-away from the span")
    def _():
        d = Day()
        d.add(30, "Essay")
        d.phone(14, 1, sd=True, apps="Bumble 60s")
        se = d.run()
        assert len(se["focus_spans"]) == 1 and len(se["focus_spans"][0]["detours"]) == 1
        assert stats.self_distraction_counts(d.segs, se) == (1, 1)

    @case("a glance at the lock screen is an interruption and never a self-distraction")
    def _():
        import sqlite3
        from . import evaluator
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute("CREATE TABLE phone_evals (pickup_key INTEGER PRIMARY KEY, start REAL, end REAL, apps TEXT,"
                  " switch_label TEXT, content TEXT, focused INTEGER, self_distraction INTEGER, rationale TEXT,"
                  " created REAL, run_id INTEGER)")
        d = Day()
        d.add(30, "Essay")
        a = T0 + 14 * 60
        evaluator.store_lock_glance(c, {"key": int(a), "start": a, "end": a + 4, "use_s": 4.0})   # the rule, as stored
        for sg in d.segs:
            sg["redesign"] = True
        d.segs = stats._carve_phone(c, d.segs, T0, d.t)
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and len(sp[0]["detours"]) == 1 and sp[0]["detours"][0]["state"] == "interrupted", sp
        assert stats.self_distraction_counts(d.segs, se) == (0, 0)

    @case("a phone pickup judged a continuation changes nothing")
    def _():
        d = Day()
        d.add(30, "Essay")
        d.phone(14, 2, switch="continuation")
        se = d.run()
        sp = se["focus_spans"]
        assert len(d.segs) == 1 and len(sp) == 1 and _mins(sp[0]) == 30 and not sp[0]["detours"], sp

    @case("a pass-through touch-back does not reset the 5-minute clock; a real re-engagement does")
    def _():
        d = Day()
        a = d.add(12, "Essay")
        b = d.add(4, "Feed", switch="interruption", focused=False, sd=True, content="passive")
        a2 = d.add(0.2, "Essay", switch="return", ante=a)
        d.add(4, "Feed", switch="return", ante=b, focused=False, sd=True, content="passive")
        d.add(12, "Essay", switch="return", ante=a2)
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [12, 12], [_mins(x) for x in sp]   # one 8-minute absence
        d = Day()
        a = d.add(12, "Essay")
        b = d.add(4, "Feed", switch="interruption", focused=False, sd=True, content="passive")
        a2 = d.add(1, "Essay", switch="return", ante=a)                      # a minute of real work
        d.add(4, "Feed", switch="return", ante=b, focused=False, sd=True, content="passive")
        d.add(12, "Essay", switch="return", ante=a2)
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and len(sp[0]["detours"]) == 2, [_mins(x) for x in sp]

    return out


def main() -> int:
    failed = 0
    for name, fn in cases():
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {str(e)[:400]}")
    print(f"span self-test: {len(cases()) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

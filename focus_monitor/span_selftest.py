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
    def __init__(self, t0=T0):
        self.segs: list[dict] = []
        self.t = t0

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

    def pause(self, minutes):
        """A short pause (a screen lock under the idle threshold, or time judged "idle"), as
        stats.labelled_segments marks it: it carries the state of the segment before it."""
        sid = self.add(minutes, "__idle__", content="idle")
        before = self.segs[-2]["state"] if len(self.segs) > 1 else "focus"
        self.segs[-1].update(state=before if before != "idle" else "focus", pause=True, idle=1, focused=None,
                             label=None, switch=None, source=None, antecedent_id=None)
        return sid

    def run(self):
        return stats.spans_and_events(self.segs, min_focus_min=10, break_min=5)

    PHONE_DDL = ("CREATE TABLE phone_evals (seg_key INTEGER PRIMARY KEY, pickup_key INTEGER, start REAL, end REAL,"
                 " app TEXT, switch_label TEXT, content TEXT, focused INTEGER, self_distraction INTEGER,"
                 " next_task INTEGER, antecedent TEXT, away INTEGER, rationale TEXT, created REAL, run_id INTEGER)")

    def phone(self, at_min, parts):
        """An iPhone pickup `at_min` minutes into the day. `parts` is a list of judged app
        segments, each (minutes, app, switch, away, self_distraction, focused) in the order used;
        carves them out of the segments the way stats.labelled_segments does (through an
        in-memory phone_evals)."""
        import sqlite3
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute(self.PHONE_DDL)
        a = pk = T0 + at_min * 60.0
        for minutes, app, switch, away, sd, focused in parts:
            c.execute("INSERT INTO phone_evals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (int(a), int(pk), a, a + minutes * 60.0, app, switch, "other", int(focused), int(sd), 0, None,
                       int(away), "test", 0, None))
            a += minutes * 60.0
        for sg in self.segs:
            sg.setdefault("redesign", True)
        self.segs = stats._carve_phone(c, self.segs, T0, self.t)
        return self


def _mins(sp):
    return round(sp["focus_min"], 1)


MEM_DDL = """
CREATE TABLE segments (id INTEGER PRIMARY KEY, start REAL, end REAL, app TEXT, title TEXT, url TEXT, domain TEXT,
    first_screenshot TEXT, last_screenshot TEXT, idle INTEGER DEFAULT 0, neutral INTEGER DEFAULT 0);
CREATE TABLE switches (id INTEGER PRIMARY KEY, ts REAL, to_segment INTEGER, status TEXT, group_id INTEGER);
CREATE TABLE ensemble (switch_id INTEGER, label TEXT, uncertain INTEGER, p_continuation REAL, p_interruption REAL,
    p_distraction REAL, p_task_change REAL, activity TEXT);
CREATE TABLE reviews (switch_id INTEGER, label TEXT);
CREATE TABLE thread_calls (switch_id INTEGER, thread TEXT);
CREATE TABLE seg_evals (segment_id INTEGER PRIMARY KEY, content TEXT, switch_label TEXT, interruption_kind TEXT,
    rationale TEXT, created REAL, uncertain INTEGER, run_id INTEGER, probs TEXT, antecedent_id INTEGER, hybrid INTEGER,
    focused INTEGER, self_distraction INTEGER, next_task INTEGER);
CREATE TABLE seg_reviews (segment_id INTEGER PRIMARY KEY, content TEXT, switch_label TEXT, interruption_kind TEXT,
    note TEXT, created REAL, antecedent_id INTEGER, hybrid INTEGER, chat_message_id INTEGER, focus_exception INTEGER,
    focused INTEGER, self_distraction INTEGER, next_task INTEGER);
CREATE TABLE inactivity (id INTEGER PRIMARY KEY, start REAL, end REAL);
CREATE TABLE toggl_entries (id INTEGER PRIMARY KEY, start REAL, stop REAL, description TEXT, project TEXT, tags TEXT, synced REAL);
"""
DB0 = 1_700_000_000.0   # a start in the past, for cases that go through stats.labelled_segments


class MemDay:
    """A day in an in-memory database, for the rules that live in stats.labelled_segments (how a
    person's ruling, a Toggl entry or an idle segment is read). Times are minutes from DB0."""

    def __init__(self):
        import sqlite3
        self.c = sqlite3.connect(":memory:")
        self.c.row_factory = sqlite3.Row
        self.c.executescript(MEM_DDL)
        self.t = DB0
        self.n = 0

    def seg(self, minutes, app, *, idle=0, ev=None, rv=None):
        """One segment; `ev` / `rv` are the evaluator's and the person's rows (column: value)."""
        self.n += 1
        a, b = self.t, self.t + minutes * 60.0
        self.c.execute("INSERT INTO segments (id, start, end, app, title, url, domain, idle) VALUES (?,?,?,?,?,?,?,?)",
                       (self.n, a, b, "__idle__" if idle else app, "", "", "", idle))
        for table, row in (("seg_evals", ev), ("seg_reviews", rv)):
            if row is not None:
                row = dict(row, segment_id=self.n, created=a)
                self.c.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        self.t = b
        return self.n

    def toggl(self, a_min, b_min):
        self.c.execute("INSERT INTO toggl_entries (start, stop, description, tags) VALUES (?,?,?,?)",
                       (DB0 + a_min * 60.0, DB0 + b_min * 60.0, "tasks", ""))

    def segs(self):
        return stats.labelled_segments(self.c, DB0, self.t)

    def run(self):
        return stats.spans_and_events(self.segs(), min_focus_min=10, break_min=5)


FOCUSED = {"content": "creative", "switch_label": "continuation", "focused": 1}


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
        d.phone(14, [(2, "Uber Eats", "interruption", True, False, True)])   # a 2-minute deliberate errand, 14 minutes in
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 28 and len(sp[0]["detours"]) == 1, [(_mins(x), x["detours"]) for x in sp]
        assert sp[0]["detours"][0]["segments"][0]["app"] == "iPhone"
        assert stats.self_distraction_counts(d.segs, se) == (0, 0)   # a deliberate look: no self-distraction

    @case("six minutes on the phone ends the span where it was left")
    def _():
        d = Day()
        d.add(40, "Essay")
        d.phone(15, [(6, "Signal", "interruption", True, False, True)])
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [15, 19], [_mins(x) for x in sp]

    @case("a phone pickup judged a self-distraction is an event, and a bounce-away from the span")
    def _():
        d = Day()
        d.add(30, "Essay")
        d.phone(14, [(1, "Bumble", "interruption", True, True, False)])
        se = d.run()
        assert len(se["focus_spans"]) == 1 and len(se["focus_spans"][0]["detours"]) == 1
        assert stats.self_distraction_counts(d.segs, se) == (1, 1)

    @case("a glance at the lock screen is an interruption and never a self-distraction")
    def _():
        import sqlite3
        from . import evaluator
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute(Day.PHONE_DDL)
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

    @case("phone segments are judged one by one: a pass-through app, then a shift, then more of it")
    def _():
        d = Day()
        d.add(30, "Essay")
        # unlocked onto Oura for 3 s (pass-through: nothing), then Bumble (self-distraction), then on to Feeld
        d.phone(14, [(0.05, "Oura", "continuation", False, False, True),
                     (1, "Bumble", "interruption", True, True, False),
                     (1, "Feeld", "continuation", True, False, False)])
        se = d.run()
        sp = se["focus_spans"]
        ph = [g for g in d.segs if g.get("phone")]
        assert [g["domain"] for g in ph] == ["Bumble", "Feeld"], [g["domain"] for g in ph]     # Oura is not carved
        assert ph[1]["switch"] == "continuation" and ph[1]["antecedent_id"] == ph[0]["id"] and ph[1]["state"] == "distracted"
        assert len(sp) == 1 and len(sp[0]["detours"]) == 1 and _mins(sp[0]) == 28, [(_mins(x), len(x["detours"])) for x in sp]
        assert stats.self_distraction_counts(d.segs, se) == (1, 1)

    @case("focused phone time that was a shift is time away, and never a span of its own")
    def _():
        d2 = Day()
        d2.add(40, "Essay")
        d2.phone(12, [(20, "Exercise Tracker", "interruption", True, False, True)])
        sp = d2.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [12], [_mins(x) for x in sp]   # 20 focused minutes on the phone start no span

    @case("an on-track phone call keeps the span going while the Mac sits idle")
    def _():
        def day(on_track):
            d = Day()
            d.add(15, "Essay")
            d.add(14, "Essay")
            d.segs[-1].update(state="idle", idle=1, label=None, content="idle")
            d.add(15, "Essay")
            if on_track:      # a 12-minute call, 1 minute into the idle time
                d.phone(16, [(12, "Phone", "continuation", False, False, True)])
            return d
        assert [_mins(x) for x in day(False).run()["focus_spans"]] == [15, 15]
        d = day(True)
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [44], [_mins(x) for x in sp]
        assert [g["domain"] for g in d.segs if g.get("phone")] == ["Phone"]

    @case("a span that runs past the end of the day is qualified as a whole; each day keeps its own minutes")
    def _():
        d = Day()
        d.add(4, "Mail", focused=False, switch="interruption", content="correspondence")
        d.add(16, "Essay", switch="interruption")
        w = d.run()["focus_spans"]
        assert [_mins(x) for x in w] == [16], [_mins(x) for x in w]
        cut = T0 + 12 * 60.0    # "4am", 8 minutes into the essay
        def half(a, b):
            out = []
            for g in d.segs:
                ca, cb = max(g["clip_start"], a), min(g["clip_end"], b)
                if cb > ca:
                    out.append(dict(g, clip_start=ca, clip_end=cb, duration=cb - ca))
            return out
        whole = [{"start": w[0]["start"], "end": w[0]["end"], "subtype": w[0]["subtype"], "focus_min": w[0]["focus_min"]}]
        for a, b in ((T0, cut), (cut, d.t)):
            alone = stats.spans_and_events(half(a, b), min_focus_min=10, break_min=5)["focus_spans"]
            assert alone == [], [_mins(x) for x in alone]              # 8 minutes does not qualify alone
            sp = stats.spans_and_events(half(a, b), min_focus_min=10, break_min=5, whole=whole)["focus_spans"]
            assert [_mins(x) for x in sp] == [8], [_mins(x) for x in sp]

    @case("phone time away from the Mac is distracted when its pickup began as a self-distraction, and only then")
    def _():
        def day(sd):
            d = Day()
            d.add(15, "Essay")
            d.add(20, "Essay")
            d.segs[-1].update(state="idle", idle=1, label=None, content="idle")
            d.add(15, "Essay")
            # 2 minutes into the idle time: Bumble, a pass-through app that is not carved, then Feeld
            d.phone(17, [(2, "Bumble", "interruption", True, sd, False),
                         (1, "Oura", "continuation", False, False, False),
                         (3, "Feeld", "continuation", True, False, False)])
            se = d.run()
            ev = sorted(se["distractions"] + se["interruptions"], key=lambda r: r["start"])
            return [(r["segments"][0]["domain"], r["state"]) for r in ev]
        assert day(True) == [("Bumble", "distracted"), ("Feeld", "distracted")], day(True)
        assert day(False) == [("Bumble", "interrupted"), ("Feeld", "interrupted")], day(False)

    @case("leaving the task and then going away from the Mac: no span is joined across the idle time")
    def _():
        d = Day()
        d.add(15, "Essay")
        r = d.add(3, "Report", switch="interruption", content="task")
        d.add(20, "Report")
        d.segs[-1].update(state="idle", idle=1, label=None, content="idle")
        d.add(12, "Report", ante=r)
        se = d.run()
        walls = sorted(round((x["end"] - x["start"]) / 60) for x in se["focus_spans"] + se["short_focus"] + se["disqualified"])
        assert walls and max(walls) <= 15, walls   # before 2026-11-20: Report 3 + idle 20 + Report 12 was one 35-minute span

    @case("a phone pickup judged a continuation changes nothing")
    def _():
        d = Day()
        d.add(30, "Essay")
        d.phone(14, [(2, "Authenticator", "continuation", False, False, True)])
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

    # ---- 2026-11-20: the nine fixes from the audit of the statistics ----

    @case("a short pause (a lock under the idle threshold) does not end the span; ten idle minutes do")
    def _():
        def day(idle_min):
            m = MemDay()
            m.seg(12, "Essay", ev=dict(FOCUSED, switch_label="interruption"))
            m.seg(idle_min, "", idle=1)
            m.seg(12, "Essay", ev=dict(FOCUSED, antecedent_id=1))
            return m
        m = day(3)
        sg = m.segs()
        assert sg[1]["pause"] and sg[1]["state"] == "focus" and sg[1]["idle"] == 1, sg[1]
        se = m.run()
        assert [_mins(x) for x in se["focus_spans"]] == [27], [_mins(x) for x in se["focus_spans"]]
        assert not se["interruptions"] and not se["distractions"], se["interruptions"]
        m = day(11)
        assert m.segs()[1]["state"] == "idle" and not m.segs()[1].get("pause")
        assert [_mins(x) for x in m.run()["focus_spans"]] == [12, 12]

    @case("time judged \"idle\" is a pause, or time away when it adds up to the idle threshold; never focus")
    def _():
        m = MemDay()     # 14 minutes of nothing, in two segments: it was a 14-minute creative span
        m.seg(3, "Essay", ev=dict(FOCUSED, switch_label="interruption"))
        m.seg(8, "Essay", ev={"content": "idle", "switch_label": "continuation"})
        m.seg(6, "Essay", ev={"content": "idle", "switch_label": "return"})
        sg = m.segs()
        assert [g["state"] for g in sg] == ["focus", "idle", "idle"], [g["state"] for g in sg]
        se = m.run()
        assert not se["focus_spans"] and not se["disqualified"], se["focus_spans"]
        m = MemDay()     # the person's "I wasn't at the computer" during a diversion
        m.seg(12, "Essay", ev=dict(FOCUSED, switch_label="interruption"))
        m.seg(3, "Feed", ev={"content": "passive", "switch_label": "interruption", "focused": 0, "self_distraction": 0})
        m.seg(4, "Feed", ev={"content": "passive", "switch_label": "continuation", "focused": 0}, rv={"content": "idle"})
        sg = m.segs()
        assert sg[2]["pause"] and sg[2]["state"] == "interrupted", sg[2]
        se = m.run()
        assert [_mins(x) for x in se["focus_spans"]] == [12]
        assert round(sum(r["duration"] for r in se["interruptions"]) / 60) == 3   # the pause is not unfocused time

    @case("a pause never opens a span, never counts toward the opening or unbroken 10 minutes, never is a diversion")
    def _():
        d = Day()
        d.pause(6); d.add(8, "Essay", switch="interruption")
        se = d.run()
        assert not se["focus_spans"] and not se["disqualified"], se
        d = Day()        # 6 + 6 focused minutes around a pause: no unbroken 10
        d.add(6, "Essay"); d.pause(3); d.add(6, "Essay")
        se = d.run()
        assert not se["focus_spans"] and [x["disq_reason"] for x in se["disqualified"]] == ["no continuous block of 10 minutes"], se
        d = Day()        # choppy opening: the span starts at its first clean 10 minutes
        d.add(4, "Essay"); d.pause(2); d.add(15, "Essay"); d.pause(2)
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 15, [_mins(x) for x in sp]   # and it does not end on a pause
        d = Day()        # a pause during time away is not on the 5-minute clock and not a diversion
        a = d.add(12, "Essay")
        d.add(4, "Feed", switch="interruption", focused=False, sd=True, content="passive")
        d.pause(3)
        d.add(10, "Essay", switch="return", ante=a)
        se = d.run()
        sp = se["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 22 and len(sp[0]["detours"]) == 1, [_mins(x) for x in sp]
        assert round(sp[0]["detour_min"]) == 4 and round(sum(r["duration"] for r in se["distractions"]) / 60) == 4
        d = Day()        # unfocused on-thread time is asked about; a pause is not unfocused time
        d.add(15, "Essay"); d.pause(9); d.add(15, "Essay")
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 39 and not sp[0]["unfocused_ask"] and sp[0]["unfocused_min"] == 0, sp

    @case("a thread chain that reaches its root loses no confidence on the way in (a different thread stays one)")
    def _():
        d = Day()
        r = d.add(3, "Atlas", switch="interruption", content="planning")
        del d.segs[-1]["antecedent_p"]              # a thread's first segment has no link
        d.add(12, "Terminal", switch="interruption")
        d.add(6, "Notes", switch="return", ante=r, content="planning")
        d.segs[-1]["antecedent_p"] = 0.54           # 0.54 x the root's default 0.85 = 0.459
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [12], [_mins(x) for x in sp]

    @case("going back to a distraction on the phone counts again; a plain return within a pickup does not")
    def _():
        def day(back_sd):
            d = Day()
            d.add(30, "Essay")
            d.phone(14, [(1, "Bumble", "interruption", True, True, False),
                         (1, "Maps", "interruption", True, False, True),
                         (1, "Bumble", "return", True, back_sd, False)])
            return d
        d = day(True)
        ph = [g for g in d.segs if g.get("phone")]
        assert ph[2]["switch"] == "return" and ph[2]["self_distraction"] and ph[2]["state"] == "distracted", ph[2]
        assert stats.self_distraction_counts(d.segs, d.run())[0] == 2
        d = day(False)
        ph = [g for g in d.segs if g.get("phone")]
        assert ph[2]["switch"] == "continuation" and not ph[2]["self_distraction"], ph[2]
        assert stats.self_distraction_counts(d.segs, d.run())[0] == 1

    @case("an old unjudged tail is assembled and counted; only a recent one waits for the evaluator")
    def _():
        import time
        def day(t0):
            d = Day(t0)
            d.add(15, "Essay")
            d.add(12, "Essay"); d.segs[-1].update(source="model", focused=None)
            d.add(8, "Feed"); d.segs[-1].update(source="model", focused=None, state="distracted", label="distraction")
            return d
        se = day(time.time() - 400 * 86400).run()
        assert [_mins(x) for x in se["focus_spans"]] == [27], [_mins(x) for x in se["focus_spans"]]
        assert round(sum(r["duration"] for r in se["distractions"]) / 60) == 8
        se = day(time.time() - 3600).run()
        assert [(_mins(x), x["ended_by"]) for x in se["focus_spans"]] == [(15, "awaiting evaluation")], se["focus_spans"]
        assert round(sum(r["duration"] for r in se["distractions"]) / 60) == 8   # its unfocused minutes still count

    @case("unfocused time on a thread that is in no span counts as unfocused minutes")
    def _():
        d = Day()
        d.add(3, "Essay"); d.add(40, "Essay", focused=False)
        se = d.run()
        assert not se["focus_spans"]
        assert round(sum(r["duration"] for r in se["interruptions"]) / 60) == 40, se["interruptions"]
        assert stats.self_distraction_counts(d.segs, se) == (0, 0)
        d = Day()   # inside a span it is focus time, and is not counted a second time
        d.add(15, "Essay"); d.add(8, "Essay", focused=False); d.add(20, "Essay")
        se = d.run()
        assert len(se["focus_spans"]) == 1 and not se["interruptions"] and not se["distractions"], se["interruptions"]

    @case("reading with writing about it in the same window is a reading span; reading alone is not")
    def _():
        d = Day()
        d.add(20, "Google Chrome", domain="lesswrong.com", content="reading")
        d.add(15, "Google Chrome", domain="lesswrong.com", content="creative")   # a long comment on the post
        d.add(13, "Google Chrome", domain="lesswrong.com", content="reading")
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and sp[0]["subtype"] == "reading", sp
        d = Day()
        d.add(48, "Google Chrome", domain="lesswrong.com", content="reading")
        se = d.run()
        assert not se["focus_spans"] and se["disqualified"][0]["disq_reason"] == "reading without note-taking", se

    @case("a person's older ruling (a shift's kind, or \"counts as focus\") outranks a re-judgment's focused field")
    def _():
        def state(ev, rv):
            m = MemDay()
            m.seg(5, "Essay", ev=dict(FOCUSED, switch_label="interruption"))
            m.seg(5, "Mail", ev=ev, rv=rv)
            return m.segs()[1]
        shift = {"content": "task", "switch_label": "interruption", "self_distraction": 0}
        g = state(dict(shift, focused=0), {"switch_label": "interruption", "interruption_kind": "focus_start"})
        assert g["state"] == "focus" and g["focused"] is True, g["state"]
        g = state(dict(shift, focused=1), {"switch_label": "interruption", "interruption_kind": "detour"})
        assert g["state"] == "interrupted" and g["user_unfocused"] and not g["self_distraction"], g["state"]
        g = state(dict(shift, focused=1), {"switch_label": "interruption", "interruption_kind": "self_distraction"})
        assert g["state"] == "distracted" and g["self_distraction"], g["state"]
        g = state({"content": "correspondence", "switch_label": "interruption", "focused": 0, "self_distraction": 1},
                  {"content": "correspondence", "focus_exception": 1})
        assert g["state"] == "focus", g["state"]
        g = state(dict(shift, focused=1), {"content": "task"})     # a ruling on something else changes nothing
        assert g["state"] == "focus" and not g["user_unfocused"]
        g = state(dict(shift, focused=1), {"interruption_kind": "detour", "focused": 1})   # their own field wins
        assert g["state"] == "focus", g["state"]

    @case("Toggl coverage is seconds of overlap, not whether an entry ran at the segment's first instant")
    def _():
        e = [{"id": 1, "start": 0.0, "stop": 100.0}, {"id": 2, "start": 90.0, "stop": 400.0}, {"id": 3, "start": 90.0, "stop": 400.0}]
        best, ivs = stats._toggl_cover(e, 50.0, 300.0, 999.0)
        assert best["id"] == 3 and ivs == [(50.0, 300.0)], (best, ivs)      # most overlap, the later-started; counted once
        assert stats._toggl_cover(e, 500.0, 600.0, 999.0) == (None, [])
        task = {"content": "task", "switch_label": "continuation", "focused": 1}
        def day(a_min, b_min):
            m = MemDay()
            m.seg(8, "Mail", ev=dict(task, switch_label="interruption"))
            m.seg(8, "Mail", ev=dict(task, antecedent_id=1))
            m.toggl(a_min, b_min)
            return m
        m = day(1, 16)         # the timer was started a minute in: 15 of 16 minutes covered
        assert m.segs()[0]["toggl_id"] == 1
        sp = m.run()["focus_spans"]
        assert len(sp) == 1 and sp[0]["subtype"] == "focused_task", sp
        se = day(-5, 2).run()  # running at the start of the first segment, stopped 2 minutes in
        assert not se["focus_spans"] and se["disqualified"][0]["disq_reason"] == "task activity without a Toggl entry", se

    @case("a sub-thread under 10 minutes is absorbed into the thread it grew out of; a longer one stands alone")
    def _():
        # 8 minutes of debugging that grew out of the task: one span of 12 + 8 + 10, no diversion
        d = Day()
        a = d.add(12, "Task")
        b = d.add(5, "Debugger", switch="interruption", content="task")
        d.segs[-1]["sub_of"] = a
        d.add(3, "Logs")                       # continues the debugging
        d.add(10, "Task", switch="return", ante=a)
        sp = d.run()["focus_spans"]
        assert len(sp) == 1 and _mins(sp[0]) == 30 and not sp[0]["detours"], [(_mins(x), x["detours"]) for x in sp]
        assert d.segs[b - 1].get("sub_absorbed") and len(sp[0]["sub_threads"]) == 1 and sp[0]["sub_threads"][0]["min"] == 8.0
        # without the mark the same 8 minutes are a different thread: the span ends where they began
        d = Day()
        a = d.add(12, "Task"); d.add(5, "Debugger", switch="interruption", content="task"); d.add(3, "Logs")
        d.add(10, "Task", switch="return", ante=a)
        assert [_mins(x) for x in d.run()["focus_spans"]] == [12, 10]
        # 12 minutes of it: a thread of its own, and the parent span ends where it budded
        d = Day()
        a = d.add(12, "Task")
        b = d.add(7, "Debugger", switch="interruption")
        d.segs[-1]["sub_of"] = a
        d.add(5, "Logs")
        d.add(10, "Task", switch="return", ante=a)
        sp = d.run()["focus_spans"]
        assert [_mins(x) for x in sp] == [12, 12, 10], [_mins(x) for x in sp]
        assert d.segs[b - 1].get("sub_thread") and not d.segs[b - 1].get("sub_absorbed")
        assert d.run()["focus_spans"][0]["end"] == sp[0]["end"]     # running it twice changes nothing

    @case("a span's verdict is the newest one that fits it, not a stale wider one with more overlap")
    def _():
        sp = {"start": 1000.0, "end": 2000.0}
        wide = {"start": 0.0, "end": 3000.0, "verdict": "not_focus", "created": 1.0}
        own = {"start": 1000.0, "end": 2000.0, "verdict": "focus", "created": 2.0}
        newer = {"start": 1010.0, "end": 1990.0, "verdict": "not_focus", "created": 3.0}
        assert stats._span_review_for([wide, own], sp) is own
        assert stats._span_review_for([own, wide], sp) is own
        assert stats._span_review_for([newer, own], sp) is newer
        assert stats._span_review_for([wide], sp) is None

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

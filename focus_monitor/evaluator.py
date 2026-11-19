"""Chunked segment evaluator: the model applies judgment over whole stretches of activity.

Every ~15 minutes (lazily, so after-context exists), one Claude call evaluates a chunk of
segments together, with surrounding context and sampled screenshots. For each segment it
judges:
  1. CONTENT - what kind of activity the segment is:
     creative | task | correspondence | reading | planning | other | passive
     ("task" describes the activity's nature regardless of Toggl - Toggl is span-level
     evidence of deliberate focus, not evidence of task-nature.)
  2. SWITCH - what the switch INTO the segment was, attention-wise:
     continuation (no shift of the object of attention) | shift (attention moved to
     something new; stored as 'interruption', see seg_models.shown_switch) |
     return (coming back to the object attention was on before a shift)
  3. FOCUSED - was attention focused during the segment, or unfocused (quality of attention,
     independent of which thread the segment belongs to).
  4. For shifts, two yes/no judgments: SELF_DISTRACTION (sought out stimulation without a
     clear intent) and NEXT_TASK (the next task under the same task-processing intention;
     the app joins those threads into one focused task block).
The older per-shift KIND (self_distraction | focus_start | detour) is no longer asked of the
model (Rules doc redesign, 2026-11-19); it is derived from these fields and still stored, so
history, the local models and the person's past corrections keep working.

The user audits and edits these (seg_reviews); edits are authoritative for state derivation
and are fed back to the evaluator as few-shot corrections, so it learns over time."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import (PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, Config,
                     load_config)

log = logging.getLogger("evaluator")

from .seg_models import (CONTENTS, HEADS, KINDS, SHOWN_SWITCHES, SWITCHES,  # single source for the label space
                         shown_switch, stored_switch)

LAZY_S = 15 * 60          # evaluate only segments at least this old (after-context exists)
CHUNK_S = 20 * 60         # evaluate about this much activity per call
CTX_S = 10 * 60           # context shown on each side of the chunk
RETURN_MAX_S = 10 * 60    # Rules doc: a return is to something left within the past 10 minutes
HOLD_MAX_S = 60 * 60      # Rules doc: the model may hold a segment back for up to an hour
HOLD_RETRY_S = 15 * 60    # after a hold, ask again this much later (more activity will be logged)
MAX_SHOTS = 8


class AnteCand(BaseModel):
    segment_id: int
    p: float = Field(description="Probability this is the right attachment GIVEN the segment continues an existing thread (the candidates' probabilities sum to ~1).")


class SegJudgement(BaseModel):
    segment_id: int
    content: str = Field(description="One of: creative, task, correspondence, reading, planning, other, passive, meeting, idle, transition.")
    focused: bool = Field(description="Quality of attention during THIS segment: true = focused (doing one thing with intent, however small), false = unfocused (half-hearted, scattered, drifting, or consuming). Independent of which thread the segment belongs to and of whether it is a diversion from something else.")
    switch: str = Field(description="The switch INTO this segment: continuation, shift, or return.")
    antecedents: list[AnteCand] = Field(default_factory=list, description="When switch is continuation or return: up to 3 RANKED candidate antecedents - each the id of the nearest earlier segment of a thread this might continue. Usually one candidate carrying ~all the mass; add a runner-up (and rarely a third) only when genuinely torn between attachments. Empty when switch is shift (a new thread).")
    self_distraction: bool = Field(default=False, description="When switch is 'shift', or a 'return' to something that is not their work: true when the person sought out stimulation or distraction WITHOUT a clear intent. False otherwise, and always false for a continuation.")
    next_task: bool = Field(default=False, description="Only when switch is 'shift': true when this is the person moving on to the NEXT TASK under the same overall intention of processing tasks. False for every other shift, and always false for continuation/return.")
    uncertain: bool = Field(default=False, description="True only when you are genuinely unsure of the content, focus or switch judgment and a human should check it. Use sparingly - these are flagged for the person to review.")
    hybrid: bool = Field(default=False, description="True when the activity (reading, task, creative, planning...) was being done INSIDE a live discussion with someone - e.g. reading a bill together on a call while taking notes. Keep content = the activity; hybrid marks the live-discussion dimension. False for solo work and for plain meetings (content='meeting').")
    note: str = Field(description="One short sentence of reasoning.")


class PhoneJudgement(BaseModel):
    pickup_id: int = Field(description="The number after 'id=P' on the EVALUATE PHONE line.")
    switch: str = Field(description="continuation (the phone use was part of what they were doing) or shift (attention left it).")
    content: str = Field(description="What the phone use was: correspondence, passive, task, planning, reading, meeting, or other.")
    focused: bool = Field(description="As for a segment: true = doing one thing with intent, false = drifting or consuming.")
    self_distraction: bool = Field(default=False, description="Only on a shift: they picked up the phone seeking stimulation or distraction WITHOUT a clear intent.")
    note: str = Field(description="One short sentence of reasoning.")


class ChunkJudgement(BaseModel):
    segments: list[SegJudgement]
    phone: list[PhoneJudgement] = Field(default_factory=list, description="One entry per line marked EVALUATE PHONE; empty when there are none.")
    hold_from: int | None = Field(default=None, description="Only when the prompt says holding is allowed. The id of the first EVALUATE segment you would rather judge LATER, because what the person does next would change your judgment of it. Judge every segment before it; leave it and every segment after it out of `segments`. null = judge everything now (the normal case).")


def kind_of(switch: str | None, focused, self_distraction) -> str | None:
    """The older per-shift kind, derived from the fields the model now gives (see the module
    docstring). Stored alongside them so everything written for the old label keeps working."""
    if switch != "interruption":
        return None
    if self_distraction:
        return "self_distraction"
    return "focus_start" if focused else "detour"


SYSTEM = """You are the segment evaluator for a personal attention monitor. You receive a stretch \
of one person's computer activity - an ordered log of segments (one active window each), with \
surrounding context before and after, Toggl entries, and sampled screenshots - and you judge every \
segment marked EVALUATE. For each one output:

1. content - what the activity IS:
   - creative: making something (writing an essay/doc, coding, video editing).
   - task: executing discrete tasks (forms, errands, admin, logistics, purchases). Judge the \
activity's nature only - whether a Toggl timer is running is NOT part of this judgment.
   - correspondence: communicating with people in writing - chatting on Signal/Messages/\
WhatsApp/Slack, reading and answering email, replying to comments or DMs. The exchange itself \
is the activity. (A live voice/video call is 'meeting'; a message that is really a piece of \
writing the person is absorbed in composing can be 'creative'.) Opening a chat only to fetch a \
link, file, or reference on the way to a task is NOT correspondence - no exchange is happening - \
it is 'transition' (or 'task'), whatever the app. Correspondence is interactive, \
so on its own it is never part of a focus span - but label the segment's nature regardless: \
email processed between other tasks under a running Toggl entry is still 'correspondence' \
content, and the span logic decides that it belongs to the task block.
   - reading: reading posts/papers/books. (Whether notes are being taken matters for spans, not \
for this label - reading without notes is still 'reading' content, unless it is really passive \
browsing/consumption.)
   - planning: metacognition - journaling, prioritizing, reviewing goals, deciding what to work on.
   - other: real engaged activity fitting none of the above.
   - passive: passive consumption - watching videos, scrolling feeds, browsing without an \
objective, reading comics/entertainment. When in doubt between 'other' and 'passive', lean passive: \
this person's calibration is that the system has been far too optimistic about focus. That lean \
applies to AMBIGUOUS ENGAGEMENT only - a segment long enough that consumption could have happened. \
It does not apply to segments too short for any engagement (see 'transition' below).
     EXCEPTION - a video (or talk, or screencast) watched IN SERVICE OF THE CURRENT TASK is not \
passive: a tutorial on exactly the thing they are stuck on, a recording they need for the piece \
they are writing. Both must be evident from the log: the video is clearly about the task they \
were just on, and they go back to that task afterwards (or it is still playing at the end of \
the log). Give it the content label of the task it serves, switch 'continuation' of that task, \
focused true. A video that is merely on a related topic, or that they drift on from to other \
videos, is passive as usual.
   - meeting: a live conversation - video call (Zoom/Meet/FaceTime), call notes or live transcript \
(e.g. Otter) during one. When the conversation is the ACTIVITY. If instead an activity (reading a \
document, working through a task, writing, planning) is being carried on INSIDE a live discussion - \
reading a bill with a colleague on a call while taking notes, pairing on code, going through a plan \
together - keep content = that activity and set hybrid=true. Evidence: a call/meeting window or a \
Toggl entry naming another person ("with Dave"), notes growing through input-idle stretches (listening), \
a transcript side panel. Hybrid stretches count as meeting time, not independent focus.
   - idle: the window was open but keyboard/mouse input was idle for most of the segment and \
nothing suggests engaged reading or watching - the person was likely away or disengaged. \
Log lines show "(input idle ...)" when recorded. A mostly-input-idle segment can NOT be task, \
creative, or planning (those require interaction); it can still be reading or passive if they \
were plausibly reading/watching the screen.
   - transition: rapidly passing through desktops, windows, or tabs on the WAY to a specific \
target - travel, not attention. Typically a chain of very short segments ending somewhere the \
person settles. Transitions are never attention shifts in themselves; their switch is \
'continuation' of whatever the person is heading toward. Also transition: a window that was \
already sitting open on a desktop and got cycled past - a segment of roughly 15 seconds or less \
with no screenshot and no scroll/typing evidence, flanked by segments of the same thread, is a \
pass-through regardless of what the window contains (a Substack post, a feed, a comic). Ten \
seconds is not consumption; do not label it passive or a shift on the strength of the \
window's name alone.
2. focused - the QUALITY of the person's attention during this segment: focused or unfocused.
   - focused (true): doing one thing with intent, engaged with it. This holds for small things \
and for things off the main work: jumping to a shop to buy one specific item is focused, even \
though it is a diversion from the essay they were writing.
   - unfocused (false): half-hearted, scattered or drifting - bouncing between surfaces with no \
one object, poking at something without engaging, lackadaisical task churn, browsing without an \
objective. Passive consumption is always unfocused.
   This is a SEPARATE question from which thread the segment belongs to (the switch, below) and \
from whether it is a diversion from something else (the app works that out). Judge the segment \
honestly on its own evidence, with the surrounding segments as context: if the segments around \
it are all the same thread of engaged work, this one is more likely focused. Do not mark time \
unfocused only because progress looks slow - the person may be thinking; mark it unfocused when \
the evidence shows attention was not on one thing. For a 'transition' or 'idle' segment give \
the value of the thread it sits inside (the app does not use it).
3. switch - did the switch INTO this segment shift the person's OBJECT OF ATTENTION?
   - continuation: same object (window-hopping within one piece of work counts - text<->notes, \
editor<->terminal, stepping through desktops to find something; also staying within one ongoing \
distraction session).
   - shift: the object of attention changed to something new. This is a judgment about this \
one switch only, and applies at any time of day, whether or not a focus span is running. (The \
person's rules use "attention-interruption event" for something narrower: a shift that happens \
during a focus span. The app works out which shifts are those; you only say whether attention \
moved.) Finishing one discrete task and starting the next (email -> taxes -> errands) IS a \
shift - attention moved to a new object - even inside a block of task processing; see next_task.
When they reach the new object by passing through other windows, those pass-through segments \
are transitions (switch 'continuation' of what they were on); 'shift' goes on the segment where \
they ARRIVE at the new object.
A BRIEF look at a todo list or task feed to choose the next task, followed by doing that \
task, is part of the chosen task's thread: the shift (with next_task, when it is one) goes on \
the todo-list segment, and the task that follows is a continuation of it. The look itself is \
neither a diversion nor a separate thread, and takes the task's content label.
   - return: attention moved back to an object they were on before an earlier shift, within \
about the last 10 minutes. Coming back to something after longer than that is a shift. When \
they get back by passing through other windows, those pass-through segments are transitions \
(switch 'continuation'); 'return' goes on the segment where they ARRIVE back on the old object.
   For continuation and return, ALSO name the antecedent(s): ranked candidates for the nearest \
EARLIER segment (visible in the log) belonging to the SAME thread - the thing this segment \
continues. For a continuation that is normally the immediately preceding non-transition segment; \
for a return, the last segment of what they were on BEFORE the shift away - the thing they \
are coming back to - and never a segment of the time away itself. Thread identity follows the \
chain of antecedents, so the nearest member is enough - never reach further back than needed. \
Probabilities are CONDITIONAL on the segment continuing an existing thread (whether it does at \
all is your switch judgment) and sum to ~1 across candidates. Usually one candidate carries \
essentially all the mass; when you are genuinely torn between two attachments, say so with a \
runner-up (and rarely a third). This pointer is the primary statement of thread structure; \
make it consistent with your switch judgment.
4. Two yes/no judgments, for switch='shift' (false otherwise; the one exception is noted):
   - self_distraction: the person sought out stimulation or distraction WITHOUT A CLEAR INTENT \
- absent-mindedly opening a feed, a blocked site, a dating or shopping app with no objective. \
It is about WHY they moved, not what is on the screen. NOT a self-distraction: a planned break \
(they stopped the timer and put on a video they meant to watch), watching something someone \
sent them, a deliberate errand or purchase, deliberately handling an email - even when the \
content is passive. EXCEPTION - going BACK to a distraction: when they leave what they \
were doing and go again to a feed or other distraction they visited a few minutes ago, the \
switch is a 'return' (to that earlier visit), and self_distraction is judged afresh for this \
visit by the same test: was THIS move stimulation-seeking without a clear intent? Each such \
visit counts separately. A return to their work is never a self-distraction.
   - next_task: this shift is the person moving on to the NEXT TASK under the same overall \
intention of processing tasks - working from a todo list or tracker and returning to it between \
tasks, executing efficiently, one task after another. The app joins tasks linked this way into \
one focused task block, and it applies the rule that a block needs a running Toggl entry; you \
judge the intention. False when the shift is not the next task: a glance at a feed, a stray \
question, drifting to something else. How long a task takes does not matter: one task that turns \
out fiddly and takes 25 slower minutes is still the next task, and so is the one after it.

5. PHONE PICKUPS. The log may contain PHONE lines: the person picked up their iPhone, with \
the apps that were on its screen and for how long. That is all that is known - which app, not \
what was in it, and there are no phone screenshots. For each line marked EVALUATE PHONE add an \
entry to `phone`:
   - switch: 'continuation' when the phone use was part of what they were doing at that moment \
- a confirmation code for the thing on the computer screen, looking something up for the task, \
a call or message exchange that IS the current activity, a timer or tracker for the current \
work. Starting or stopping their time tracker (Toggl) or jotting a note is never a diversion, \
like flipping to their daily page. Otherwise 'shift': attention left what they were doing. \
Their rule is that looking at the phone is a diversion unless what they look at is clearly \
related to the current thread - so when the app is not evidently part of the current activity \
it is a shift, however short. When the computer shows nothing going on around the pickup (idle, \
away), there is nothing to continue: it is a shift.
   - content and focused: as for a segment, about the phone use itself (messaging is \
correspondence, a feed or dating app is passive, an errand is task).
   - self_distraction: on a shift, the test of section 4 - did they pick it up seeking \
stimulation without a clear intent (absent-mindedly opening a dating app, a feed, a shop)? \
Not a self-distraction: answering a message that just arrived ("opened from a notification" \
points that way), a deliberate errand, logging something in a tracker.
   A pickup never changes how you label the computer segments themselves: the app cuts the \
phone time out of the segment it falls in, so do not turn a segment into a shift or a return \
because a PHONE line sits inside it. A PHONE line that already has a judgment shows it as \
<phone: continuation> or <phone: shift>.

Log lines for segments that already have a judgment show it as tags: <focused> / <unfocused>, \
and <switch: shift> or <switch: return> (no switch tag = continuation). Segments judged before \
2026-11-19 carry an older tag instead - <switch: shift (detour)>, (self_distraction) or \
(focus_start) - from when the kind of shift was a single label; read those as a shift.

Judge each segment in the context of the whole stretch - what came before AND after it. A flip to \
a notes app mid-video does not make the video session work; a quick message check inside a writing \
session is a shift away and a return, not the end of the work.

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}

{examples}"""


_ev = None


def _get():
    global _ev
    if _ev is None:
        _ev = ChunkEvaluator(load_config())
    return _ev


class ChunkEvaluator:
    def __init__(self, cfg: Config):
        import anthropic
        from .config import load_api_key
        load_api_key()
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials")
        self._examples_stamp = None
        self._system = None

    def _examples(self, conn) -> str:
        rows = conn.execute("""
            SELECT r.segment_id, r.content, r.switch_label, r.focused, r.self_distraction, r.next_task,
                   r.note, r.hybrid, g.app, g.domain, g.title
            FROM seg_reviews r JOIN segments g ON g.id = r.segment_id
            ORDER BY r.created DESC LIMIT 20""").fetchall()
        if not rows:
            return ""
        lines = ["The person has corrected past judgments - follow these precedents:"]
        for r in rows:
            what = f"{r['app']} {r['domain'] or ''} | {(r['title'] or '')[:60]}"
            verdict = " / ".join(v for v in (
                r["content"], shown_switch(r["switch_label"]),
                None if r["focused"] is None else ("focused" if r["focused"] else "unfocused"),
                "self-distraction" if r["self_distraction"] else None,
                "next task in the block" if r["next_task"] else None) if v)
            if r["hybrid"]:
                verdict += " + hybrid (inside a live discussion)"
            elif r["hybrid"] == 0:
                verdict += " (solo, not hybrid)"
            note = f" ({r['note']})" if r["note"] else ""
            lines.append(f"- {what} -> {verdict}{note}")
        return "\n".join(lines)

    def _system_blocks(self, conn) -> list[dict]:
        from .classifiers.claude_vision import _rules_doc, _rules_mtime
        stamp = (conn.execute("SELECT COALESCE(MAX(created),0) FROM seg_reviews").fetchone()[0], _rules_mtime())
        if self._system is None or stamp != self._examples_stamp:
            self._system = SYSTEM.format(rules=_rules_doc(),
                                         about_me=self.cfg.about_me.strip() or "(no description given)",
                                         examples=self._examples(conn))
            self._examples_stamp = stamp
        return [{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}]

    def evaluate_chunk(self, conn, t0: float, t1: float, extra: str | None = None,
                       phone_only: bool = False) -> int:
        """Evaluate all real segments starting in [t0, t1). Local models predict first (and are
        stored for track-record scoring); segments where the locals have EARNED confident
        deferral skip the LLM; the rest go into one LLM call, and the resolved judgment is the
        accuracy-weighted ensemble of all models. Returns how many segments were judged."""
        from . import seg_models, stats, summaries, toggl
        from .classifiers.claude_vision import _b64
        segs = stats.labelled_segments(conn, t0 - CTX_S, t1 + CTX_S, phone=False)  # pickups are shown as their own lines
        target = [s for s in segs if t0 <= s["clip_start"] < t1 and not s["idle"]
                  and not s.get("neutral") and s["duration"] > 0]
        if phone_only:
            # Phone data that arrived late: judge the pickups against the activity already
            # judged, and leave the computer segments' judgments alone. Re-judging them too was
            # tried (2026-11-19): it cost 6x as much per call and rewrote 23 settled judgments
            # in one stretch for no reason connected to the phone.
            target = []
        # iPhone pickups in view (phone.pickups). Those starting in [t0, t1) that have no
        # judgment yet are judged in this call; on a re-judgment of the segments all of them are.
        from . import phone as _phone
        picks_all = _phone.pickups(self.cfg, t0 - CTX_S, t1 + CTX_S)
        pk_have = {r["pickup_key"]: r for r in conn.execute(
            "SELECT * FROM phone_evals WHERE start >= ? AND start < ?", (t0 - CTX_S - 1, t1 + CTX_S + 1))}
        picks = [p for p in picks_all if t0 <= p["start"] < t1
                 and ((extra is not None and not phone_only) or p["key"] not in pk_have
                      or pk_have[p["key"]]["switch_label"] is None)]
        if not target and not picks:
            return 0
        # -- local models + deferral decision ---------------------------------------------
        sc = seg_models.scores(conn)
        real = [s for s in segs if not s["idle"] and not s.get("neutral") and s["duration"] > 0]
        local: dict = {}   # id -> (per-model preds, combined dist per head)
        deferred: set = set()
        now = time.time()
        for s in target:
            i = real.index(s)
            prev = real[i - 1] if i > 0 else None
            lp = seg_models.local_predict(conn, self.cfg, s, prev)
            for m, p in lp.items():
                seg_models.store_prediction(conn, s["id"], m, p)
            combined = {h: seg_models.combine_head({m: lp[m][h] for m in lp},
                                                   seg_models.weights_for(sc, h, list(lp)))
                        for h in HEADS}
            local[s["id"]] = (lp, combined)
            if extra is None and seg_models.can_defer(sc, combined, self.cfg):
                deferred.add(s["id"])
        for s in target:
            if s["id"] not in deferred:
                continue
            _, comb = local[s["id"]]
            c = max(comb["content"], key=comb["content"].get)
            sw = max(comb["switch"], key=comb["switch"].get)
            k = max(comb["kind"], key=comb["kind"].get) if sw == "interruption" else None
            conn.execute(
                "INSERT OR REPLACE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, probs, created)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (s["id"], c, sw, k, 0, "(deferred to local models - earned track record)",
                 json.dumps(comb), now))
        target = [s for s in target if s["id"] not in deferred]
        if not target and not picks:
            log.info("chunk %s-%s: all %d segments deferred to local models ($0)",
                     dt.datetime.fromtimestamp(t0).strftime("%H:%M"),
                     dt.datetime.fromtimestamp(t1).strftime("%H:%M"), len(deferred))
            conn.execute("INSERT INTO eval_runs(t0, t1, n_segments, cost_usd, created) VALUES (?,?,?,?,?)",
                         (t0, t1, len(deferred), 0.0, now))
            return len(deferred)
        target_ids = {s["id"] for s in target}
        all_ids = [s["id"] for s in segs if s["duration"] > 0]
        rats = {}
        if all_ids:
            qsr = ",".join("?" * len(all_ids))
            rats = {r2["segment_id"]: r2["rationale"] for r2 in conn.execute(
                f"SELECT segment_id, rationale FROM seg_evals WHERE segment_id IN ({qsr})", all_ids)
                if r2["rationale"]}
        lines, line_ts = [], []
        for s in segs:
            if s["duration"] <= 0:
                continue
            mark = f"EVALUATE id={s['id']}: " if s["id"] in target_ids else "(context) "
            lines.append(mark + summaries.seg_line(s)
                         + (f"  // prior judgment: {rats[s['id']][:80]}" if s["id"] in rats else ""))
            line_ts.append(s["clip_start"])
        pick_keys = {p["key"] for p in picks}
        for p in picks_all:   # PHONE lines, slotted into the log by time
            hm = lambda t: dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")
            apps = ", ".join(f"{a} {_dur(d)}" + (f" (opened from {'a notification' if how == 'notification' else how if how == 'another app' else 'the ' + how})" if how else "")
                             for a, d, how in p["apps"])
            had = pk_have.get(p["key"])
            tag = ""
            if p["key"] not in pick_keys and had is not None and had["switch_label"]:
                tag = ("  <phone: shift" + (", self-distraction" if had["self_distraction"] else "") + ">"
                       if had["switch_label"] == "interruption" else "  <phone: continuation>")
            line = ((f"EVALUATE PHONE id=P{p['key']}: " if p["key"] in pick_keys else "(context) PHONE ")
                    + f"{hm(p['start'])}-{hm(p['end'])}  iPhone in use {_dur(p['use_s'])}: {apps}"
                    + ("  (then left open, untouched)" if p["put_down"] else "") + tag)
            i = sum(1 for t in line_ts if t <= p["start"])
            lines.insert(i, line)
            line_ts.insert(i, p["start"])
        tg = toggl.entries_between(conn, t0 - CTX_S, t1 + CTX_S)
        tgtxt = "\n".join(f"  {dt.datetime.fromtimestamp(e['start']).strftime('%H:%M')}"
                          f"-{dt.datetime.fromtimestamp(e['stop'] or time.time()).strftime('%H:%M')}: "
                          f"{e['description'] or '(unnamed)'}" for e in tg) or "  (none)"
        content: list[dict] = []
        from . import shots as _shots
        shot_ids = list(target_ids) or [s["id"] for s in segs if s["duration"] > 0 and not s["idle"] and any(
            s["clip_start"] < p["end"] and s["clip_end"] > p["start"] for p in picks)]  # the Mac window behind a pickup
        shot_rows = _shots.pick(conn, shot_ids, MAX_SHOTS) if shot_ids else []  # the active display at each moment
        if not shot_rows:  # segments recorded before per-shot history existed
            shot_rows = [{"segment_id": s["id"], "ts": s["clip_start"], "path": s["first_screenshot"], "display": 0}
                         for s in target if s.get("first_screenshot")]
        app_of = {s["id"]: s["app"] for s in segs}
        shot_meta: list[dict] = []
        for r2 in shot_rows[:MAX_SHOTS]:
            b = _b64(r2["path"])
            if b:
                where = f" - EXTERNAL display {r2['display']}" if r2.get("display") else ""
                label = f"Screenshot of segment id={r2['segment_id']} ({dt.datetime.fromtimestamp(r2['ts']).strftime('%H:%M')} {app_of.get(r2['segment_id'], '')}{where}):"
                shot_meta.append({"path": r2["path"], "label": label})
                content.append({"type": "text", "text": label})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        # Hold-off (Rules doc): only for first-time judgments of activity under an hour old.
        can_hold = extra is None and bool(target) and time.time() - min(s["clip_end"] for s in target) < HOLD_MAX_S
        prompt_text = (f"Toggl entries in this window:\n{tgtxt}\n\nActivity log:\n" + "\n".join(lines)
                       + (f"\n\nIMPORTANT - {extra}" if extra else "")
                       + ("\n\nJudge every segment marked EVALUATE (all of them, by id)." if target else
                          "\n\nNo segment is marked EVALUATE: leave `segments` empty.")
                       + (" Judge every line marked EVALUATE PHONE, in `phone`." if picks else "")
                       + ("\n\nHolding is allowed for this chunk: if judging the LAST segments well depends on "
                          "what the person does after the activity shown here (for example a new piece of "
                          "work has only just begun and you cannot yet tell what it is), set hold_from to the "
                          "id of the first segment you would rather judge later, and leave it and everything "
                          "after it out. You will be shown them again in about 15 minutes with more activity "
                          "logged. Use this sparingly; most chunks need no hold." if can_hold else ""))
        content.append({"type": "text", "text": prompt_text})
        system_text = self._system_blocks(conn)[0]["text"]
        resp = llm.parse(self.client, site="evaluator", 
            model=self.cfg.claude_model, max_tokens=8000,
            system=self._system_blocks(conn),
            messages=[{"role": "user", "content": content}],
            output_format=ChunkJudgement,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default")
        cost = ((resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
                + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
                + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)
        n = n_held = n_phone = 0
        if resp.parsed_output is not None:
            now = time.time()
            per_seg_cost = cost / max(len(target), 1)
            seg_starts = {s["id"]: s["clip_start"] for s in segs if s["duration"] > 0}
            seg_ends = {s["id"]: s["clip_end"] for s in segs if s["duration"] > 0}
            # Hold-off (Rules doc): the model may ask to judge the tail of the chunk later.
            hold_id = resp.parsed_output.hold_from if can_hold else None
            held_from = seg_starts.get(hold_id) if hold_id in target_ids else None
            with db.tx(conn):
                run_id = conn.execute(
                    "INSERT INTO eval_runs(t0, t1, n_segments, cost_usd, created, system, prompt, shots, response, model)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (t0, t1, 0, cost, now, system_text, prompt_text, json.dumps(shot_meta),
                     resp.parsed_output.model_dump_json(), self.cfg.claude_model)).lastrowid
                for j in resp.parsed_output.segments:
                    if j.segment_id not in target_ids:
                        continue
                    c = j.content.strip().lower()
                    sw = stored_switch(j.switch.strip().lower())  # the model says 'shift'
                    if c not in CONTENTS or sw not in SWITCHES:
                        log.warning("bad judgement for segment %d: %s/%s", j.segment_id, c, sw)
                        continue
                    if held_from is not None and seg_starts.get(j.segment_id, 0) >= held_from:
                        continue  # the model asked to judge this one later
                    foc = bool(j.focused)
                    sd = bool(j.self_distraction) and sw in ("interruption", "return")
                    nxt = bool(j.next_task) and sw == "interruption"
                    k = kind_of(sw, foc, sd)   # the older per-shift kind, derived
                    # Claude's discrete judgment as (pseudo-)probabilities, so it can be scored
                    # and weighted alongside the local models.
                    top_p = 0.55 if j.uncertain else 0.85
                    cp = {"content": seg_models._uniformish(CONTENTS, c, top_p),
                          "switch": seg_models._uniformish(SWITCHES, sw, top_p),
                          "kind": seg_models._uniformish(KINDS, k, top_p) if k else seg_models._uniformish(KINDS)}
                    seg_models.store_prediction(conn, j.segment_id, "claude", cp, per_seg_cost)
                    # Resolved judgment = accuracy-weighted ensemble of every model's output.
                    lp, _ = local.get(j.segment_id, ({}, None))
                    allp = dict(lp); allp["claude"] = cp
                    comb = {h: seg_models.combine_head({m: allp[m][h] for m in allp},
                                                       seg_models.weights_for(sc, h, list(allp)))
                            for h in HEADS}
                    rc = max(comb["content"], key=comb["content"].get)
                    rsw = max(comb["switch"], key=comb["switch"].get)
                    unc = int(j.uncertain or max(comb["content"].values()) < 0.5
                              or max(comb["switch"].values()) < 0.5)
                    # Antecedents (thread pointers): stored raw from the evaluator, never
                    # ensembled - the local heads cannot represent "which thread". Ranked
                    # candidates (top + runner-up + third) with probabilities conditional on
                    # same-thread; a candidate must name a real, EARLIER segment in window.
                    cands = []
                    for cand in (j.antecedents or [])[:3]:
                        if (cand.segment_id in seg_starts and cand.segment_id != j.segment_id
                                and seg_starts[cand.segment_id] < seg_starts.get(j.segment_id, 0)):
                            cands.append((cand.segment_id, min(max(cand.p, 0.0), 1.0)))
                        else:
                            log.warning("dropping invalid antecedent %s for segment %d",
                                        cand.segment_id, j.segment_id)
                    # Rules doc: coming back to something after more than 10 minutes is a shift,
                    # not a return. The pointer says how long ago the old thread was last touched.
                    if rsw == "return" and cands and cands[0][0] in seg_ends:
                        if seg_starts[j.segment_id] - seg_ends[cands[0][0]] > RETURN_MAX_S:
                            rsw = "interruption"
                            sd = nxt = False
                            j.note = (j.note or "") + " [return after more than 10 minutes: stored as a shift]"
                    if rsw == "interruption":
                        cands = []  # a new thread has no antecedent
                    else:
                        sd = nxt = False
                    rk = kind_of(rsw, foc, sd)
                    cands.sort(key=lambda x: -x[1])
                    # The candidates' probabilities are conditional on same-thread and must sum
                    # to 1. The model sometimes returns one candidate carrying 0.2 (or three
                    # summing to 0.7); taken at face value that reads as doubt about the
                    # attachment and raises a pointless review question. Renormalize.
                    tot = sum(p for _, p in cands)
                    if tot > 0:
                        cands = [(i, p / tot) for i, p in cands]
                    elif cands:
                        # Every candidate at 0.0 is a malformed answer, not doubt: stored as is,
                        # the lone survivor read as "0% sure which thread" and raised a thread
                        # question about an obvious return to a meeting (segment 34907,
                        # 2026-11-16). Share the mass evenly: one candidate becomes certain,
                        # several stay a real tie.
                        cands = [(i, 1.0 / len(cands)) for i, _ in cands]
                    ante = cands[0][0] if cands else None
                    if cands:
                        comb["antecedent"] = {str(i): p for i, p in cands}
                    conn.execute(
                        "INSERT OR REPLACE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, probs, created, run_id, antecedent_id, hybrid, focused, self_distraction, next_task)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (j.segment_id, rc, rsw, rk, unc, j.note, json.dumps(comb), now, run_id, ante, int(bool(j.hybrid)),
                         int(foc), int(sd), int(nxt)))
                    n += 1
                # Phone pickups: stored one row each. A pickup the model left out is stored as a
                # continuation (changes nothing), so it is not queued for judgment again and again.
                pj = {j.pickup_id: j for j in (resp.parsed_output.phone or [])}
                for p in picks:
                    j = pj.get(p["key"])
                    psw = stored_switch(j.switch.strip().lower()) if j is not None else "continuation"
                    if psw not in ("continuation", "interruption"):
                        psw = "continuation"
                    pc = j.content.strip().lower() if j is not None else None
                    conn.execute(
                        "INSERT OR REPLACE INTO phone_evals(pickup_key, start, end, apps, switch_label, content, focused,"
                        " self_distraction, rationale, created, run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (p["key"], p["start"], p["end"], ", ".join(f"{a} {d:.0f}s" for a, d, _ in p["apps"]), psw,
                         pc if pc in CONTENTS else "other", int(bool(j.focused)) if j is not None else None,
                         int(bool(j is not None and j.self_distraction and psw == "interruption")),
                         j.note if j is not None else "(model omitted this pickup; defaulted)", now, run_id))
                    n_phone += 1
                # A target the model omitted must not stall the queue forever: store an
                # explicit default the user can audit and correct.
                judged = {j.segment_id for j in resp.parsed_output.segments}
                if held_from is not None:
                    first_held = min((s for s in target if s["clip_start"] >= held_from), key=lambda s: s["clip_start"])
                    conn.execute("INSERT OR REPLACE INTO eval_holds(segment_id, until, created) VALUES (?,?,?)",
                                 (first_held["id"], now + HOLD_RETRY_S, now))
                    n_held = sum(1 for s in target if s["clip_start"] >= held_from)
                for s in target:
                    if held_from is not None and s["clip_start"] >= held_from:
                        continue
                    if s["id"] not in judged:
                        conn.execute(
                            "INSERT OR IGNORE INTO seg_evals(segment_id, content, switch_label, interruption_kind, rationale, created, run_id)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (s["id"], "other", "continuation", None, "(model omitted this segment; defaulted)", now, run_id))
                conn.execute("UPDATE eval_runs SET n_segments=? WHERE id=?", (n + len(deferred), run_id))
        log.info("chunk %s-%s: %d/%d segments judged, %d deferred, %d held for later, %d phone pickup(s) ($%.3f)",
                 dt.datetime.fromtimestamp(t0).strftime("%H:%M"),
                 dt.datetime.fromtimestamp(t1).strftime("%H:%M"), n, len(target), len(deferred), n_held, n_phone, cost)
        # effort-ladder shadows: same prompt at lower efforts, compared + kept for review
        if resp.parsed_output is not None and extra is None:
            from . import efforts as _eff

            def _call(eff):
                return llm.parse(self.client, site="evaluator_shadow", 
                    model=self.cfg.claude_model, max_tokens=8000,
                    system=self._system_blocks(conn),
                    messages=[{"role": "user", "content": content}],
                    output_format=ChunkJudgement,
                    output_config={"effort": eff},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")

            def _cmp(a, b):
                pa = {j.segment_id: (j.content, stored_switch(j.switch), bool(j.focused)) for j in a.segments}
                pb = {j.segment_id: (j.content, stored_switch(j.switch), bool(j.focused)) for j in b.segments}
                keys = set(pa) & set(pb)
                if not keys:
                    return 0.0, "no overlapping segments"
                same = sum(1 for k in keys if pa[k] == pb[k])
                diffs = [f"id={k}: {pa[k]} vs {pb[k]}" for k in keys if pa[k] != pb[k]][:6]
                return same / len(keys), ("; ".join(diffs) or "all verdicts identical")

            _eff.run_shadows(conn, self.cfg, self.client, "segment_evaluator", str(run_id),
                             self.cfg.claude_effort, resp, _call, _cmp)
        return n + len(deferred) + n_phone


def refine_incoherent(conn, cfg: Config | None = None, max_spans: int = 1) -> int:
    """Model judgments are not sacred: when the coherence check catches a span containing two
    objects of attention, RE-EVALUATE its segments with that finding in hand and correct them
    (the user's own edits always take precedence in state derivation and are never the target).
    Bounded: a stretch already refined twice in 24h falls back to asking the user instead."""
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    rows = conn.execute("SELECT * FROM span_summaries WHERE coherent=0 AND refined=0"
                        " ORDER BY created DESC LIMIT ?", (max_spans,)).fetchall()
    n = 0
    for r in rows:
        conn.execute("UPDATE span_summaries SET refined=1 WHERE id=?", (r["id"],))
        prior = conn.execute(
            "SELECT COUNT(*) FROM span_summaries WHERE refined=1 AND id != ? AND start < ? AND end > ? AND created >= ?",
            (r["id"], r["end"], r["start"], time.time() - 86400)).fetchone()[0]
        if prior >= 2:
            log.info("refinement budget exhausted for %s - leaving for the user",
                     dt.datetime.fromtimestamp(r["start"]).strftime("%H:%M"))
            continue
        try:
            extra = (f"a whole-span coherence review of this stretch found: {r['issue']} "
                     f"Re-judge with that in hand. If a NEW piece of work begins mid-stretch, "
                     f"mark the segment where it begins as switch='shift' (with focused=true "
                     f"if the new work is focused).")
            self_n = _get().evaluate_chunk(conn, r["start"] - 1, r["end"] + 1, extra=extra)
            log.info("coherence-driven re-evaluation of %s-%s: %d segments re-judged",
                     dt.datetime.fromtimestamp(r["start"]).strftime("%H:%M"),
                     dt.datetime.fromtimestamp(r["end"]).strftime("%H:%M"), self_n)
            n += 1
        except Exception:
            log.exception("refinement failed")
    return n


def derive_kind(conn, cfg: Config | None = None, segment_id: int = 0, note: str | None = None) -> int:
    """The person ruled a segment a SEPARATE THREAD (switch='interruption') without choosing a
    kind - by design: thread identity is theirs, but whether that thread became a sustained
    object (focus_start) or not (detour) is derived from the recorded data. Re-judge the
    stretch with their ruling in hand. Their switch verdict is sacred (seg_reviews outranks
    field-by-field); only the derived kind shows through the review's NULL kind."""
    cfg = cfg or load_config()
    g = conn.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
    if not g:
        return 0
    t0 = g["start"] - 300
    t1 = min((g["end"] or g["start"]) + 2400, time.time() - 60)  # after-context decides the kind
    extra = (f"The person has RULED that segment id={segment_id} began a SEPARATE thread from "
             f"what preceded it (their ruling is authoritative - do not relitigate it"
             + (f"; their note: {note}" if note else "") + "). "
             f"Your job is the rest: with that segment as switch='shift', re-judge the "
             f"affected segments - whether each was focused or unfocused, whether the shift "
             f"was a self-distraction or the next task in a task block, and the downstream "
             f"switches (continuations of the new thread, returns to the old one).")
    try:
        n = _get().evaluate_chunk(conn, t0, t1, extra=extra)
        log.info("kind derivation after thread ruling on segment %d: %d segments re-judged",
                 segment_id, n)
        return n
    except Exception:
        log.exception("kind derivation failed for segment %d", segment_id)
        return 0


CHAT_TOOLS = [
    {"name": "propose_judgment",
     "description": ("Propose the judgment of THIS segment that the person's clarification implies, for "
                     "them to confirm with one click. Call it when they have told you what was really "
                     "going on and no offered option (or your own judgment) captures it, or when they "
                     "correct you. Do NOT call it for a mere question, or when an existing option already "
                     "says the same thing (point them to that option instead). The label must be in "
                     "their own words, as a review-question option would read."),
     "input_schema": {"type": "object", "properties": {
         "label": {"type": "string", "description": "The answer as an option label, in the person's own terms, e.g. 'A window that was just open on my desktop, not looked at'."},
         "content": {"type": ["string", "null"], "enum": list(CONTENTS) + [None], "description": "Content label this asserts, or null to leave the content as judged."},
         "switch_label": {"type": ["string", "null"], "enum": list(SHOWN_SWITCHES) + [None], "description": "continuation | shift (a separate thread began) | return, or null to leave the thread relation as judged."},
         "focused": {"type": ["boolean", "null"], "description": "true = the person says their attention was focused in this segment; false = they say it was unfocused; null = no ruling on that."},
         "antecedent_id": {"type": ["integer", "null"], "description": "For continuation/return: the id of the segment this continues (from the log), else null."},
         "hybrid": {"type": ["boolean", "null"], "description": "true = the activity was done inside a live discussion; false = solo; null = no ruling."},
         "note": {"type": "string", "description": "One sentence, in the person's words, stored as the precedent future judgments see: what was really going on and why it is labelled this way."}},
         "required": ["label", "note"]}},
    {"name": "propose_rule",
     "description": ("Propose ONE general rule for the person's rules document, for them to approve. Only "
                     "when their clarification reveals a GENERAL lesson the current rules/precedents miss "
                     "(e.g. 'sub-15-second windows with no input are pass-throughs, whatever the page') - "
                     "never a restatement of this one case, and never something the rules already say."),
     "input_schema": {"type": "object", "properties": {
         "rule": {"type": "string", "description": "The rule, one or two sentences, in the second person as the rules doc is written ('If I ...')."}},
         "required": ["rule"]}},
]


def chat_reply(conn, cfg: Config, segment_id: int, messages: list[dict],
               question_id: int | None = None) -> dict:
    """Talk to "the one who made the decision": reconstitute the EXACT evaluation conversation
    (same system prompt, same user message with the same screenshots, its actual response), then
    continue it with the reviewer's questions - so mistakes can be triangulated and the
    instruction that would have prevented them identified. Falls back to an informed stand-in
    for segments judged by the local models (deferral) or before transcripts were kept."""
    import json as _json
    from .classifiers.claude_vision import _b64
    ev_row = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    run = None
    if ev_row is not None and ev_row["run_id"]:
        run = conn.execute("SELECT * FROM eval_runs WHERE id=?", (ev_row["run_id"],)).fetchone()
    seg = conn.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
    who = f"segment id={segment_id} ({seg['app']} | {(seg['title'] or '')[:60]})" if seg else f"segment id={segment_id}"
    ev = _get()
    if run is not None and run["system"] and run["response"]:
        system = [{"type": "text", "text": run["system"], "cache_control": {"type": "ephemeral"}}]
        content: list[dict] = []
        for m in _json.loads(run["shots"] or "[]"):
            b = _b64(m["path"])
            if b:
                content.append({"type": "text", "text": m["label"]})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        content.append({"type": "text", "text": run["prompt"]})
        api_msgs: list[dict] = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": run["response"],
                                               "cache_control": {"type": "ephemeral"}}]}]
        preface = (f"[The person you monitor is now reviewing your judgments above, in particular "
                   f"{who}. Answer their questions about what you saw and why you judged as you did; "
                   f"if you now think you were wrong, say so plainly and say what instruction or "
                   f"information would have led you to the right judgment. You cannot change any "
                   f"record yourself: when they tell you what was really going on, call propose_judgment "
                   f"(and propose_rule for a general lesson) - the proposal appears as a button they "
                   f"click to apply; otherwise nothing you say changes anything.]\n\n")
    else:
        # No transcript: an informed stand-in built from what actually decided this segment.
        preds = conn.execute("SELECT model, p_content, p_switch, p_kind FROM seg_predictions WHERE segment_id=?",
                             (segment_id,)).fetchall()
        ptxt = "\n".join(f"  {p['model']}: content={p['p_content']} switch={p['p_switch']}" for p in preds) or "  (none stored)"
        vtxt = (f"resolved: content={ev_row['content']}, switch={shown_switch(ev_row['switch_label'])}, "
                f"kind={ev_row['interruption_kind']}, rationale: {ev_row['rationale']}") if ev_row else "(no evaluation recorded)"
        system = [{"type": "text", "text": ev._system_blocks(conn)[0]["text"], "cache_control": {"type": "ephemeral"}}]
        api_msgs = []
        preface = (f"[No verbatim transcript exists for the evaluation of {who} - it was judged "
                   f"before transcripts were kept, or deferred to the local (non-LLM) models. "
                   f"What is on record:\n{vtxt}\nPer-model probability outputs:\n{ptxt}\n"
                   f"Help the person triangulate whether the judgment was right.]\n\n")
    if question_id is not None:
        # A follow-up on a review question: the person is looking at that question and its
        # options, so the evaluator must know exactly what was asked.
        qrow = conn.execute("SELECT * FROM review_questions WHERE id=?", (question_id,)).fetchone()
        if qrow is not None:
            opts = "\n".join(f"  {n}. {o.get('label')}" for n, o in enumerate(_json.loads(qrow["options"]), 1))
            if qrow["status"] == "none":
                # Every option was rejected: the job is to find the reading that fits and hand
                # it back as a proposal (which becomes a new option when applied).
                preface += (f"[The person was shown this review question about that segment and answered "
                            f"NONE OF THE ABOVE - no option describes what actually happened"
                            + (f"; their note: \"{qrow['note']}\"" if qrow["note"] else "")
                            + f". Work out with them what it really was (ask one concrete thing if you "
                            f"must), then call propose_judgment with the answer in their words so it can "
                            f"be applied. Be brief.\n"
                            f"Context shown: {qrow['context']}\n"
                            f"Question: {qrow['question']}\nRejected options:\n{opts}]\n\n")
            else:
                preface += (f"[The person was shown this review question about that segment and is asking "
                            f"a follow-up before answering it. Help them decide: say concretely what you "
                            f"saw (screens, titles, times), what would distinguish the options, and which "
                            f"option you lean toward and how sure you are. Be brief.\n"
                            f"Context shown: {qrow['context']}\n"
                            f"Question: {qrow['question']}\nOptions:\n{opts}]\n\n")
    for i, m in enumerate(messages):
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = (m.get("text") or "").strip()
        if role == "user" and preface:
            text = preface + text
            preface = ""
        api_msgs.append({"role": role, "content": text})
    model = run["model"] if run is not None and run["model"] else cfg.claude_model
    system = llm.with_preamble(system)
    tools = CHAT_TOOLS + [llm.CONTACT_TOOL]
    resp = ev.client.messages.create(model=model, max_tokens=1024, system=system, messages=api_msgs,
                                     tools=tools, output_config={"effort": "low"})
    from .classifiers.claude_vision import _cost
    rec = llm.contacted("evaluator_chat", model, system, api_msgs, resp, tools)
    text = (llm.contact_notice(rec) if rec
            else "(Claude declined to answer that.)" if resp.stop_reason == "refusal"
            else "".join(b.text for b in resp.content if b.type == "text").strip())
    proposal: dict = {}
    for b in ([] if rec else resp.content):
        if b.type != "tool_use":
            continue
        if b.name == "propose_judgment":
            j = dict(b.input)
            if j.get("content") not in CONTENTS:
                j["content"] = None
            j["switch_label"] = stored_switch(j.get("switch_label"))
            if j["switch_label"] not in SWITCHES:
                j["switch_label"] = None
            if j["switch_label"] not in ("continuation", "return"):
                j["antecedent_id"] = None
            if not (j.get("label") or "").strip():
                continue
            proposal["judgment"] = {"label": j["label"].strip(), "content": j["content"],
                                    "switch_label": j["switch_label"], "antecedent_id": j.get("antecedent_id"),
                                    "hybrid": j.get("hybrid"), "focused": j.get("focused"),
                                    "note": (j.get("note") or "").strip()}
        elif b.name == "propose_rule" and (b.input.get("rule") or "").strip():
            proposal["rule"] = b.input["rule"].strip()
    if not text:
        text = "(proposal below)" if proposal else "(no reply)"
    from .chat import log_exchange
    mid = log_exchange(conn, "evaluator", str(segment_id), messages, text, _cost(resp.usage), model,
                       question_id, proposal or None)
    return {"reply": text, "proposal": proposal or None, "message_id": mid}


class AuditFlag(BaseModel):
    segment_id: int
    field: str = Field(description="Which judgment looks wrong: content, switch, or focused.")
    suggested: str = Field(description="The label you would give instead.")
    reason: str = Field(description="One short sentence.")


class DayAudit(BaseModel):
    flags: list[AuditFlag] = Field(description="ONLY segments whose judgment looks wrong. Empty list if the day looks right.")


AUDIT_PROMPT = """Below is one full day of this person's activity, with the resolved judgments \
(state, [content=...], switch labels) each segment received - some judged by cheap local models. \
Re-check the day as a whole. List ONLY the segments whose content or switch judgment looks wrong, \
with what you'd say instead. These get flagged for the person to review, so precision matters more \
than recall - do not nitpick defensible calls. An empty list is the expected answer for a clean day.

{log}"""


def daily_audit(conn, cfg: Config | None = None) -> int:
    """Once per day: one LLM sweep over yesterday's resolved judgments; suspected errors are
    flagged (uncertain=1) for the user's review queue - labels are never silently changed."""
    from . import stats, summaries
    from .config import logical_date
    cfg = cfg or load_config()
    if not cfg.daily_audit:
        return 0
    day = logical_date() - dt.timedelta(days=1)
    key = day.isoformat()
    if conn.execute("SELECT 1 FROM audit_runs WHERE day=?", (key,)).fetchone():
        return 0
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    t0, t1 = stats.day_bounds(day)
    segs = [s for s in stats.labelled_segments(conn, t0, t1, phone=False)
            if not s["idle"] and not s.get("neutral") and s["duration"] > 0]
    evaluated = [s for s in segs if s.get("content")]
    if not evaluated:
        conn.execute("INSERT INTO audit_runs(day, n_flags, cost_usd, created) VALUES (?,?,?,?)",
                     (key, 0, 0.0, time.time()))
        return 0
    ids = [s["id"] for s in segs]
    rats = {}
    if ids:
        qsr = ",".join("?" * len(ids))
        rats = {r2["segment_id"]: r2["rationale"] for r2 in conn.execute(
            f"SELECT segment_id, rationale FROM seg_evals WHERE segment_id IN ({qsr})", ids)
            if r2["rationale"]}
    lines = [f"id={s['id']} [content={s.get('content') or '?'}] " + summaries.seg_line(s)
             + (f"  // {rats[s['id']][:80]}" if s["id"] in rats else "") for s in segs]
    content: list = []
    if ids:
        from .classifiers.claude_vision import _b64
        from . import shots as _shots
        for sh in _shots.pick(conn, ids, 8):
            b = _b64(sh["path"])
            if b:
                content.append({"type": "text", "text": f"Screenshot at {dt.datetime.fromtimestamp(sh['ts']).strftime('%H:%M')} (segment id={sh['segment_id']}){' - EXTERNAL display' if sh.get('display') else ''}:"})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
    content.append({"type": "text", "text": AUDIT_PROMPT.format(log="\n".join(lines))})
    ev = _get()
    resp = llm.parse(ev.client, site="audit", 
        model=cfg.claude_model, max_tokens=4000,
        system=ev._system_blocks(conn),
        messages=[{"role": "user", "content": content}],
        output_format=DayAudit,
        output_config={"effort": cfg.claude_effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default")
    cost = ((resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
            + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
            + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)
    if resp.parsed_output is not None:
        from . import efforts as _eff

        def _acall(eff):
            return llm.parse(ev.client, site="audit_shadow", 
                model=cfg.claude_model, max_tokens=4000,
                system=ev._system_blocks(conn),
                messages=[{"role": "user", "content": content}],
                output_format=DayAudit,
                output_config={"effort": eff},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")

        def _acmp(a, b):
            fa = {(f.segment_id, f.field) for f in a.flags}
            fb = {(f.segment_id, f.field) for f in b.flags}
            union = fa | fb
            if not union:
                return 1.0, "both flagged nothing"
            same = len(fa & fb) / len(union)
            return same, f"flags primary={sorted(fa)} vs {sorted(fb)}"

        _eff.run_shadows(conn, cfg, ev.client, "daily_audit", key, cfg.claude_effort,
                         resp, _acall, _acmp, prompt_text=AUDIT_PROMPT.format(log="(day log, see audit)")[:200])
    n = 0
    ids = {s["id"] for s in evaluated}
    with db.tx(conn):
        if resp.parsed_output is not None:
            for f in resp.parsed_output.flags:
                if f.segment_id not in ids:
                    continue
                if conn.execute("SELECT 1 FROM seg_reviews WHERE segment_id=?", (f.segment_id,)).fetchone():
                    continue  # the user already ruled on this one
                conn.execute("UPDATE seg_evals SET uncertain=1, rationale = COALESCE(rationale,'') || ? WHERE segment_id=?",
                             (f" | daily audit: {f.field} looks like '{f.suggested}' - {f.reason}", f.segment_id))
                n += 1
        conn.execute("INSERT INTO audit_runs(day, n_flags, cost_usd, created) VALUES (?,?,?,?)",
                     (key, n, cost, time.time()))
    log.info("daily audit %s: %d flag(s) ($%.3f)", key, n, cost)
    return n


def _invalidate_derived(conn, t0: float, t1: float, questions: bool, summaries: bool) -> tuple[int, int]:
    """Drop what was built ON TOP of judgments that just changed, so it gets rebuilt from the new
    ones: open review questions for re-judged segments (answered ones stay - the person's ruling
    stands), and span summaries overlapping the stretch (the summarizer regenerates any span
    without one, and the coherence check re-runs on the new summary)."""
    nq = ns = 0
    if questions:
        nq = conn.execute("""DELETE FROM review_questions WHERE status IN ('open', 'declined') AND segment_id IN
                             (SELECT id FROM segments WHERE start >= ? AND start < ?)""", (t0, t1)).rowcount
    if summaries:
        ns = conn.execute("DELETE FROM span_summaries WHERE start < ? AND end > ?", (t1, t0)).rowcount
    return nq, ns


WRONG_DISPLAY_REASON = "screenshots shown were of the wrong display"
PHONE_REASON = "phone data arrived"
PHONE_LOOKBACK_S = 3 * 86400     # the phone pushes its log hours to a day late
PHONE_MERGE_S = 300.0            # pickups this close together are re-judged in one call
_phone_seen = (None, 0.0)        # (phone.through, when last checked)


def _dur(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.0f}m"


def queue_phone_rejudge(conn, cfg: Config | None = None) -> int:
    """Phone data arrives after the Mac activity it overlaps has been judged (Eli, 2026-11-19:
    "re-judge when the phone data lands"). Queue a re-judgment (reeval_queue) of each stretch
    that holds a pickup with no judgment, so the evaluator sees it again with the pickup in
    view. Only stretches that contain a pickup, and only behind the point the first-pass
    evaluation has reached - pickups ahead of it are judged in the first pass. Each queued
    pickup gets a placeholder row in phone_evals (no label) so it is queued once."""
    global _phone_seen
    from . import phone
    cfg = cfg or load_config()
    thr = phone.through(cfg)
    now = time.time()
    if thr is None or (thr == _phone_seen[0] and now - _phone_seen[1] < 1800):
        return 0
    _phone_seen = (thr, now)
    horizon = now - LAZY_S
    row = conn.execute("""
        SELECT MIN(g.start) FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
        WHERE g.start >= ? AND g.start < ? AND g.idle = 0 AND g.neutral = 0
          AND COALESCE(g.end, g.start) > g.start AND e.segment_id IS NULL""",
        (now - PHONE_LOOKBACK_S, horizon)).fetchone()
    frontier = min(row[0], horizon) if row and row[0] else horizon
    have = {r[0] for r in conn.execute("SELECT pickup_key FROM phone_evals WHERE start >= ?",
                                       (now - PHONE_LOOKBACK_S - 3600,))}
    new = [p for p in phone.pickups(cfg, now - PHONE_LOOKBACK_S, frontier)
           if p["key"] not in have and p["end"] < frontier]
    windows: list[list[float]] = []
    with db.tx(conn):
        for p in new:
            on_mac = conn.execute("SELECT 1 FROM segments WHERE start < ? AND COALESCE(end, ?) > ? LIMIT 1",
                                  (p["end"], now, p["start"])).fetchone()
            conn.execute("INSERT OR IGNORE INTO phone_evals(pickup_key, start, end, apps, rationale, created)"
                         " VALUES (?,?,?,?,?,?)",
                         (p["key"], p["start"], p["end"], ", ".join(f"{a} {d:.0f}s" for a, d, _ in p["apps"]),
                          None if on_mac else "(the Mac was not recording: not judged)", now))
            if not on_mac:
                continue
            a, b = p["start"] - 30, p["end"] + 30
            if windows and a - windows[-1][1] <= PHONE_MERGE_S and b - windows[-1][0] <= CHUNK_S:
                windows[-1][1] = max(windows[-1][1], b)
            else:
                windows.append([a, b])
        for a, b in windows:
            conn.execute("INSERT INTO reeval_queue(t0, t1, reason, created) VALUES (?,?,?,?)",
                         (a, b, PHONE_REASON, now))
    if windows:
        log.info("phone data through %s: %d pickup(s) in %d stretch(es) queued for judgment",
                 dt.datetime.fromtimestamp(thr).strftime("%m-%d %H:%M"), len(new), len(windows))
    return len(windows)


def queue_wrong_display_rejudge(conn) -> int:
    """Queue for re-judgment every stretch whose current judgment came from a run that showed
    the evaluator a segment only on a display that did NOT hold its active window (the
    row-sampling bug fixed 2026-11-16, see shots.py). eval_runs.shots records what each run
    saw. Safe to rerun: a re-judged segment belongs to a new run that saw the right display,
    and a stretch already waiting in the queue is not added twice."""
    import json as _json
    from .shots import _display_of
    seg_of = {r["path"]: r["segment_id"] for r in conn.execute("SELECT segment_id, path FROM segment_shots")}
    active = {r["id"]: {d for d in (_display_of(r["first_screenshot"]), _display_of(r["last_screenshot"])) if d is not None}
              for r in conn.execute("SELECT id, first_screenshot, last_screenshot FROM segments")}
    owner = {r["segment_id"]: r["run_id"] for r in conn.execute("SELECT segment_id, run_id FROM seg_evals")}
    bad = []
    for r in conn.execute("SELECT id, t0, t1, shots FROM eval_runs WHERE shots IS NOT NULL ORDER BY t0"):
        shown: dict[int, set] = {}
        for s in _json.loads(r["shots"] or "[]"):
            if s["path"] in seg_of:
                shown.setdefault(seg_of[s["path"]], set()).add(_display_of(s["path"]))
        if any(active.get(seg) and not (ds & active[seg]) and owner.get(seg) == r["id"] for seg, ds in shown.items()):
            bad.append((r["t0"], r["t1"]))
    windows: list[list[float]] = []
    for t0, t1 in bad:
        if windows and t0 - windows[-1][1] < 60:
            windows[-1][1] = max(windows[-1][1], t1)
        else:
            windows.append([t0, t1])
    n = 0
    for t0, t1 in windows:
        if conn.execute("SELECT 1 FROM reeval_queue WHERE done_at IS NULL AND t0 <= ? AND t1 >= ?", (t0, t1)).fetchone():
            continue
        conn.execute("INSERT INTO reeval_queue(t0, t1, reason, created) VALUES (?,?,?,?)",
                     (t0, t1, WRONG_DISPLAY_REASON, time.time()))
        n += 1
    log.info("queued %d stretch(es) for re-judgment: %s", n, WRONG_DISPLAY_REASON)
    return n


def run_reevaluations(conn, cfg: Config | None = None, max_chunks: int = 1) -> int:
    """Re-judge stretches whose upstream data changed after they were evaluated (reeval_queue -
    e.g. a Toggl entry the person corrected later). Same evaluator, same context, plus a note
    saying why; the person's own edits still outrank the result. Each re-judged chunk retires
    the open questions written from the old judgments; when the whole stretch is done its span
    summaries are dropped so they are rewritten from the new judgments."""
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    done = 0
    for _ in range(max_chunks):
        # Newest first: recent days are the ones still under review.
        q = conn.execute("SELECT * FROM reeval_queue WHERE done_at IS NULL ORDER BY t0 DESC LIMIT 1").fetchone()
        if not q:
            break
        horizon = time.time() - LAZY_S
        start = max(q["t0"], q["done_to"] or q["t0"])
        end = min(q["t1"], horizon)
        if end - start < 30:   # nothing (left) to re-judge
            _, ns = _invalidate_derived(conn, q["t0"], q["t1"], questions=False, summaries=True)
            conn.execute("UPDATE reeval_queue SET done_at=? WHERE id=?", (time.time(), q["id"]))
            continue
        c1 = end if end - start <= CHUNK_S * 1.5 else start + CHUNK_S
        if q["reason"] == PHONE_REASON:
            extra = ("the phone log for this window arrived AFTER the computer activity was judged: the "
                     "PHONE lines are new. Judge each one marked EVALUATE PHONE, against the activity "
                     "around it. No computer segment is up for judgment in this call.")
        elif q["reason"] == WRONG_DISPLAY_REASON:
            extra = ("your earlier judgment of this window was made from screenshots of the WRONG "
                     "display: the person had two monitors and you were shown the one they were not "
                     "working on. The screenshots above are of the display that held the active "
                     "window. Re-judge every segment from them; where your earlier judgment "
                     "(including the 'prior judgment' notes) rested on what the other screen showed, "
                     "change it.")
        else:
            extra = (f"the {q['reason']} for this window AFTER your earlier judgment of it - the Toggl "
                     f"entries listed above are the corrected ones and are AUTHORITATIVE: if a screenshot "
                     f"shows a Toggl timer with a different name or still running, that is the uncorrected "
                     f"state and must be ignored. Re-judge every segment against the listed entries; where "
                     f"your earlier judgment (including the 'prior judgment' notes) relied on the old "
                     f"entries, change it.")
        phone_q = q["reason"] == PHONE_REASON
        try:
            n = _get().evaluate_chunk(conn, start, c1, extra=extra, phone_only=phone_q)
        except Exception:
            log.exception("re-evaluation failed (%s)", dt.datetime.fromtimestamp(start))
            conn.execute("UPDATE reeval_queue SET done_at=? WHERE id=?", (time.time(), q["id"]))
            break
        # (a phone re-judgment leaves the segments' judgments, and so their questions, as they were)
        nq, _ = _invalidate_derived(conn, start, c1, questions=not phone_q, summaries=False)
        finished = c1 >= end
        if finished:
            _, ns = _invalidate_derived(conn, q["t0"], q["t1"], questions=False, summaries=True)
            conn.execute("UPDATE reeval_queue SET done_to=?, done_at=? WHERE id=?", (c1, time.time(), q["id"]))
        else:
            ns = 0
            conn.execute("UPDATE reeval_queue SET done_to=? WHERE id=?", (c1, q["id"]))
        log.info("re-evaluated %s-%s (%s): %d segments, %d stale question(s) retired%s",
                 dt.datetime.fromtimestamp(start).strftime("%m-%d %H:%M"),
                 dt.datetime.fromtimestamp(c1).strftime("%H:%M"), q["reason"], n, nq,
                 f", {ns} span summary(ies) dropped for rewrite" if finished else "")
        done += 1
    return done


def run_pending(conn, cfg: Config | None = None, max_chunks: int = 1, lookback_s: float | None = None) -> int:
    """Evaluate the oldest unevaluated stretch that is at least LAZY_S old. Called from the
    classifier loop; also usable for backfill with a longer lookback and more chunks."""
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    if lookback_s is None:  # default scope: the current logical day only (user 2026-09-01)
        from .config import day_bounds, logical_date
        lookback_s = time.time() - day_bounds(logical_date())[0]
    horizon = time.time() - LAZY_S
    done = 0
    for _ in range(max_chunks):
        row = conn.execute("""
            SELECT g.start, g.id FROM segments g
            LEFT JOIN seg_evals e ON e.segment_id = g.id
            WHERE g.start >= ? AND g.start < ? AND g.idle = 0 AND g.neutral = 0
              AND COALESCE(g.end, g.start) > g.start
              AND COALESCE(g.end, g.start) <= ? AND e.segment_id IS NULL
            ORDER BY g.start LIMIT 1""",
            (time.time() - lookback_s, horizon, horizon)).fetchone()
        if not row or row[0] is None:
            break
        # The evaluator asked to judge this segment later (hold-off): wait until then. Everything
        # after it waits too, so spans are never assembled across an unjudged gap.
        hold = conn.execute("SELECT until FROM eval_holds WHERE segment_id=?", (row[1],)).fetchone()
        if hold and hold["until"] > time.time():
            break
        t0 = row[0] - 1
        t1 = min(t0 + CHUNK_S, horizon)
        try:
            if _get().evaluate_chunk(conn, t0, t1) == 0:
                break  # nothing judged: don't spin on the same window
        except Exception:
            log.exception("chunk evaluation failed (%s)", dt.datetime.fromtimestamp(t0))
            break
        done += 1
    return done

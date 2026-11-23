"""Model-generated review questions - the primary review stream.

Instead of showing the user raw switches to label, the model identifies the judgments most
worth checking and asks a CONTEXTUALIZED question about each, in natural language, with
options that map onto concrete segment edits underneath. The canonical framing rule (the
user's own example): after writing -> brief email -> long Claude conversation, the pertinent
question is NOT "did attention shift between email and Claude?" (trivially yes) but "Is
talking to Claude a continuation of the writing task, or a shift of attention?"

Question sources - self-reported uncertainty alone misses confident errors, so the checkers
feed in too:
  self_uncertain - the evaluator flagged its own judgment
  disagreement   - the ensemble's models disagree on a head
  coherence      - a span's own summary contradicts the focus-span definition (boundary doubt)
  audit          - the end-of-day sweep flagged a suspected error
Answers write through as seg_reviews (training all models) and mark the question answered."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import (PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, Config,
                     load_config)
from .seg_models import CONTENTS, KINDS, SWITCHES, shown_switch, stored_switch

log = logging.getLogger("questions")

MAX_TRIGGERS_PER_CALL = 6


class QOption(BaseModel):
    label: str = Field(description="The answer in the person's own terms, e.g. 'a continuation of the essay writing'.")
    segment_id: int = Field(description="The segment this option's edit applies to.")
    switch_label: str | None = Field(default=None, description="continuation | shift | return, when the option settles the thread relation. (shift = attention moved to something new, a separate thread.)")
    antecedent_id: int | None = Field(default=None, description="When the option asserts SAME thread (continuation/return): the id of the segment in that thread the person would be agreeing this continues - the A-thread segment your question names. Null for 'separate thread' (shift) and content-only options.")
    focused: bool | None = Field(default=None, description="Whether the person's attention was focused in the segment, when the option's reading says: true for a reading in which they were getting something specific done (an errand, a lookup, the task itself), false for a reading in which they were not (a drift, browsing with nothing in mind). None when the reading does not say.")
    self_distraction: bool | None = Field(default=None, description="For a segment that is a shift or a return (or that this option makes one): true when the option's reading is that they drifted there without deciding to, false when its reading is that they went there on purpose. Set on every option of such a question; None only when the reading says nothing about why they went there.")
    grew_out_of_id: int | None = Field(default=None, description="Only on an option with switch_label='shift' whose reading is that the new thing GREW OUT OF the thread before it (a sub-thread): the id of the nearest segment of that parent thread. Null on every other option.")
    content: str | None = Field(default=None, description="A content label, when content is what's in doubt.")
    hybrid: bool | None = Field(default=None, description="True when the option says the activity was being done INSIDE a live discussion with someone (e.g. 'reading the bill with Dave on a call while taking notes') - content stays the activity; False when the option asserts solo work; None when the option says nothing about it.")


class GeneratedQuestion(BaseModel):
    trigger_segment_id: int
    context: str = Field(description="1-2 sentences of narrative: what the person was doing before/around this moment. Sets up the question; no jargon.")
    question: str = Field(description="The decision-relevant question, phrased against the WORK THREAD, not the adjacent window.")
    options: list[QOption] = Field(description="2-4 mutually exclusive answers. Every option must map to a concrete judgment (fields set); an 'it was all one thing' option may set switch_label='continuation' or 'return'.")


class QuestionBatch(BaseModel):
    questions: list[GeneratedQuestion]


SYSTEM = """You write review questions for a personal attention monitor. The system has judged \
every segment of the person's day (its content; whether their attention in it was focused or \
unfocused; and the switch into it - a continuation, a shift of their object of attention, or a \
return - with, on a shift, whether it was a self-distraction), but some judgments are in doubt. For each DOUBT presented, you write ONE question \
the person can answer in seconds, with 2-4 natural-language options.

DESCRIBE THE ACTIVITY, NOT THE WINDOWS. The context must say what the person was DOING - the
undertaking - inferred from the screenshots, titles, content labels, and the per-segment notes \
(after "// note:" in the log): "you were editing your personal site - coding in VS Code, running it in Terminal, \
checking the rendered pages" rather than "going back and forth between Code, Terminal, and \
elityre.com". Use epistemic markers ("apparently", "likely") when inferring. Window names are \
evidence, not the description.

THE FRAMING RULE (the person's own words): if they were writing, briefly checked email, then \
started a long Claude conversation, the pertinent question is NOT "did your attention shift \
between email and Claude?" (trivially yes) - it is "You were writing, then briefly went to your \
email. Then you started talking to Claude. Was talking to Claude a continuation of the writing \
task, or a shift in your focus of attention?" Always frame against the WORK THREAD, naming what \
they were doing, skipping over contained detours.

WHAT TO ASK (the person's own division of labor): the person is ground truth for INTENTION and \
for AMBIGUOUS CONTENT - nothing else. Two canonical thread forms:
  1. "You were doing A, then switched to B. Was B part of the A task, or a separate thread?" \
- same thread: switch_label='continuation'; separate thread: switch_label='shift'.
  2. "You were doing A, briefly went to B, then switched to C. Was C the same thread/intention \
as A, or something different?" - same: switch_label='return'; different: \
switch_label='shift'.
Content questions are equally welcome when content is the doubt: "was drafting that Signal \
message routine correspondence ('correspondence') or writing you were absorbed in ('creative')?" - each \
option sets a content label.
ONE THING PER QUESTION (the person's rule, 2026-11-12): a question resolves exactly one dimension \
- the thread relation (same thread, and which, or a separate one), OR the content, OR whether it \
was done inside a live discussion (hybrid). Its options differ along that one dimension and \
nothing else. When a DOUBT is uncertain along two dimensions, write two questions for the same \
trigger_segment_id - "was the Amazon stretch part of the checklist work, or its own thing?" and \
"were you deliberately getting an errand done there, or perusing without intent?" - never one \
question whose options mix them. "Drifting", "browsing without intent", "perusing", "getting \
pulled in" is content='passive' and belongs to a content question; on a thread question such \
wording is just flavour, and two thread options that differ only in words about intent or mood \
are ONE option.
ON PURPOSE OR A DRIFT (added 2026-11-20). On every shift, and on a return to a distraction, the \
app keeps one more judgment: self-distraction - the person drifted there without deciding to - \
and each one is counted as an event in their day. The current judgment line of a DOUBT says \
"self-distraction" when the app has it as one. An answer changes that record only through the \
option's own fields, so the fields must say everything the label says: before this rule, the \
answer "I was after something specific: an errand" set content='task' and nothing else, and the \
visit stayed counted as a self-distraction and as unfocused time. Therefore, when the segment in \
doubt is a shift or a return (switch=shift or switch=return in its current judgment), or an \
option would make it one:
  - set self_distraction on EVERY option: true when its reading is that they drifted there \
without deciding to ("a drift, opened it with nothing in mind", "browsing the deals with no \
particular goal"), false when its reading is that they went there on purpose ("an errand", \
"checking something specific", "a planned break", "part of the A task"). Leave it null only on \
an option whose reading says nothing about why they went there (a bare "a separate thread").
  - set focused on every option whose reading implies it: true for getting something specific \
done, false for a drift.
This is not a dimension of its own and never a third question: it travels with the reading. The \
question "were you after something specific, or was it a drift?" is the content question, and \
its options are (content='task', focused=true, self_distraction=false) against \
(content='passive', focused=false, self_distraction=true).
A thread question presupposes a thread: it only makes sense when there has been enough \
sustained attention for there to be one. A few short segments of poking around (a minute here, \
two minutes there, inside a bigger stretch of something else) are not a thread to be "the same \
as" or "different from" - do not ask whether such a segment continues or starts one; ask about \
its content if that is in doubt, otherwise omit the DOUBT.
In form 2, the "something different" option must say what the DIFFERENT undertaking on \
return would be ("back on the page only to grab a quote for the message you were writing"), \
never that the diversion "had become the real focus" or "was a new beginning": how long the \
diversion ran, and whether that ended the span, is the app's break rule (5 minutes), not the \
person's call - and a diversion under the threshold cannot have become a span at all. If you \
cannot name a plausible different undertaking on return, the doubt is not a thread question; \
omit it.
In general, do NOT ask the person whether something was "a quick detour or the start of a new \
focus" - that answer mixes two facts: whether it was a separate thread (theirs to say) and \
whether it went on long and coherently enough to be its own span (the app records the data and \
derives this). Options therefore never say what kind of shift it was; asserting 'shift' is \
enough - the app works out the rest from what followed. The standing exception: a confusion or \
uncertainty the system genuinely cannot resolve from its data should always be surfaced to the \
person - but then the question must name the actual confusion in its own terms, not launder it \
into a label choice.

Rules for options:
- Natural language first ("a continuation of the essay work"), schema underneath: each option \
sets switch_label (a thread relation) or a content label, plus self_distraction and focused \
where ON PURPOSE OR A DRIFT asks for them.
- A SAME-thread option (continuation/return) also names its antecedent_id: the id of the \
segment in thread A that the person would be agreeing this continues (pick the nearest \
A-thread segment visible in the log). This makes the answer an edge in the thread graph, \
not just a label.
- For boundary and coherence doubts (a stretch that may contain TWO objects of attention), ask \
about the moment the SECOND object would begin - the split point named or implied in the \
doubt's reason - and you MUST include an option asserting the separate thread: \
switch_label='shift' on the segment where it begins (its id is visible in the log). \
"Was <activity> at <time> a separate piece of work, or still part of <thread A>?" is the shape.
- SUB-THREADS. Between "still part of A" and "a separate thread" there is a third reading, and \
when it is plausible you MUST offer it as its own option: the new thing GREW OUT OF A and \
became an object of attention of its own (fixing the tool the task depends on; from planning \
the day with an assistant to reworking how the assistant behaves). Word it in the person's \
terms ("A sub-thread: it grew out of the nap planning and became its own thing") and set \
switch_label='shift', grew_out_of_id=<the nearest segment of A>, self_distraction=false. The \
plain separate-thread option then means "unrelated to A", and the same-thread option means "it \
was A itself". The app decides what a sub-thread does to the span (under 10 minutes it is \
counted inside A); the person only says how the two relate.
- Options must be mutually exclusive IN THE EDIT THEY MAKE and cover the plausible readings. \
Two options that set the same fields to the same values are one option, however differently \
worded: never split "a separate thread" by what came after it ("a one-off question" vs "the \
start of a run of lookups") - what the following segments continue is judged on those \
segments. Do not editorialize; the person decides.
- NEVER write a "just passing through" / transition option yourself: the app appends a neutral \
one ("a window I passed through, not actually looked at") to EVERY question. Your options are \
the substantive readings only - what the moment was if the person DID engage with it. In \
particular, do not frame a pass-through as "part of the <thread A> conversation": a window \
cycled past on the way back is not part of any thread's content, and an option that says it \
is becomes a false precedent.

WHAT YOU ARE SHOWN FOR EACH DOUBT: screenshots of the segment in doubt (several when it was \
long) and one of each segment it might continue; why it is in doubt, in full - for a doubt \
raised by one of the person's own rulings on another segment, that ruling; for a doubt about a \
span, the span; the app's current judgment with its whole note; the probabilities its models \
gave each label, and the evaluator's ranked candidates for what the segment continues, each with \
its log line (name one of THESE in an option's antecedent_id when it is the thread you mean); \
and the activity log from 20 minutes before the segment to 20 minutes after its end, where a \
line the person has already ruled on says RULED BY THE PERSON with their note - a ruling next to \
the doubt often answers it, and then the DOUBT is to be omitted.

{legend}

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""


_gen = None


def _client():
    global _gen
    if _gen is None:
        import anthropic
        from .config import load_api_key
        load_api_key()
        c = anthropic.Anthropic()
        if not (getattr(c, "api_key", None) or getattr(c, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials")
        _gen = c
    return _gen


def _triggers(conn, lookback_s: float = 3 * 86400) -> list[dict]:
    """Doubts worth a question, newest first, deduped against existing questions/reviews."""
    t0 = time.time() - lookback_s
    out = []
    # No time cutoff on uncertainty flags: every flag lights a needs-review dot on the
    # timeline, and every dot must have a question waiting - however old the segment.
    # A declined, dismissed or archived row does not count as that question: each of those
    # clears the flag when it is written, so a flag that is up now was raised AFTER it (a
    # correction propagating, the daily audit) and is a new doubt. Counting them left dots with
    # no question behind them (segment 34836, 2026-11-16: declined as a thread doubt at 17:40,
    # then flagged by propagation, and never asked about).
    # A block the person settled without a ruling ("can't tell", "none of these") and that a later
    # re-judgment flagged again: asking again cannot help, and the join below would leave its dot
    # with no question behind it (segment 34748, 2026-11-21). The flag is dropped.
    conn.execute("""UPDATE seg_evals SET uncertain = 0 WHERE uncertain = 1
                    AND segment_id IN (SELECT segment_id FROM review_questions WHERE status IN ('unsure', 'none') AND phone_key IS NULL)
                    AND segment_id NOT IN (SELECT segment_id FROM review_questions WHERE status = 'open')
                    AND segment_id NOT IN (SELECT segment_id FROM seg_reviews)""")
    rows = conn.execute("""
        SELECT e.segment_id, e.uncertain, e.rationale, g.start
        FROM seg_evals e JOIN segments g ON g.id = e.segment_id
        LEFT JOIN seg_reviews r ON r.segment_id = e.segment_id
        LEFT JOIN review_questions q ON q.segment_id = e.segment_id
                                    AND q.status NOT IN ('declined', 'dismissed', 'archived')
                                    AND NOT (q.source = 'sample' AND q.status = 'answered')
        WHERE e.uncertain = 1 AND r.segment_id IS NULL AND q.id IS NULL
        ORDER BY g.start DESC""").fetchall()
    for r in rows:
        rat = r["rationale"] or ""
        src = ("audit" if "daily audit" in rat
               else "propagation" if "your correction" in rat else "self_uncertain")
        # The whole note (it was the last 200 characters, cut mid-word: segment 36561's reason
        # began "r correction of segment 36555 apply here?"). Marks after " | " say what raised the flag.
        out.append({"segment_id": r["segment_id"], "source": src,
                    "reason": " ".join((r["rationale"] or "").split()), "ts": r["start"]})
    # coherence flags -> a boundary question anchored at the CHANGEPOINT: the first segment
    # of the longest stretch whose content differs from the span's opening content (asking at
    # the span's head produces questions about the wrong end).
    for r in conn.execute("""
        SELECT s.start, s.end, s.issue, s.title, s.summary FROM span_summaries s
        WHERE s.coherent = 0 AND s.refined = 1 AND s.start >= ?""", (t0,)).fetchall():
        segs = conn.execute("""SELECT g.id, g.start, e.content,
                                      rv.segment_id AS reviewed, q.id AS questioned
            FROM segments g
            LEFT JOIN seg_evals e ON e.segment_id = g.id
            LEFT JOIN seg_reviews rv ON rv.segment_id = g.id
            LEFT JOIN review_questions q ON q.segment_id = g.id
            WHERE g.start >= ? AND g.start < ? AND g.idle=0 AND g.neutral=0
            ORDER BY g.start""", (r["start"] - 1, r["end"])).fetchall()
        real = [s for s in segs if s["content"] and s["content"] not in ("transition", "idle")]
        # Changepoint = the boundary that best splits the span into two internally-homogeneous
        # halves (duration-weighted): where the second object of attention begins.
        def _dom_share(part):
            from collections import Counter
            c = Counter()
            for s2 in part:
                c[s2["content"]] += max((conn.execute("SELECT COALESCE(end,start)-start FROM segments WHERE id=?",
                                                      (s2["id"],)).fetchone()[0]), 1.0)
            tot = sum(c.values()) or 1.0
            return max(c.values()) / tot, tot
        anchor, best_score = None, -1.0
        for i in range(1, len(real)):
            ls, lw = _dom_share(real[:i])
            rs, rw = _dom_share(real[i:])
            score = (ls * lw + rs * rw) / (lw + rw)
            if score > best_score:
                best_score, anchor = score, real[i]
        if anchor is None and segs:
            anchor = segs[0]
        if anchor is None or anchor["reviewed"] is not None or anchor["questioned"] is not None:
            continue
        from .config import local_dt
        hm = lambda ts: local_dt(ts).strftime("%H:%M")     # the zone the Mac was in then
        hint = hm(anchor["start"])
        # with the span itself: its bounds, title and summary say which stretch was read as one
        # object and what the reviewer took it for (only `issue` was passed on before)
        out.append({"segment_id": anchor["id"], "source": "coherence",
                    "reason": (f"span summary contradicts the one-object definition: {r['issue']} "
                               f"(likely split point: segment id={anchor['id']} at {hint}). The span: {hm(r['start'])}-{hm(r['end'])}, "
                               f"{(r['end'] - r['start']) / 60:.0f} min" + (f", titled “{r['title']}”" if r["title"] else "")
                               + f"; its summary: “{' '.join((r['summary'] or '').split())}”"),
                    "ts": r["start"]})
    # Every switch the legacy ensemble flags for review (stakes x uncertainty) gets an
    # LLM-framed question too - never a raw label prompt.
    from . import stats as _stats
    from .config import day_bounds, logical_date
    today = logical_date()
    for i in range(3):
        d0, d1 = day_bounds(today - dt.timedelta(days=i))
        for sid in _stats.select_for_review(conn, d0, d1, 40):  # priority scoring filters; do not starve it here
            row = conn.execute("""
                SELECT s.to_segment AS seg, s.ts, e.label FROM switches s
                LEFT JOIN ensemble e ON e.switch_id = s.id
                LEFT JOIN seg_reviews rv ON rv.segment_id = s.to_segment
                LEFT JOIN review_questions q ON q.segment_id = s.to_segment
                WHERE s.id = ? AND rv.segment_id IS NULL AND q.id IS NULL""", (sid,)).fetchone()
            if row:
                out.append({"segment_id": row["seg"], "source": "ensemble",
                            "reason": f"the switch models flagged this as high-stakes and uncertain (their call: {row['label']})",
                            "ts": row["ts"]})
    seen: set = set()
    # Low-margin thread links: the evaluator was torn about WHICH thread a segment continues
    # - exactly the ambiguity the person is ground truth for (a canonical thread question).
    for r in conn.execute("""
        SELECT e.segment_id, e.probs, g.start FROM seg_evals e
        JOIN segments g ON g.id = e.segment_id
        LEFT JOIN seg_reviews rv ON rv.segment_id = e.segment_id
        LEFT JOIN review_questions q ON q.segment_id = e.segment_id
        WHERE g.start >= ? AND e.probs IS NOT NULL
          AND e.switch_label IN ('continuation','return')
          AND rv.segment_id IS NULL AND q.id IS NULL
        ORDER BY g.start DESC LIMIT 400""", (time.time() - 2 * 86400,)).fetchall():
        try:
            ap = json.loads(r["probs"]).get("antecedent") or {}
        except ValueError:
            continue
        if ap and ante_top(ap) < 0.6:
            out.append({"segment_id": r["segment_id"], "source": "thread_link",
                        "reason": (f"the evaluator was torn about which thread this continues "
                                   f"(top attachment only {ante_top(ap):.0%} confident)"),
                        "ts": r["start"]})
    out = [t for t in out if not (t["segment_id"] in seen or seen.add(t["segment_id"]))]
    # No thread question without a thread (user rule 2026-11-08): asking whether a segment is
    # part of a thread presupposes enough sustained attention for there to be one. When the
    # thread the segment belongs to runs under the break threshold, it is a contained event
    # inside whatever surrounds it - no answer could move a span - so the doubt is dropped and
    # its needs-review flag cleared. Doubts about WHAT was being done are not thread questions.
    from .config import load_config as _lc
    floor_s = 60.0 * _lc().break_min
    kept = []
    for t in out:
        if t["source"] in ("coherence", "audit") or _content_doubt(conn, t["segment_id"]) or _spot_disagreed(conn, t["segment_id"]):
            kept.append(t)
            continue
        mass = thread_mass_s(conn, t["segment_id"], stop_at=floor_s)
        if mass >= floor_s:
            kept.append(t)
            continue
        log.info("no thread question for segment %d (%s): its thread is %.0fs, under %.0fs",
                 t["segment_id"], t["source"], mass, floor_s)
        decline(conn, t["segment_id"], t["source"], t["reason"],
                f"the thread is {mass:.0f}s, under the {floor_s:.0f}s floor")
    if len(kept) != len(out):
        conn.commit()
    out = kept
    out.sort(key=lambda x: ({"coherence": 0, "propagation": 1, "audit": 2, "ensemble": 3}.get(x["source"], 4), -x["ts"]))
    return out


def ante_top(ap: dict) -> float:
    """The evaluator's confidence in its top thread attachment, with the candidates renormalized
    to sum to 1 (rows stored before 2026-11-05 may carry a lone candidate at 0.2)."""
    tot = sum(ap.values())
    return max(ap.values()) / tot if tot > 0 else 0.0


def thread_mass_s(conn, segment_id: int, max_hops: int = 400, stop_at: float | None = None) -> float:
    """Seconds of attention in the thread a segment belongs to under the current judgment:
    the segment, the chain of antecedents behind it, and everything chained forward from it.
    A person's review of a segment overrides the evaluator's link. With stop_at, the walk ends
    as soon as the total reaches it (callers only ask whether a floor is cleared)."""
    def link(sid):
        r = conn.execute("""SELECT COALESCE(rv.antecedent_id, e.antecedent_id) AS a,
                                     COALESCE(rv.switch_label, e.switch_label) AS sw
                              FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
                              LEFT JOIN seg_reviews rv ON rv.segment_id = g.id WHERE g.id = ?""", (sid,)).fetchone()
        return (r["a"] if r and r["sw"] in ("continuation", "return") else None)
    dur = lambda sid: float(conn.execute("SELECT COALESCE(end, start) - start FROM segments WHERE id=?",
                                         (sid,)).fetchone()[0] or 0.0)
    seen = {segment_id}
    total = dur(segment_id)
    cur, hops = link(segment_id), 0
    while cur is not None and cur not in seen and hops < max_hops:
        seen.add(cur); total += dur(cur); cur = link(cur); hops += 1
        if stop_at is not None and total >= stop_at:
            return total
    frontier = [segment_id]
    while frontier and hops < max_hops and (stop_at is None or total < stop_at):
        nxt = []
        for sid in frontier:
            # Candidates from both indexed columns, then confirmed through link() so a review
            # that re-pointed (or cut) a link wins over the evaluator's.
            cands = {r[0] for r in conn.execute("SELECT segment_id FROM seg_evals WHERE antecedent_id=?", (sid,))}
            cands |= {r[0] for r in conn.execute("SELECT segment_id FROM seg_reviews WHERE antecedent_id=?", (sid,))}
            for cid in sorted(cands):
                if cid not in seen and link(cid) == sid:
                    seen.add(cid); total += dur(cid); nxt.append(cid); hops += 1
        frontier = nxt
    return total


def _content_doubt(conn, segment_id: int) -> bool:
    """The evaluator's doubt is about WHAT the person was doing (content head under 50%),
    which does not presuppose a thread."""
    r = conn.execute("SELECT probs FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    try:
        pr = json.loads(r["probs"]) if r and r["probs"] else {}
        return bool(pr.get("content")) and max(pr["content"].values()) < 0.5
    except (ValueError, AttributeError, TypeError):
        return False


PASSTHROUGH_LABEL = "Neither - just a window I passed through or had open, not actually looked at"


PASSTHROUGH_MIN_P = 0.1   # the evaluator's p(content=transition) below which a long segment was plainly used


def passthrough_applies(conn, segment_id: int) -> bool:
    """Whether the pass-through reading is even possible for a segment. A window held longer
    than the transit threshold (the measurement scheme's definition of a transition) that the
    evaluator gave essentially no chance of being a transition was in use: offering "just
    passed through" there contradicts what the model can see (user, 2026-11-12). Short
    segments, and ones with no evaluation yet, keep the option."""
    cfg = load_config()
    g = conn.execute("SELECT COALESCE(end, start) - start AS secs FROM segments WHERE id=?", (segment_id,)).fetchone()
    if g is None or g["secs"] <= cfg.transit_seconds:
        return True
    row = conn.execute("SELECT probs FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    if row is None or not row["probs"]:
        return True
    try:
        p = (json.loads(row["probs"]).get("content") or {}).get("transition")
    except ValueError:
        return True
    return p is None or p >= PASSTHROUGH_MIN_P


def _spot_disagreed(conn, segment_id: int) -> bool:
    """The person answered 'No' to a spot check on this segment's kind of activity or switch
    (web.py raises the flag and leaves this mark). That doubt is theirs, not the app's, so it
    always gets its question: before 2026-11-20 the short-thread floor dropped it, and the 'No'
    did nothing beyond the tally (86 of 283 eligible segments would have gone that way)."""
    r = conn.execute("SELECT rationale FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    return bool(r and r["rationale"] and SPOT_MARK in r["rationale"])


SPOT_MARK = " | spot check: the person says "
DECLINED_TEXT = "(no question: nothing the person could answer would resolve this doubt)"


def decline(conn, segment_id: int, source: str, reason: str, why: str) -> None:
    """Record that a doubt was looked at and no question was asked, so it is not gathered and
    sent to the model again on the next pass (it used to be, every ~18 s, all night). A
    'declined' row counts as handled wherever a question row does; the queue never shows it,
    and re-evaluating the segment deletes it so the doubt can be raised afresh. The
    needs-review flag goes with it: a doubt nobody can be asked about is not reviewable."""
    conn.execute("INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created, status)"
                 " VALUES (?,?,?,?,'[]',0,?,'declined')",
                 (segment_id, source, reason[:300], DECLINED_TEXT, time.time()))
    conn.execute("UPDATE seg_evals SET uncertain=0, rationale=COALESCE(rationale,'') || ? WHERE segment_id=? AND uncertain=1",
                 (f" | no question: {why}", segment_id))


def _resolved_edit(conn, o: dict) -> tuple:
    """The judgment an option would leave on its segment, blank fields filled from the current
    evaluation (or review), so options are compared by effect."""
    cur = conn.execute("""SELECT COALESCE(r.switch_label, e.switch_label) AS sw,
                                 COALESCE(r.antecedent_id, e.antecedent_id) AS ante,
                                 COALESCE(r.content, e.content) AS content, COALESCE(r.hybrid, e.hybrid) AS hybrid
                          FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
                          LEFT JOIN seg_reviews r ON r.segment_id = g.id WHERE g.id=?""", (o["segment_id"],)).fetchone()
    sw = o["switch_label"] if o["switch_label"] is not None else (cur["sw"] if cur else None)
    ante = (o["antecedent_id"] if o["switch_label"] is not None else (cur["ante"] if cur else None))
    if sw == "interruption":
        ante = None
    content = o["content"] if o["content"] is not None else (cur["content"] if cur else None)
    hybrid = o["hybrid"] if o["hybrid"] is not None else (bool(cur["hybrid"]) if cur and cur["hybrid"] is not None else None)
    # self_distraction as written, not filled in: an option that sets it and one that leaves it
    # alone are different rulings (2026-11-20), and it only lands on a shift or a return.
    sd = o.get("self_distraction") if sw in ("interruption", "return") else None
    return (o["segment_id"], sw, ante, content, hybrid, o.get("focused"), sd,
            (o.get("sub_of") or None) if sw == "interruption" else None)


def passthrough_option(segment_id: int, opts: list[dict]) -> dict:
    """The neutral 'transition' answer every question offers: content=transition, continuation
    of whatever thread preceded (antecedent borrowed from a same-thread option when one exists)."""
    ante = next((o.get("antecedent_id") for o in opts
                 if o.get("segment_id") == segment_id and o.get("antecedent_id") is not None), None)
    return {"label": PASSTHROUGH_LABEL, "segment_id": segment_id,
            "switch_label": "continuation", "antecedent_id": ante,
            "interruption_kind": None, "content": "transition", "hybrid": None}


# Headline metrics an answer can move, with a weight turning each into minute-equivalents.
STAKE_METRICS = {"total_focus_min": ("independent focus", 1.0),
                 "longest_focus_min": ("longest span", 1.0),
                 "longest_absorbed_min": ("longest fully absorbed", 1.0),
                 "self_distraction_events": ("self-distraction events", 10.0)}
PRIORITY_CUT = 4.0   # minute-equivalents x uncertainty; at or above = the priority tier


def _uncertainty(conn, segment_id: int, source: str) -> float:
    """How unsure the model is about the segment the question is about: 1 - its confidence in
    its top content and switch calls (the lower of the two). Sources that do not come from the
    model's own probabilities (a contradiction, a correction, the audit) are at least 0.7."""
    row = conn.execute("SELECT probs FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    u = 0.5
    if row and row["probs"]:
        try:
            pr = json.loads(row["probs"])
            tops = [max(pr[k].values()) for k in ("content", "switch") if pr.get(k)]
            if source == "thread_link" and pr.get("antecedent"):
                tops = [ante_top(pr["antecedent"])]
            if tops:
                u = 1.0 - min(tops)
        except (ValueError, AttributeError, TypeError):
            pass
    if source in ("coherence", "propagation", "audit"):
        u = max(u, 0.7)
    if source in ("phone", "phone_content"):   # the model's own confidence is about the computer segment, not the phone
        u = 0.8
    return max(0.0, min(1.0, u))


def _day_stamp(conn, t0: float, t1: float) -> float:
    """When the day's judgments last changed: the newest evaluator row or user review in it."""
    r = conn.execute("""SELECT MAX(x) FROM (
        SELECT MAX(e.created) AS x FROM seg_evals e JOIN segments g ON g.id=e.segment_id WHERE g.start>=? AND g.start<?
        UNION ALL
        SELECT MAX(r.created) FROM seg_reviews r JOIN segments g ON g.id=r.segment_id WHERE g.start>=? AND g.start<?)""",
                     (t0, t1, t0, t1)).fetchone()
    return float(r[0] or 0.0)


def score_questions(conn, limit: int | None = None, only_stale: bool = True) -> int:
    """Compute stakes x uncertainty for open questions whose score is missing or older than the
    day's latest change. Stakes = for each headline metric, the spread of its value across
    'answer option k applied' (plus the status quo), in minute-equivalents; the max over metrics.
    No model calls - a handful of in-memory metric recomputes per question (~15 ms each)."""
    from . import stats
    from .config import day_bounds, logical_date
    rows = conn.execute("""SELECT q.id, q.segment_id, q.source, q.options, q.scored_at, g.start
                           FROM review_questions q JOIN segments g ON g.id = q.segment_id
                           WHERE q.status = 'open' ORDER BY g.start DESC""").fetchall()
    base_cache: dict = {}
    stamp_cache: dict = {}
    n = 0
    for r in rows:
        day = logical_date(r["start"])
        if day not in stamp_cache:
            stamp_cache[day] = _day_stamp(conn, *day_bounds(day))
        if only_stale and r["scored_at"] and r["scored_at"] >= stamp_cache[day]:
            continue
        if limit is not None and n >= limit:
            break
        try:
            opts = json.loads(r["options"])
        except ValueError:
            opts = []
        if day not in base_cache:
            base_cache[day] = stats.daily_metrics(conn, day)
        vals = {k: [base_cache[day][k]] for k in STAKE_METRICS}
        for o in opts:
            if o.get("phone"):      # a ruling on a phone segment (phone_questions)
                m = stats.daily_metrics(conn, day, {"phone": {o["phone_key"]: o["phone"]}})
                for k in STAKE_METRICS:
                    vals[k].append(m[k])
                continue
            sid = o.get("segment_id") or r["segment_id"]
            ov = {sid: {k: o.get(k) for k in ("content", "switch_label", "interruption_kind", "antecedent_id", "hybrid", "focused", "self_distraction")}}
            if not any(v is not None for v in ov[sid].values()):
                continue
            m = stats.daily_metrics(conn, day, ov)
            for k in STAKE_METRICS:
                vals[k].append(m[k])
        detail = {k: round((max(v) - min(v)), 2) for k, v in vals.items()}
        stakes = max(detail[k] * STAKE_METRICS[k][1] for k in STAKE_METRICS)
        unc = _uncertainty(conn, r["segment_id"], r["source"])
        # Uncertainty modulates rather than zeroes: a sure-looking call that would move the day
        # by an hour still deserves a look before a coin-flip that moves nothing.
        prio = stakes * (0.25 + 0.75 * unc)
        conn.execute("UPDATE review_questions SET stakes=?, stakes_detail=?, uncertainty=?, priority=?, scored_at=?"
                     " WHERE id=?", (stakes, json.dumps(detail), unc, prio, time.time(), r["id"]))
        n += 1
    if n:
        conn.commit()
    return n


def stakes_text(detail_json: str | None) -> str:
    """'could move independent focus by 23 min, bounce-aways by 1' - the biggest movers only."""
    try:
        d = json.loads(detail_json or "{}")
    except ValueError:
        return ""
    parts = []
    for k, (name, w) in STAKE_METRICS.items():
        # stakes scored before 2026-11-19 stored the all-day count under its old name
        v = d.get(k, d.get("bounce_aways", 0) if k == "self_distraction_events" else 0) or 0
        if k == "self_distraction_events" and v >= 1:
            parts.append((v * w, f"{name} by {v:.0f}"))
        elif k != "self_distraction_events" and v >= 1:
            parts.append((v * w, f"{name} by {v:.0f} min"))
    parts.sort(reverse=True)
    return ("could move " + ", ".join(t for _, t in parts[:2])) if parts else "no effect on the day's metrics"


_last_sweep = 0.0
SWEEP_EVERY_S = 1800.0
_last_archive = 0.0
ARCHIVED_MARK = " | review question archived unanswered"


def archive_stale_questions(conn, cfg: Config | None = None, force: bool = False) -> int:
    """Archive open questions nobody answered within cfg.question_archive_days of being asked
    (user ruling 2026-11-14: past three days they will not be got to, and ~1,500 of them were
    making the queue and its scoring slow). Nothing is deleted: the row keeps everything and
    gets status 'archived' + archived_at. The segment's needs-review flag is cleared, as it is
    for a dismissed question, so the timeline stops showing a dot with nothing behind it; the
    marker appended to the rationale says which flags went this way. The model's judgment on
    the segment stands. To bring one back: status='open', archived_at=NULL, seg_evals.uncertain=1
    - and a `created` inside the window, or the next pass archives it again."""
    global _last_archive
    if not force and time.time() - _last_archive < SWEEP_EVERY_S:
        return 0
    _last_archive = time.time()
    days = (cfg or load_config()).question_archive_days
    if not days or days <= 0:
        return 0
    now = time.time()
    with db.tx(conn):
        rows = conn.execute("SELECT id, segment_id FROM review_questions WHERE status='open' AND created < ?",
                            (now - days * 86400,)).fetchall()
        for r in rows:
            conn.execute("UPDATE review_questions SET status='archived', archived_at=? WHERE id=?", (now, r["id"]))
            conn.execute("UPDATE seg_evals SET uncertain=0, rationale=COALESCE(rationale,'') || ? WHERE segment_id=? AND uncertain=1",
                         (ARCHIVED_MARK, r["segment_id"]))
    if rows:
        log.info("archived %d review question(s) left unanswered for %.3g days", len(rows), days)
    return len(rows)


def retire_threadless_questions(conn, force: bool = False) -> int:
    """Apply the no-thread-without-a-thread rule to questions already in the queue: an OPEN
    thread question whose thread runs under the break threshold is dismissed (its flag cleared),
    with the reason on the question so the record says why it went."""
    global _last_sweep
    if not force and time.time() - _last_sweep < SWEEP_EVERY_S:
        return 0
    _last_sweep = time.time()
    from .config import load_config as _lc
    floor_s = 60.0 * _lc().break_min
    n = 0
    rows = conn.execute("""SELECT id, segment_id, source FROM review_questions
                           WHERE status = 'open' AND source IN ('self_uncertain', 'propagation', 'thread_link', 'ensemble')""").fetchall()
    for r in rows:
        if _content_doubt(conn, r["segment_id"]):
            continue
        mass = thread_mass_s(conn, r["segment_id"], stop_at=floor_s)
        if mass >= floor_s:
            continue
        with db.tx(conn):
            conn.execute("UPDATE review_questions SET status='dismissed', answered_at=?, note=? WHERE id=?",
                         (time.time(), f"(retired: no thread question without a thread - this one is {mass:.0f}s)", r["id"]))
            conn.execute("UPDATE seg_evals SET uncertain=0 WHERE segment_id=?", (r["segment_id"],))
        n += 1
    if n:
        log.info("retired %d open thread question(s) on threads under %.0fs", n, floor_s)
    return n


# Phone judgments are low confidence (the model sees an app name and timing, nothing else) and
# the phone is secondary, so they are not asked about - unless one is load-bearing (Eli,
# 2026-11-19: "Make a review question if it's loadbearing for some important evaluation").
# Load-bearing here = reading the phone time the other way (away from the work <-> part of it)
# moves the day's independent focus time or its longest span by this many minutes or more.
PHONE_Q_MIN = 10.0
# ...and only for phone use long enough for there to be a real other reading: a 2-second
# pass-through "read the other way" is a 2-second shift, which can still unmake a span's opening
# 10 unbroken minutes - the first live run asked about two of those (2026-11-19).
PHONE_Q_MIN_S = 30.0
PHONE_Q_NEAR_S = 2 * 3600.0   # no question while computer time this close to the phone use is still unjudged


def _phone_call_text(p) -> str:
    return ("a self-distraction" if p["away"] and p["self_distraction"] else
            "a deliberate trip away from the work" if p["away"] else "part of what you were doing")


def _phone_effect_text(k: str, delta: float) -> str:
    what = "independent focus time" if k == "total_focus_min" else "longest focus span"
    return f" Read the other way, the day's {what} would be {abs(delta):.0f} min {'longer' if delta > 0 else 'shorter'}."
PHONE_OPTIONS = (
    ("Part of what I was doing on the computer", {"switch_label": "continuation", "away": 0, "focused": 1, "self_distraction": 0}),
    ("A deliberate break or errand", {"switch_label": "interruption", "away": 1, "self_distraction": 0}),
    ("A self-distraction: I picked it up without a clear task", {"switch_label": "interruption", "away": 1, "self_distraction": 1}),
    ("The phone was just open; I was not looking at it", {"switch_label": "continuation", "content": "transition", "away": 0, "self_distraction": 0}),
)
_phone_checked: dict = {}   # seg_key -> the day stamp it was last checked at (in memory: cheap to redo)

# What was the phone used FOR (Eli, 2026-11-21: "We should be asking phone questions, because we
# want to surface the rules about what phone usage is passive consumption and which isn't"). The
# question above asks only when an answer would move the day's focus numbers, and only about
# away-or-not. This one asks about the content, whatever the stakes, so that the person's
# answers build up per app into precedents (the evaluator sees them with its rulings) and into
# rules (rule_drafts reads phone rulings too). One open question per phone app at a time, and an
# app stops being asked about once PHONE_RULE_ANSWERS answers about it are on record: the aim is
# the rule, not a ruling on every pickup.
PHONE_RULE_ANSWERS = 4
PHONE_CONTENT_OPTIONS = (
    ("Passive: scrolling, watching or browsing, nothing in particular", {"content": "passive", "focused": 0}),
    ("Messaging: reading or replying to someone", {"content": "correspondence"}),
    ("A specific errand or task (ordering, booking, logging, a setting)", {"content": "task", "focused": 1}),
    ("Reading or looking up something specific", {"content": "reading", "focused": 1}),
    ("The phone was just open; I was not looking at it", {"switch_label": "continuation", "content": "transition", "away": 0, "self_distraction": 0}),
)


def phone_content_questions(conn, lookback_s: float = 7 * 86400) -> int:
    """Ask what a phone app was being used for (see PHONE_CONTENT_OPTIONS), for each phone app with
    fewer than PHONE_RULE_ANSWERS answers on record and no question open: about its longest recent
    use of PHONE_Q_MIN_S or more. No model call: the options are the same for every app."""
    from .config import local_dt
    try:
        answered = {r[0]: r[1] for r in conn.execute(
            """SELECT p.app, COUNT(*) FROM phone_reviews r JOIN phone_evals p ON p.seg_key = r.seg_key
               WHERE r.content IS NOT NULL GROUP BY p.app""")}
        open_apps = {r[0] for r in conn.execute(
            """SELECT p.app FROM review_questions q JOIN phone_evals p ON p.seg_key = q.phone_key
               WHERE q.status = 'open'""")}
        rows = conn.execute("""SELECT p.* FROM phone_evals p
            LEFT JOIN phone_reviews r ON r.seg_key = p.seg_key
            LEFT JOIN review_questions q ON q.phone_key = p.seg_key
            WHERE p.switch_label IS NOT NULL AND p.app != 'Lock screen' AND p.start >= ?
              AND p.end - p.start >= ? AND COALESCE(p.content, '') != 'transition'
              AND r.seg_key IS NULL AND q.id IS NULL
            ORDER BY p.end - p.start DESC""", (time.time() - lookback_s, PHONE_Q_MIN_S)).fetchall()
    except Exception:   # no phone tables yet
        return 0
    n = 0
    for p in rows:
        app = p["app"] or ""
        if not app or app in open_apps or answered.get(app, 0) >= PHONE_RULE_ANSWERS:
            continue
        g = conn.execute("""SELECT id, app, title FROM segments WHERE start <= ? AND COALESCE(end, start) > ?
                            ORDER BY idle, start DESC LIMIT 1""", (p["start"], p["start"])).fetchone()
        if g is None:
            continue
        when = local_dt(p["start"]).strftime("%a %H:%M")
        secs = p["end"] - p["start"]
        length = f"{secs / 60:.0f} min" if secs >= 90 else f"{secs:.0f} s"
        ctx = (f"On {when} your phone had {app} open for {length}"
               f"{', while the computer sat idle' if g['app'] == '__idle__' else ', while the computer was on ' + g['app']}. "
               f"The app sees only which phone app was open and when; it guessed: {p['content'] or 'unknown'}. "
               f"This is not about whether it was a break or a distraction, only what the use was: your answers "
               f"about {app} become precedents for judging it, and once a pattern holds, a rule for your rules "
               f"document ({answered.get(app, 0)} of {PHONE_RULE_ANSWERS} answers about {app} so far).")
        opts = [{"label": lab, "segment_id": g["id"], "phone_key": p["seg_key"], "phone": edit,
                 "switch_label": None, "antecedent_id": None, "interruption_kind": None, "content": None, "hybrid": None}
                for lab, edit in PHONE_CONTENT_OPTIONS]
        conn.execute("INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created, phone_key)"
                     " VALUES (?,?,?,?,?,0,?,?)",
                     (g["id"], "phone_content", ctx, f"What were you doing in {app} on your phone?",
                      json.dumps(opts), time.time(), p["seg_key"]))
        open_apps.add(app)
        n += 1
    if n:
        conn.commit()
        log.info("wrote %d question(s) about what phone apps were used for", n)
    return n


def phone_questions(conn, lookback_s: float = 3 * 86400) -> int:
    """Write a review question for each load-bearing phone judgment (see PHONE_Q_MIN). No model
    call: the model knows nothing more about the phone than the app and the timing, so the
    question is a fixed one, anchored on the computer segment the phone time falls in (which
    gives the card its timeline, with the phone lane under it)."""
    from . import stats
    from .config import logical_date, day_bounds
    try:
        rows = conn.execute("""SELECT p.* FROM phone_evals p
            LEFT JOIN phone_reviews r ON r.seg_key = p.seg_key
            LEFT JOIN review_questions q ON q.phone_key = p.seg_key
            WHERE p.switch_label IS NOT NULL AND p.app != 'Lock screen' AND p.start >= ?
              AND p.end - p.start >= ? AND COALESCE(p.content, '') != 'transition'
              AND r.seg_key IS NULL AND q.id IS NULL ORDER BY p.start""", (time.time() - lookback_s, PHONE_Q_MIN_S)).fetchall()
    except Exception:   # no phone tables yet
        return 0
    n, base, stamps = 0, {}, {}

    def stamp(day):
        if day not in stamps:
            d0, d1 = day_bounds(day)
            ps = conn.execute("SELECT MAX(created) FROM phone_evals WHERE start >= ? AND start < ?", (d0, d1)).fetchone()[0]
            pr = conn.execute("SELECT MAX(r.created) FROM phone_reviews r JOIN phone_evals p ON p.seg_key = r.seg_key"
                              " WHERE p.start >= ? AND p.start < ?", (d0, d1)).fetchone()[0]
            stamps[day] = max(_day_stamp(conn, d0, d1), float(ps or 0.0), float(pr or 0.0))
        return stamps[day]

    def effect(p):
        """(metric, minutes) the day moves by when this phone time is read the other way."""
        day = logical_date(p["start"])
        flip = ({"switch_label": "continuation", "away": 0, "focused": 1, "self_distraction": 0} if p["away"]
                else {"switch_label": "interruption", "away": 1, "self_distraction": 0})
        if day not in base:
            base[day] = stats.daily_metrics(conn, day)
        m = stats.daily_metrics(conn, day, {"phone": {p["seg_key"]: flip}})
        moved = {k: m[k] - base[day][k] for k in ("total_focus_min", "longest_focus_min")}
        return max(moved.items(), key=lambda kv: abs(kv[1]))

    # the evaluator only judges the current and the previous day (run_pending): a segment left
    # unjudged from before that never will be, and must not hold a question back for good
    still_judged = day_bounds(logical_date() - dt.timedelta(days=1))[0]

    def unjudged_near(p) -> bool:
        """Computer time around the phone use that the evaluator has not judged yet. The effect
        is not known until it has: on 2026-11-23 a question said "26 min longer" about a 76 s
        Oura check, because the phone data arrived (and was judged) a minute before the 40
        minutes of computer work around it were; once they were judged it moved 1 minute."""
        return conn.execute("""SELECT 1 FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
            WHERE g.idle = 0 AND g.neutral = 0 AND e.segment_id IS NULL AND g.end - g.start >= 5   -- zero-length ones are never judged
              AND g.start < ? AND g.end > ? AND g.start > ? LIMIT 1""",
                            (p["end"] + PHONE_Q_NEAR_S, p["start"] - PHONE_Q_NEAR_S,
                             max(p["start"] - PHONE_Q_NEAR_S - 86400, still_judged))).fetchone() is not None

    # Open questions first: the figure in each was right when it was written and the day has
    # since changed. One that no longer clears the bar is archived; the others get today's figure.
    n_gone = 0
    for q in conn.execute("""SELECT q.id, q.context, p.* FROM review_questions q JOIN phone_evals p ON p.seg_key = q.phone_key
                             WHERE q.source = 'phone' AND q.status = 'open' AND p.start >= ?""", (time.time() - lookback_s,)).fetchall():
        key = ("q", q["id"])
        st = stamp(logical_date(q["start"]))
        if _phone_checked.get(key) == st or unjudged_near(q):
            continue
        _phone_checked[key] = st
        k, delta = effect(q)
        if abs(delta) < PHONE_Q_MIN:
            conn.execute("UPDATE review_questions SET status='archived', archived_at=? WHERE id=?", (time.time(), q["id"]))
            n_gone += 1
            continue
        mark = "so its call is a low-confidence one: "
        head = q["context"].split(" Read the other way,")[0]
        if mark in head:   # the call itself may have been re-judged since
            head = head.split(mark)[0] + mark + _phone_call_text(q) + "."
        ctx = head + _phone_effect_text(k, delta)
        if ctx != q["context"]:
            conn.execute("UPDATE review_questions SET context=? WHERE id=?", (ctx, q["id"]))
    if n_gone:
        conn.commit()
        log.info("archived %d phone question(s) that no longer move the day by %g minutes", n_gone, PHONE_Q_MIN)

    for p in rows:
        day = logical_date(p["start"])
        if _phone_checked.get(p["seg_key"]) == stamp(day):
            continue
        if unjudged_near(p):
            continue   # asked about once the computer time around it is judged (not cached: tried again next pass)
        _phone_checked[p["seg_key"]] = stamp(day)
        k, delta = effect(p)
        if abs(delta) < PHONE_Q_MIN:
            continue
        g = conn.execute("""SELECT id, app, title FROM segments WHERE start <= ? AND COALESCE(end, start) > ?
                            ORDER BY idle, start DESC LIMIT 1""", (p["start"], p["start"])).fetchone()
        if g is None:
            continue
        from .config import local_dt
        hhmm = local_dt(p["start"]).strftime("%H:%M")
        secs = p["end"] - p["start"]
        length = f"{secs / 60:.0f} min" if secs >= 90 else f"{secs:.0f} s"
        call = _phone_call_text(p)
        ctx = (f"At {hhmm} your phone had {p['app']} open for {length}, while the computer was on {g['app']}"
               f"{' (' + g['title'][:60] + ')' if g['title'] else ''}. The app sees only which phone app was open and when, "
               f"so its call is a low-confidence one: {call}." + _phone_effect_text(k, delta))
        opts = [{"label": lab, "segment_id": g["id"], "phone_key": p["seg_key"], "phone": edit,
                 "switch_label": None, "antecedent_id": None, "interruption_kind": None, "content": None, "hybrid": None}
                for lab, edit in PHONE_OPTIONS]
        conn.execute("INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created, phone_key)"
                     " VALUES (?,?,?,?,?,0,?,?)",
                     (g["id"], "phone", ctx, f"What was the {p['app']} use on your phone at {hhmm}?", json.dumps(opts),
                      time.time(), p["seg_key"]))
        n += 1
    conn.commit()
    if n:
        log.info("wrote %d question(s) about load-bearing phone judgments", n)
    return n


SPOT_GAP_S = 2 * 3600.0     # at most one new spot check per this long, so they spread over the day
SPOT_MIN_S = 120.0          # shorter segments are not drawn (Eli, 2026-11-22: no spot checks under 2 minutes; it was 20 s.
                            #  A short segment can still get an ordinary review question when its judgment matters)
SPOT_SETTLE_S = 600.0       # ...nor ones that ended less than this ago (their judgment may still change)


SPOT_AXES = ("content", "focused", "switch")
_SPOT_SWITCH = {"continuation": "a continuation of what came before", "interruption": "a shift to something new",
                "return": "a return to something from earlier"}


SPOT_SYSTEM = """You write the short context shown above a review question in one person's \
focus-tracking app. The app judged a stretch of their computer activity, and the person is about \
to be asked whether one part of that judgment is right. Your summary reminds them what was \
happening, so they can answer from memory.

Write 2-3 plain sentences in the second person ("You ..."), with clock times: what they were \
doing in the stretch before, what they did in the marked segment, and what came next if it helps. \
Describe the ACTIVITY, not the windows: the undertaking, inferred from titles, the screenshot and \
the notes after "// note:" in the log ("you were editing your personal site"), not a list of apps. Use \
"apparently" or "likely" where you are inferring.

Do NOT say or hint at the app's verdict: no "focused", "unfocused", "distracted", "drifting", \
"continuing", "shifted", "returned", and no content-type label. The person must judge that \
themselves. No question, no preamble; only the summary."""


class SpotContext(BaseModel):
    context: str = Field(description="2-3 sentences, second person, with clock times; no verdict words.")


SPOT_IMAGES = 8          # screenshots of the spot-checked segment
DOUBT_IMAGES = 60        # screenshots in one question-writer call (evaluator.IMAGE_BUDGET), shared by its doubts
OTHER_LISTS = 40         # "could not answer" / "none of the above" questions shown, newest first (12 before)
MAX_TOKENS = 16000       # up to 12 questions with options, and thinking counts against it (4,000 before)


def _spot_summary(conn, cfg: Config, g) -> tuple[str | None, float]:
    """One small model call: what was happening around a spot-checked segment, written the way the
    other review questions set up theirs (Eli, 2026-11-19). Returns (summary or None, cost)."""
    from . import shots as _shots, stats, summaries
    from .classifiers.claude_vision import _b64
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return None, 0.0
    # 2026-11-20: every chosen capture moment of the segment (it was the first frame only, of a
    # segment that can run for minutes), the evaluator's full log lines (notes were cut at 90
    # characters), from 20 minutes before to 10 after its END.
    from . import prompt_parts as pp
    mom = _shots.moments(conn, [g["id"]])
    blocks, meta = pp.image_blocks(pp.fit(pp.seg_images(mom, g, "the marked segment,", both=False), SPOT_IMAGES))
    lines, _ = pp.log_lines(conn, cfg, g["start"] - 1200, g["end"] + 600, marks={g["id"]: f">>> (the marked segment, id={g['id']}) "},
                            default_mark="    (id={id}) ", dated=True)
    blocks.append({"type": "text", "text": (f"Screenshots above: {len(meta)}, all of the marked segment.\n" if meta else
                                            f"No screenshot of the marked segment ({_shots.why_none(conn, g['id'])}).\n")
                   + "Activity log (>>> marks the segment the question is about; the tags after \"judged:\" and the notes "
                   "are the app's own, and its verdict on the marked segment is what you must not give away):\n" + "\n".join(lines)})
    try:
        resp = llm.parse(_client(), site="spot_check", model=cfg.claude_model, max_tokens=4000,   # thinking counts against it (600 before)
                         system=[{"type": "text", "text": SPOT_SYSTEM + "\n\n" + pp.LOG_LEGEND}],
                         messages=[{"role": "user", "content": blocks}],
                         output_format=SpotContext, output_config={"effort": cfg.claude_effort},   # "low" was written in
                         betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except Exception:
        log.exception("spot-check summary failed (segment %d)", g["id"])
        return None, 0.0
    cost = float(getattr(resp, "cost_usd", 0.0) or 0.0) or llm.call_cost(resp.usage)
    out = resp.parsed_output.context.strip() if resp.parsed_output is not None else ""
    return (out or None), cost


def _spot_question(g, axis: str, summary: str | None = None) -> tuple[str, str, list[dict]]:
    """The wording of one spot check. It asks about ONE axis of the judgment (Eli, 2026-11-19:
    a question is only ever about one axis), and 'Yes' rules on that axis alone. The context is
    the model-written summary of what was happening; without one, a bare where-and-when line."""
    from .config import local_dt
    hhmm = local_dt(g["start"]).strftime("%H:%M")     # the zone the Mac was in then
    secs = g["end"] - g["start"]
    length = f"{secs / 60:.0f} min" if secs >= 90 else f"{secs:.0f} s"
    where = g["app"] + (f" · {g['domain']}" if g["domain"] else "") + (f" ({g['title'][:60]})" if g["title"] else "")
    ctx = summary or f"At {hhmm}, {where}, for {length}."
    blank = {"segment_id": g["id"], "switch_label": None, "antecedent_id": None, "interruption_kind": None,
             "content": None, "hybrid": None, "axis": axis, "summarised": bool(summary)}
    yes = dict(blank, label="Yes", sample_verdict="agree")
    no = dict(blank, label="No", sample_verdict="disagree")
    if axis == "content":
        question = f"Spot check: the app judged the kind of activity in the {length} from {hhmm} as: {g['content']}. Is that right?"
        yes["content"] = g["content"]
    elif axis == "focused":
        f = bool(g["focused"])
        question = f"Spot check: the app judged that you were {'focused' if f else 'unfocused'} for the {length} from {hhmm}. Is that right?"
        yes.update(focused=f, label=f"Yes, {'focused' if f else 'unfocused'}")
        # two-valued, so 'No' is itself the ruling and no further question is needed
        no.update(focused=not f, label=f"No, {'unfocused' if f else 'focused'}")
    else:
        question = f"Spot check: the app judged the {length} from {hhmm} as {_SPOT_SWITCH.get(g['switch_label'], g['switch_label'])}. Is that right?"
        yes.update(switch_label=g["switch_label"],
                   antecedent_id=g["antecedent_id"] if g["switch_label"] in ("continuation", "return") else None)
    return ctx, question, [yes, no]


def _refresh_open_spot_checks(conn, cfg: Config) -> None:
    """Rewrite open spot checks written before 2026-11-19: ones that asked about the whole
    judgment get one axis, and ones without a summary of what was happening get one."""
    import random
    for q in conn.execute("SELECT id, segment_id, options FROM review_questions WHERE source='sample' AND status='open'").fetchall():
        try:
            old = json.loads(q["options"] or "[]")[0]
        except (ValueError, IndexError):
            old = {}
        # Has its summary: done. Tried and failed (network down, budget spent): try again, but
        # not more than every half hour. (Until 2026-11-20 a failed one was never retried and
        # kept its bare where-and-when line for good.)
        if old.get("summarised") or (old.get("summarised") is False and time.time() - (old.get("summary_tried") or 0) < 1800):
            continue
        g = conn.execute("""SELECT g.id, g.start, COALESCE(g.end, g.start) AS end, g.app, g.domain, g.title,
                                   e.content, e.switch_label, e.focused, e.antecedent_id, e.rationale
                            FROM segments g JOIN seg_evals e ON e.segment_id = g.id WHERE g.id=?""", (q["segment_id"],)).fetchone()
        if g is None or g["focused"] is None:
            continue
        axis = old.get("axis") or random.choice([a for a in SPOT_AXES if a != "switch" or g["switch_label"]])
        summary, cost = _spot_summary(conn, cfg, g)
        ctx, question, opts = _spot_question(g, axis, summary)
        opts[0]["summary_tried"] = time.time()
        with db.tx(conn):
            conn.execute("UPDATE review_questions SET context=?, question=?, options=?, cost_usd=cost_usd+? WHERE id=?",
                         (ctx, question, json.dumps(opts), cost, q["id"]))
            conn.execute("UPDATE agreement_samples SET axis=? WHERE segment_id=?", (axis, q["segment_id"]))


def spot_checks(conn, cfg: Config | None = None) -> int:
    """Draw today's next spot check, if one is due: a segment whose judgment the evaluator was
    confident of (no doubt flag, no review, no question), picked at random with a chance
    proportional to its length, so the weekly figure reads as "share of judged time the app had
    right". Each check asks about one axis of the judgment (content, focused, or the switch),
    also drawn at random. The question is fixed; one small model call writes the summary of what
    was happening that sits above it. 'Yes' stores that axis as the
    person's own ruling; 'No' on focused is the opposite ruling; 'No' on the other two flags the
    segment, and the usual question about what it really was follows."""
    import random
    from .config import day_bounds, logical_date
    cfg = cfg or load_config()
    # open spot checks drawn under the old 20-second floor are archived, not left in the queue
    with db.tx(conn):
        k = conn.execute("""UPDATE review_questions SET status='archived', archived_at=? WHERE source='sample' AND status='open'
                            AND segment_id IN (SELECT id FROM segments WHERE COALESCE(end, start) - start < ?)""",
                         (time.time(), SPOT_MIN_S)).rowcount
    if k:
        log.info("archived %d open spot check(s) on segments under %d s", k, SPOT_MIN_S)
    _refresh_open_spot_checks(conn, cfg)
    n = cfg.spot_checks_per_day
    if not n or n <= 0:
        return 0
    day = logical_date()
    have = conn.execute("SELECT COUNT(*), MAX(created) FROM agreement_samples WHERE day=?", (day.isoformat(),)).fetchone()
    now = time.time()
    if have[0] >= n or (have[1] and now - have[1] < SPOT_GAP_S):
        return 0
    d0, d1 = day_bounds(day)
    rows = conn.execute("""
        SELECT g.id, g.start, COALESCE(g.end, g.start) AS end, g.app, g.domain, g.title,
               e.content, e.switch_label, e.focused, e.self_distraction, e.next_task, e.antecedent_id, e.rationale
        FROM segments g JOIN seg_evals e ON e.segment_id = g.id
        LEFT JOIN seg_reviews r ON r.segment_id = g.id
        LEFT JOIN review_questions q ON q.segment_id = g.id
        LEFT JOIN agreement_samples a ON a.segment_id = g.id
        WHERE g.start >= ? AND g.start < ? AND g.idle = 0 AND g.neutral = 0 AND e.uncertain = 0
          AND e.focused IS NOT NULL AND e.content NOT IN ('transition', 'idle')
          AND r.segment_id IS NULL AND q.id IS NULL AND a.segment_id IS NULL
          AND COALESCE(g.end, g.start) - g.start >= ? AND COALESCE(g.end, g.start) < ?""",
                        (d0, d1, SPOT_MIN_S, now - SPOT_SETTLE_S)).fetchall()
    if not rows:
        return 0
    g = random.choices(rows, weights=[r["end"] - r["start"] for r in rows])[0]
    axis = random.choice([a for a in SPOT_AXES if a != "switch" or g["switch_label"]])
    summary, cost = _spot_summary(conn, cfg, g)
    ctx, question, opts = _spot_question(g, axis, summary)
    with db.tx(conn):
        cur = conn.execute("INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created)"
                           " VALUES (?,?,?,?,?,?,?)",
                           (g["id"], "sample", ctx, question, json.dumps(opts), cost, now))
        conn.execute("INSERT INTO agreement_samples(segment_id, day, question_id, created, axis) VALUES (?,?,?,?,?)",
                     (g["id"], day.isoformat(), cur.lastrowid, now, axis))
    hhmm = dt.datetime.fromtimestamp(g["start"]).strftime("%H:%M")
    log.info("spot check drawn: segment %d (%s), on %s", g["id"], hhmm, axis)
    return 1


def _doubt_parts(conn, cfg: Config, trigs: list[dict]) -> list[tuple[list, str]]:
    """Per doubt: (image blocks, text). What the question writer is shown, since 2026-11-20
    (Eli: every model call gets all the information that could be relevant). Before, a doubt
    was one screenshot (the first frame; 359 of 559 doubted segments had more capture moments,
    up to 50, and the segment the question compares it with had none), the last 200 characters
    of the reason, and 20 minutes each side of the segment's START in the short line form with
    notes cut at 90 characters, no thread pointers and nothing marking the person's own rulings
    (33 of 559 windows held one). The evaluator's alternatives (seg_evals.probs, stored for 550
    of 559) were not shown: a thread_link doubt said "top attachment only N% confident" and
    did not name the candidates."""
    from . import chat_tools as ct, evaluator as ev, prompt_parts as pp, shots as _shots
    from .summaries import verdict_text
    per = max(4, DOUBT_IMAGES // max(len(trigs), 1))
    out = []
    for i, t in enumerate(trigs):
        sid = t["segment_id"]
        g = conn.execute("SELECT * FROM segments WHERE id=?", (sid,)).fetchone()
        end = g["end"] or g["start"]
        e = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (sid,)).fetchone()
        # the segments it is compared with: the evaluator's candidates for what it continues, and the segment before it
        try:
            ap = (json.loads(e["probs"]) if e is not None and e["probs"] else {}).get("antecedent") or {}
        except ValueError:
            ap = {}
        others = [int(k) for k, _ in sorted(ap.items(), key=lambda x: -x[1]) if str(k).isdigit()][:2]
        if e is not None and e["antecedent_id"] and e["antecedent_id"] not in others:
            others.insert(0, e["antecedent_id"])
        prev = pp._prev_id(conn, sid)
        if prev and prev not in others:
            others.append(prev)
        others = [o for o in others if o != sid][:3]
        orows = [r for r in (conn.execute("SELECT * FROM segments WHERE id=?", (o,)).fetchone() for o in others) if r is not None and not r["idle"]]
        mom = _shots.moments(conn, [sid] + [r["id"] for r in orows])
        cmp_imgs = [im for r in orows for im in pp.seg_images(mom, r, f"DOUBT {i + 1}: a segment it may continue,", max_moments=1, both=False)]
        own = pp.seg_images(mom, g, f"DOUBT {i + 1}: THE SEGMENT IN DOUBT,")
        if len(own) > per - len(cmp_imgs):
            own = pp.fit(pp.seg_images(mom, g, f"DOUBT {i + 1}: THE SEGMENT IN DOUBT,", both=False), max(1, per - len(cmp_imgs)))
        blocks, meta = pp.image_blocks(own + cmp_imgs)
        lines, _ = pp.log_lines(conn, cfg, g["start"] - 1200, end + 1200,     # the after-window from its END: 4 doubted segments ran past start + 20 min
                                marks={sid: f">>> id={sid} (THE SEGMENT IN DOUBT) "}, default_mark="    id={id} ", dated=True)
        cur = ("current judgment: " + pp.judged_text(pp.judged_before(conn, sid, time.time()))) if e is not None else "(no judgment)"
        why = [f"source: {t['source']}; {t['reason']}"]
        if t["source"] == "propagation":
            # the ruling that raised it: what the person ruled on which segment, and the judge's note
            # (from the record of carried-over corrections, and from the mark in the note: flags
            # raised before that record existed, 2026-11-20, are only in the note)
            import re
            notes = {p["from_segment_id"]: p["note"] for p in conn.execute(
                "SELECT * FROM seg_propagated WHERE segment_id=? AND verdict='uncertain' ORDER BY created", (sid,))}
            for src in re.findall(r"does your correction of segment (\d+) apply here\?", t["reason"]):
                notes.setdefault(int(src), None)
            for src, pnote in notes.items():
                rv = conn.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (src,)).fetchone()
                why.append(f"The doubt was raised by the person's ruling on segment id={src}: the app's propagation judge "
                           "thought it might apply here too" + (f" (“{pp.one(pnote)}”)" if pnote else "") + ". That ruling: "
                           + ((verdict_text(rv) or "(no field set)") + (f"; their note: “{pp.one(rv['note'])}”" if pp.one(rv["note"]) else "")
                              if rv is not None else "since withdrawn")
                           + "\n  the ruled segment: " + (pp.seg_line(conn, cfg, src) or "(no longer on record)"))
        probs = ct.probs_text(conn, cfg, sid, indent="  ")
        n_own = sum(1 for m in meta if "THE SEGMENT IN DOUBT" in m["label"])
        out.append((blocks, f"DOUBT {i + 1} - segment id={sid} ({pp.t(g['start'], '%a %Y-%m-%d %H:%M:%S')}, {ev._durtext(end - g['start'])})\n"
                    + "Why it is in doubt: " + "\n".join(why) + "\n" + cur + "\n"
                    + (f"Screenshots above for this doubt: {n_own} of the segment itself"
                       + ("" if n_own else f" ({_shots.why_none(conn, sid)})") + f", {len(meta) - n_own} of segments it may continue.\n")
                    + ("What the app's models gave each label, and what the segment might continue:\n" + probs + "\n" if probs else "")
                    + "Surrounding activity (>>> marks the segment in doubt; 20 minutes before it to 20 minutes after its end):\n" + "\n".join(lines)))
    return out


def generate_pending(conn, cfg: Config | None = None) -> int:
    """One LLM call turning up to MAX_TRIGGERS_PER_CALL doubts into questions."""
    from . import stats, summaries
    from .classifiers.claude_vision import _rules_doc
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    retire_threadless_questions(conn)
    trigs = _triggers(conn)[:MAX_TRIGGERS_PER_CALL]
    if not trigs:
        return 0
    content_blocks: list = []
    parts = []
    for blk, part in _doubt_parts(conn, cfg, trigs):
        content_blocks += blk
        parts.append(part)
    unsure = conn.execute("""SELECT q.question, q.note, g.app, g.domain, g.title, COALESCE(g.end, g.start) - g.start AS secs
                             FROM review_questions q JOIN segments g ON g.id = q.segment_id
                             WHERE q.status = 'unsure' ORDER BY q.answered_at DESC LIMIT ?""", (OTHER_LISTS,)).fetchall()
    if unsure:
        ulines = [f"- [{u['app']} {u['domain'] or ''} | {u['title'] or ''}, {u['secs']:.0f}s] {u['question']}"
                  + (f"  (their note: {u['note']})" if u["note"] else "") for u in unsure]
        content_blocks.append({"type": "text", "text":
            "The person could NOT answer these recent questions - they looked and genuinely could not tell. "
            "Doubts of the same shape (same kind of moment, same kind of distinction) are not worth raising: "
            "if a DOUBT below is one of those, either ask something the person CAN know instead, or omit it.\n"
            + "\n".join(ulines)})
    missed = conn.execute("""SELECT q.question, q.options, q.note, g.app, g.domain, g.title
                             FROM review_questions q JOIN segments g ON g.id = q.segment_id
                             WHERE q.status = 'none' ORDER BY q.answered_at DESC LIMIT ?""", (OTHER_LISTS,)).fetchall()
    if missed:
        mlines = []
        for u in missed:
            try:
                labels = "; ".join(o.get("label", "") for o in json.loads(u["options"])
                                   if o.get("label") != PASSTHROUGH_LABEL)
            except ValueError:
                labels = ""
            mlines.append(f"- [{u['app']} {u['domain'] or ''} | {u['title'] or ''}] {u['question']}"
                          f"  (offered: {labels})" + (f"  (what it really was, in their words: {u['note']})" if u["note"] else ""))
        content_blocks.append({"type": "text", "text":
            "For these recent questions the person answered NONE OF THE ABOVE - the options did not "
            "include what actually happened. The options must cover the plausible readings: when a "
            "DOUBT below is of the same shape, include the kind of reading they named.\n"
            + "\n".join(mlines)})
    content_blocks.append({"type": "text", "text": "\n\n".join(parts)
              + "\n\nWrite one question per dimension in doubt - usually one per DOUBT, two when both the "
              "thread and the content are in doubt (trigger_segment_id = the doubted segment's id on each). "
              "A DOUBT may be omitted when no question the person could answer would resolve it."})
    # The person's past rulings on segments like the doubted ones: a doubt one of them already
    # settles should not be asked again (until 2026-11-19 the question writer saw none of them).
    try:
        from . import evaluator as _ev
        tsegs = [dict(g) for g in (conn.execute("SELECT id, app, domain, title FROM segments WHERE id=?", (t["segment_id"],)).fetchone()
                                   for t in trigs) if g]
        prec = _ev._get()._similar_precedents(conn, tsegs, skip_recent=False)
    except Exception:
        log.exception("could not gather precedents for the question writer")
        prec = ""
    if prec:
        content_blocks.append({"type": "text", "text": prec + "\n\nWhen one of these past rulings already answers a DOUBT "
                               "(the same kind of situation, ruled on before), omit that DOUBT: do not ask the person "
                               "something they have already told the app."})
    from . import prompt_parts as _pp
    system = SYSTEM.format(rules=_rules_doc(), about_me=cfg.about_me.strip() or "(none)", legend=_pp.LOG_LEGEND)
    resp = llm.parse(_client(), site="questions", 
        model=cfg.claude_model, max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content_blocks}],
        output_format=QuestionBatch,
        # The configured effort ("low" was written in, as it was for the evaluator until
        # 2026-11-20): a declined doubt clears the needs-review flag for good.
        output_config={"effort": cfg.claude_effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default")
    if resp.parsed_output is None:
        return 0
    cost = ((resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
            + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
            + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)
    by_id = {t["segment_id"]: t for t in trigs}
    n = 0
    asked: set = set()
    per_q = cost / max(len(resp.parsed_output.questions), 1)
    with db.tx(conn):
        for q in resp.parsed_output.questions:
            t = by_id.get(q.trigger_segment_id)
            if t is None or not q.options:
                continue
            opts = []
            for o in q.options:
                o.switch_label = stored_switch(o.switch_label)  # tolerate the shown name 'shift'
                if o.switch_label is not None and o.switch_label not in SWITCHES:
                    continue
                if o.content is not None and o.content not in CONTENTS:
                    continue
                # Kinds are never solicited: the person asserts the thread relation; the app
                # derives detour/focus_start/self_distraction from what the thread became.
                if o.content == "transition":
                    # The model's pass-through options come framed as topically related
                    # ("part of the Signal conversation - just passing through"), which the
                    # person then has to disown; the neutral one appended below replaces them.
                    continue
                ante = o.antecedent_id if o.switch_label in ("continuation", "return") else None
                # self_distraction: the person's ruling on why they went there (true = drifted
                # without deciding to, false = on purpose). Until 2026-11-20 no option could carry
                # it, so "I was after something specific: an errand" (q2942) left the evaluator's
                # self-distraction standing. It means nothing on a continuation.
                opts.append({"label": o.label, "segment_id": o.segment_id,
                             "switch_label": o.switch_label, "antecedent_id": ante,
                             "interruption_kind": None, "content": o.content,
                             "hybrid": o.hybrid, "focused": o.focused,
                             "self_distraction": None if o.switch_label == "continuation" else o.self_distraction,
                             "sub_of": o.grew_out_of_id if o.switch_label == "interruption" else None})
            # Options must differ in the EDIT they make, not just in their wording. The model
            # likes to split one reading by what came after it ("a one-off question" vs "the
            # start of a run of lookups") - both write the same new thread on this segment, and
            # what the next segments continue is their own judgment. Keep the first label.
            # A blank field means "leave the current judgment", so an option that says nothing
            # about the switch and one that sets it to what the model already thinks are the
            # same ruling: compare the edits as they would land, not as written.
            seen_edits: set = set()
            distinct = []
            for o in opts:
                key = _resolved_edit(conn, o)
                if key in seen_edits:
                    log.warning("dropping duplicate option %r (same edit as an earlier one) on segment %d",
                                o["label"][:60], q.trigger_segment_id)
                    continue
                seen_edits.add(key)
                distinct.append(o)
            opts = distinct
            if len(opts) < 1:  # at least one substantive reading + the pass-through below
                continue
            if t["source"] == "coherence" and not any(o.get("switch_label") == "interruption" for o in opts):
                log.warning("rejecting coherence question without a split option (segment %d)", q.trigger_segment_id)
                continue
            # Offer the pass-through reading - the person may know they were only travelling
            # to another window (or it was simply sitting open on that desktop) - unless the
            # segment was plainly in use. It continues whatever thread came before, so it
            # attaches to that antecedent.
            if passthrough_applies(conn, q.trigger_segment_id):
                opts.append(passthrough_option(q.trigger_segment_id, opts))
            conn.execute(
                "INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created)"
                " VALUES (?,?,?,?,?,?,?)",
                (q.trigger_segment_id, t["source"], q.context, q.question, json.dumps(opts), per_q, time.time()))
            n += 1
            asked.add(q.trigger_segment_id)
        # Every doubt the model saw and passed over (or whose question failed validation) is
        # written down as declined; otherwise the same doubt is gathered and sent again on the
        # next pass, and the next, for as long as the model keeps passing.
        declined = [t for t in trigs if t["segment_id"] not in asked]
        for t in declined:
            decline(conn, t["segment_id"], t["source"], t["reason"], "the model found none the person could answer")
    log.info("generated %d review question(s), declined %d ($%.3f)", n, len(declined), cost)
    return n

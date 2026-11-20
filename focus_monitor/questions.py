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
    focused: bool | None = Field(default=None, description="Only when the DOUBT is whether the person's attention was focused in the segment: true for the option saying it was focused, false for the option saying it was unfocused. None for every other option.")
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
every segment of the person's day (content + whether the switch into it shifted their object of \
attention), but some judgments are in doubt. For each DOUBT presented, you write ONE question \
the person can answer in seconds, with 2-4 natural-language options.

DESCRIBE THE ACTIVITY, NOT THE WINDOWS. The context must say what the person was DOING - the
undertaking - inferred from titles, content labels, and the per-segment rationales (after //
in the log): "you were editing your personal site - coding in VS Code, running it in Terminal, \
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
are ONE option (the app merges them). Whether a separate thread then counts as a self-distraction \
is derived from the content and what followed; it is never asked.
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
sets switch_label (a thread relation) or a content label.
- A SAME-thread option (continuation/return) also names its antecedent_id: the id of the \
segment in thread A that the person would be agreeing this continues (pick the nearest \
A-thread segment visible in the log). This makes the answer an edge in the thread graph, \
not just a label.
- For boundary and coherence doubts (a stretch that may contain TWO objects of attention), ask \
about the moment the SECOND object would begin - the split point named or implied in the \
doubt's reason - and you MUST include an option asserting the separate thread: \
switch_label='shift' on the segment where it begins (its id is visible in the log). \
"Was <activity> at <time> a separate piece of work, or still part of <thread A>?" is the shape.
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
        out.append({"segment_id": r["segment_id"], "source": src,
                    "reason": (r["rationale"] or "")[-200:], "ts": r["start"]})
    # coherence flags -> a boundary question anchored at the CHANGEPOINT: the first segment
    # of the longest stretch whose content differs from the span's opening content (asking at
    # the span's head produces questions about the wrong end).
    for r in conn.execute("""
        SELECT s.start, s.end, s.issue FROM span_summaries s
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
        hint = dt.datetime.fromtimestamp(anchor["start"]).strftime("%H:%M")
        out.append({"segment_id": anchor["id"], "source": "coherence",
                    "reason": (f"span summary contradicts the one-object definition: {r['issue']} "
                               f"(likely split point: segment id={anchor['id']} at {hint})"),
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
    return (o["segment_id"], sw, ante, content, hybrid, o.get("focused"))


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
    if source == "phone":   # the model's own confidence is about the computer segment, not the phone
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
            ov = {sid: {k: o.get(k) for k in ("content", "switch_label", "interruption_kind", "antecedent_id", "hybrid", "focused")}}
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
PHONE_OPTIONS = (
    ("Part of what I was doing on the computer", {"switch_label": "continuation", "away": 0, "focused": 1, "self_distraction": 0}),
    ("A deliberate break or errand", {"switch_label": "interruption", "away": 1, "self_distraction": 0}),
    ("A self-distraction: I picked it up without a clear task", {"switch_label": "interruption", "away": 1, "self_distraction": 1}),
    ("The phone was just open; I was not looking at it", {"switch_label": "continuation", "content": "transition", "away": 0, "self_distraction": 0}),
)
_phone_checked: dict = {}   # seg_key -> the day stamp it was last checked at (in memory: cheap to redo)


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
    for p in rows:
        day = logical_date(p["start"])
        if day not in stamps:
            d0, d1 = day_bounds(day)
            ps = conn.execute("SELECT MAX(created) FROM phone_evals WHERE start >= ? AND start < ?", (d0, d1)).fetchone()[0]
            stamps[day] = max(_day_stamp(conn, d0, d1), float(ps or 0.0))
        if _phone_checked.get(p["seg_key"]) == stamps[day]:
            continue
        _phone_checked[p["seg_key"]] = stamps[day]
        flip = ({"switch_label": "continuation", "away": 0, "focused": 1, "self_distraction": 0} if p["away"]
                else {"switch_label": "interruption", "away": 1, "self_distraction": 0})
        if day not in base:
            base[day] = stats.daily_metrics(conn, day)
        m = stats.daily_metrics(conn, day, {"phone": {p["seg_key"]: flip}})
        moved = {k: m[k] - base[day][k] for k in ("total_focus_min", "longest_focus_min")}
        k, delta = max(moved.items(), key=lambda kv: abs(kv[1]))
        if abs(delta) < PHONE_Q_MIN:
            continue
        g = conn.execute("""SELECT id, app, title FROM segments WHERE start <= ? AND COALESCE(end, start) > ?
                            ORDER BY idle, start DESC LIMIT 1""", (p["start"], p["start"])).fetchone()
        if g is None:
            continue
        hhmm = dt.datetime.fromtimestamp(p["start"]).strftime("%H:%M")
        secs = p["end"] - p["start"]
        length = f"{secs / 60:.0f} min" if secs >= 90 else f"{secs:.0f} s"
        call = ("a self-distraction" if p["away"] and p["self_distraction"] else
                "a deliberate trip away from the work" if p["away"] else "part of what you were doing")
        what = "independent focus time" if k == "total_focus_min" else "longest focus span"
        ctx = (f"At {hhmm} your phone had {p['app']} open for {length}, while the computer was on {g['app']}"
               f"{' (' + g['title'][:60] + ')' if g['title'] else ''}. The app sees only which phone app was open and when, "
               f"so its call is a low-confidence one: {call}. Read the other way, the day's {what} "
               f"would be {abs(delta):.0f} min {'longer' if delta > 0 else 'shorter'}.")
        opts = [{"label": lab, "segment_id": g["id"], "phone_key": p["seg_key"], "phone": edit,
                 "switch_label": None, "antecedent_id": None, "interruption_kind": None, "content": None, "hybrid": None}
                for lab, edit in PHONE_OPTIONS]
        conn.execute("INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created, phone_key)"
                     " VALUES (?,?,?,?,?,0,?,?)",
                     (g["id"], "phone", ctx, f"What was the {p['app']} use on your phone at {hhmm}?", json.dumps(opts),
                      time.time(), p["seg_key"]))
        n += 1
    if n:
        conn.commit()
        log.info("wrote %d question(s) about load-bearing phone judgments", n)
    return n


SPOT_GAP_S = 2 * 3600.0     # at most one new spot check per this long, so they spread over the day
SPOT_MIN_S = 20.0           # shorter segments are not drawn
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
the notes after // in the log ("you were editing your personal site"), not a list of apps. Use \
"apparently" or "likely" where you are inferring.

Do NOT say or hint at the app's verdict: no "focused", "unfocused", "distracted", "drifting", \
"continuing", "shifted", "returned", and no content-type label. The person must judge that \
themselves. No question, no preamble; only the summary."""


class SpotContext(BaseModel):
    context: str = Field(description="2-3 sentences, second person, with clock times; no verdict words.")


def _spot_summary(conn, cfg: Config, g) -> tuple[str | None, float]:
    """One small model call: what was happening around a spot-checked segment, written the way the
    other review questions set up theirs (Eli, 2026-11-19). Returns (summary or None, cost)."""
    from . import shots as _shots, stats, summaries
    from .classifiers.claude_vision import _b64
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return None, 0.0
    blocks: list = []
    sh = next(iter(_shots.pick(conn, [g["id"]], 1)), None)
    b = _b64(sh["path"]) if sh else None
    if b:
        blocks.append({"type": "text", "text": "Screenshot from the marked segment:"})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
    segs = [x for x in stats.labelled_segments(conn, g["start"] - 1200, g["end"] + 600) if x["duration"] > 0]
    ids = [x["id"] for x in segs]
    rats = {r["segment_id"]: r["rationale"] for r in conn.execute(
        f"SELECT segment_id, rationale FROM seg_evals WHERE segment_id IN ({','.join('?' * len(ids))})", ids)
        if r["rationale"]} if ids else {}
    lines = [("  >>> " if x["id"] == g["id"] else "      ") + summaries.seg_line(x)
             + (f"  // {rats[x['id']][:90]}" if x["id"] in rats else "") for x in segs]
    blocks.append({"type": "text", "text": "Activity log (>>> marks the segment the question is about):\n" + "\n".join(lines)})
    try:
        resp = llm.parse(_client(), site="spot_check", model=cfg.claude_model, max_tokens=600,
                         system=[{"type": "text", "text": SPOT_SYSTEM}],
                         messages=[{"role": "user", "content": blocks}],
                         output_format=SpotContext, output_config={"effort": "low"},
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
    hhmm = dt.datetime.fromtimestamp(g["start"]).strftime("%H:%M")
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
        if old.get("summarised") is not None:   # already tried (true, or false = the call failed: leave it)
            continue
        g = conn.execute("""SELECT g.id, g.start, COALESCE(g.end, g.start) AS end, g.app, g.domain, g.title,
                                   e.content, e.switch_label, e.focused, e.antecedent_id, e.rationale
                            FROM segments g JOIN seg_evals e ON e.segment_id = g.id WHERE g.id=?""", (q["segment_id"],)).fetchone()
        if g is None or g["focused"] is None:
            continue
        axis = old.get("axis") or random.choice([a for a in SPOT_AXES if a != "switch" or g["switch_label"]])
        summary, cost = _spot_summary(conn, cfg, g)
        ctx, question, opts = _spot_question(g, axis, summary)
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
    from .classifiers.claude_vision import _b64
    content_blocks: list = []
    parts = []
    for i, t in enumerate(trigs):
        from . import shots as _shots
        sh = next(iter(_shots.pick(conn, [t["segment_id"]], 1)), None)
        if sh:
            b = _b64(sh["path"])
            if b:
                content_blocks.append({"type": "text", "text": f"Screenshot of DOUBT {i + 1}'s segment (id={t['segment_id']}){' - EXTERNAL display' if sh['display'] else ''}:"})
                content_blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        seg_ctx = stats.labelled_segments(conn, t["ts"] - 1200, t["ts"] + 1200)
        ctx_ids = [s["id"] for s in seg_ctx]
        rats = {}
        if ctx_ids:
            qs = ",".join("?" * len(ctx_ids))
            rats = {r2["segment_id"]: r2["rationale"] for r2 in conn.execute(
                f"SELECT segment_id, rationale FROM seg_evals WHERE segment_id IN ({qs})", ctx_ids)
                if r2["rationale"]}
        lines = [("  >>> " if s["id"] == t["segment_id"] else "      ")
                 + f"id={s['id']} " + summaries.seg_line(s)
                 + (f"  // {rats[s['id']][:90]}" if s["id"] in rats else "")
                 for s in seg_ctx if s["duration"] > 0]
        ev = conn.execute("SELECT content, switch_label, focused, self_distraction, next_task, rationale FROM seg_evals WHERE segment_id=?",
                          (t["segment_id"],)).fetchone()
        cur = (f"current judgment: content={ev['content']}, switch={shown_switch(ev['switch_label'])}"
               + ("" if ev["focused"] is None else (", focused" if ev["focused"] else ", unfocused"))
               + (", self-distraction" if ev["self_distraction"] else "")
               + (", next task in the block" if ev["next_task"] else "")
               + f" - {ev['rationale']}") if ev else "(no judgment)"
        parts.append(f"DOUBT {i + 1} - segment id={t['segment_id']} (source: {t['source']}; {t['reason']})\n"
                     f"{cur}\nSurrounding activity (>>> marks the segment in doubt):\n" + "\n".join(lines))
    unsure = conn.execute("""SELECT q.question, q.note, g.app, g.domain, g.title, COALESCE(g.end, g.start) - g.start AS secs
                             FROM review_questions q JOIN segments g ON g.id = q.segment_id
                             WHERE q.status = 'unsure' ORDER BY q.answered_at DESC LIMIT 12""").fetchall()
    if unsure:
        ulines = [f"- [{u['app']} {u['domain'] or ''} | {(u['title'] or '')[:50]}, {u['secs']:.0f}s] {u['question']}"
                  + (f"  (their note: {u['note']})" if u["note"] else "") for u in unsure]
        content_blocks.append({"type": "text", "text":
            "The person could NOT answer these recent questions - they looked and genuinely could not tell. "
            "Doubts of the same shape (same kind of moment, same kind of distinction) are not worth raising: "
            "if a DOUBT below is one of those, either ask something the person CAN know instead, or omit it.\n"
            + "\n".join(ulines)})
    missed = conn.execute("""SELECT q.question, q.options, q.note, g.app, g.domain, g.title
                             FROM review_questions q JOIN segments g ON g.id = q.segment_id
                             WHERE q.status = 'none' ORDER BY q.answered_at DESC LIMIT 12""").fetchall()
    if missed:
        mlines = []
        for u in missed:
            try:
                labels = "; ".join(o.get("label", "") for o in json.loads(u["options"])
                                   if o.get("label") != PASSTHROUGH_LABEL)
            except ValueError:
                labels = ""
            mlines.append(f"- [{u['app']} {u['domain'] or ''} | {(u['title'] or '')[:50]}] {u['question']}"
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
    system = SYSTEM.format(rules=_rules_doc(), about_me=cfg.about_me.strip() or "(none)")
    resp = llm.parse(_client(), site="questions", 
        model=cfg.claude_model, max_tokens=4000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content_blocks}],
        output_format=QuestionBatch,
        output_config={"effort": "low"},
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
                opts.append({"label": o.label, "segment_id": o.segment_id,
                             "switch_label": o.switch_label, "antecedent_id": ante,
                             "interruption_kind": None, "content": o.content,
                             "hybrid": o.hybrid, "focused": o.focused})
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

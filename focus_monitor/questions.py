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
from .seg_models import CONTENTS, KINDS, SWITCHES

log = logging.getLogger("questions")

MAX_TRIGGERS_PER_CALL = 6


class QOption(BaseModel):
    label: str = Field(description="The answer in the person's own terms, e.g. 'a continuation of the essay writing'.")
    segment_id: int = Field(description="The segment this option's edit applies to.")
    switch_label: str | None = Field(default=None, description="continuation | interruption | return, when the option settles the thread relation.")
    antecedent_id: int | None = Field(default=None, description="When the option asserts SAME thread (continuation/return): the id of the segment in that thread the person would be agreeing this continues - the A-thread segment your question names. Null for 'separate thread' (interruption) and content-only options.")
    interruption_kind: str | None = Field(default=None, description="ALWAYS leave None. Kinds (detour vs focus_start vs self_distraction) are derived by the app from what the thread became - the person is never asked to choose them.")
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
- same thread: switch_label='continuation'; separate thread: switch_label='interruption'.
  2. "You were doing A, briefly went to B, then switched to C. Was C the same thread/intention \
as A, or something different?" - same: switch_label='return'; different: \
switch_label='interruption'.
Content questions are equally welcome when content is the doubt: "was drafting that Signal \
message routine correspondence ('correspondence') or writing you were absorbed in ('creative')?" - each \
option sets a content label.
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
derives this). Options therefore never set interruption_kind; asserting 'interruption' is \
enough - the app derives the kind from what followed. The standing exception: a confusion or \
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
switch_label='interruption' on the segment where it begins (its id is visible in the log). \
"Was <activity> at <time> a separate piece of work, or still part of <thread A>?" is the shape.
- Options must be mutually exclusive and cover the plausible readings. Do not editorialize; \
the person decides.
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
    rows = conn.execute("""
        SELECT e.segment_id, e.uncertain, e.rationale, g.start
        FROM seg_evals e JOIN segments g ON g.id = e.segment_id
        LEFT JOIN seg_reviews r ON r.segment_id = e.segment_id
        LEFT JOIN review_questions q ON q.segment_id = e.segment_id
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
    out.sort(key=lambda x: ({"coherence": 0, "propagation": 1, "audit": 2, "ensemble": 3}.get(x["source"], 4), -x["ts"]))
    return out


def ante_top(ap: dict) -> float:
    """The evaluator's confidence in its top thread attachment, with the candidates renormalized
    to sum to 1 (rows stored before 2026-11-05 may carry a lone candidate at 0.2)."""
    tot = sum(ap.values())
    return max(ap.values()) / tot if tot > 0 else 0.0


PASSTHROUGH_LABEL = "Neither - just a window I passed through or had open, not actually looked at"


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
                 "bounce_aways": ("bounce-aways", 10.0)}
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
            sid = o.get("segment_id") or r["segment_id"]
            ov = {sid: {k: o.get(k) for k in ("content", "switch_label", "interruption_kind", "antecedent_id", "hybrid")}}
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
        v = d.get(k, 0) or 0
        if k == "bounce_aways" and v >= 1:
            parts.append((v * w, f"{name} by {v:.0f}"))
        elif k != "bounce_aways" and v >= 1:
            parts.append((v * w, f"{name} by {v:.0f} min"))
    parts.sort(reverse=True)
    return ("could move " + ", ".join(t for _, t in parts[:2])) if parts else "no effect on the day's metrics"


def generate_pending(conn, cfg: Config | None = None) -> int:
    """One LLM call turning up to MAX_TRIGGERS_PER_CALL doubts into questions."""
    from . import stats, summaries
    from .classifiers.claude_vision import _rules_doc
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    trigs = _triggers(conn)[:MAX_TRIGGERS_PER_CALL]
    if not trigs:
        return 0
    from .classifiers.claude_vision import _b64
    content_blocks: list = []
    parts = []
    for i, t in enumerate(trigs):
        sh = conn.execute("SELECT path, display FROM segment_shots WHERE segment_id=? ORDER BY ts LIMIT 1",
                          (t["segment_id"],)).fetchone()
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
        ev = conn.execute("SELECT content, switch_label, interruption_kind, rationale FROM seg_evals WHERE segment_id=?",
                          (t["segment_id"],)).fetchone()
        cur = (f"current judgment: content={ev['content']}, switch={ev['switch_label']}, "
               f"kind={ev['interruption_kind']} - {ev['rationale']}") if ev else "(no judgment)"
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
              + "\n\nWrite one question per DOUBT (trigger_segment_id = the doubted segment's id). "
              "A DOUBT may be omitted when no question the person could answer would resolve it."})
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
    per_q = cost / max(len(resp.parsed_output.questions), 1)
    with db.tx(conn):
        for q in resp.parsed_output.questions:
            t = by_id.get(q.trigger_segment_id)
            if t is None or not q.options:
                continue
            opts = []
            for o in q.options:
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
                             "hybrid": o.hybrid})
            if len(opts) < 1:  # at least one substantive reading + the pass-through below
                continue
            if t["source"] == "coherence" and not any(o.get("switch_label") == "interruption" for o in opts):
                log.warning("rejecting coherence question without a split option (segment %d)", q.trigger_segment_id)
                continue
            # Always offer the pass-through reading - the person may know they were only
            # travelling to another window (or it was simply sitting open on that desktop).
            # It continues whatever thread came before, so it attaches to that antecedent.
            opts.append(passthrough_option(q.trigger_segment_id, opts))
            conn.execute(
                "INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created)"
                " VALUES (?,?,?,?,?,?,?)",
                (q.trigger_segment_id, t["source"], q.context, q.question, json.dumps(opts), per_q, time.time()))
            n += 1
    log.info("generated %d review question(s) ($%.3f)", n, cost)
    return n

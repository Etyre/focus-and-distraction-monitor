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

from . import db
from .config import (PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, Config,
                     load_config)
from .seg_models import CONTENTS, KINDS, SWITCHES

log = logging.getLogger("questions")

MAX_TRIGGERS_PER_CALL = 6


class QOption(BaseModel):
    label: str = Field(description="The answer in the person's own terms, e.g. 'a continuation of the essay writing'.")
    segment_id: int = Field(description="The segment this option's edit applies to.")
    switch_label: str | None = Field(default=None, description="continuation | interruption | return, when the option settles the switch judgment.")
    interruption_kind: str | None = Field(default=None, description="self_distraction | focus_start | detour, when switch_label is 'interruption'.")
    content: str | None = Field(default=None, description="A content label, when the option settles content.")


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

Rules for options:
- Natural language first ("a continuation of the essay work"), schema underneath: each option \
sets switch_label (continuation | interruption | return) and, for interruption, the kind \
(self_distraction | focus_start | detour); or a content label when content is what's in doubt.
- For boundary doubts ("did a new piece of work start here?"), the split option is \
switch_label='interruption', interruption_kind='focus_start' on the segment where the new work \
would begin.
- For coherence doubts (a span that contains TWO objects of attention), ask about the moment \
the SECOND object begins - the split point named or implied in the doubt's reason - and you MUST \
include an option affirming the split: switch_label='interruption', \
interruption_kind='focus_start' on the segment where the second object begins (its id is \
visible in the log). "Did a new piece of work start at <time>?" is the shape.
- Options must be mutually exclusive and cover the plausible readings. Do not editorialize; \
the person decides.

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
    rows = conn.execute("""
        SELECT e.segment_id, e.uncertain, e.rationale, g.start
        FROM seg_evals e JOIN segments g ON g.id = e.segment_id
        LEFT JOIN seg_reviews r ON r.segment_id = e.segment_id
        LEFT JOIN review_questions q ON q.segment_id = e.segment_id
        WHERE g.start >= ? AND e.uncertain = 1 AND r.segment_id IS NULL AND q.id IS NULL
        ORDER BY g.start DESC""", (t0,)).fetchall()
    for r in rows:
        src = "audit" if (r["rationale"] or "").find("daily audit") >= 0 else "self_uncertain"
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
        for sid in _stats.select_for_review(conn, d0, d1, 8):
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
    out = [t for t in out if not (t["segment_id"] in seen or seen.add(t["segment_id"]))]
    out.sort(key=lambda x: ({"coherence": 0, "audit": 1, "ensemble": 2}.get(x["source"], 3), -x["ts"]))
    return out


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
    parts = []
    for i, t in enumerate(trigs):
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
    prompt = ("\n\n".join(parts)
              + "\n\nWrite one question per DOUBT (trigger_segment_id = the doubted segment's id).")
    system = SYSTEM.format(rules=_rules_doc(), about_me=cfg.about_me.strip() or "(none)")
    resp = _client().beta.messages.parse(
        model=cfg.claude_model, max_tokens=4000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
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
                if o.interruption_kind is not None and o.interruption_kind not in KINDS:
                    continue
                if o.content is not None and o.content not in CONTENTS:
                    continue
                opts.append({"label": o.label, "segment_id": o.segment_id,
                             "switch_label": o.switch_label,
                             "interruption_kind": o.interruption_kind, "content": o.content})
            if len(opts) < 2:
                continue
            if t["source"] == "coherence" and not any(o.get("interruption_kind") == "focus_start" for o in opts):
                log.warning("rejecting coherence question without a split option (segment %d)", q.trigger_segment_id)
                continue
            # Always offer the pass-through reading - the person may know they were only
            # travelling to another window, which the model rarely proposes on its own.
            if not any(o.get("content") == "transition" for o in opts):
                opts.append({"label": "Just passing through - a transition on my way to something else",
                             "segment_id": q.trigger_segment_id,
                             "switch_label": "continuation", "interruption_kind": None,
                             "content": "transition"})
            conn.execute(
                "INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd, created)"
                " VALUES (?,?,?,?,?,?,?)",
                (q.trigger_segment_id, t["source"], q.context, q.question, json.dumps(opts), per_q, time.time()))
            n += 1
    log.info("generated %d review question(s) ($%.3f)", n, cost)
    return n

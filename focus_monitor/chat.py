"""Interactive chats with Claude: about one switch, one focus span, or a whole day."""
from __future__ import annotations

import logging

import anthropic

from . import LABEL_HELP, db, llm
from .classifiers.base import build_context
from .classifiers.claude_vision import _b64
from .config import Config

log = logging.getLogger("chat")

SYSTEM = """You are helping this person review their own attention-monitoring log. They are looking at ONE \
window switch and want to reason about it with you. You can see the before/after screenshots, the \
surrounding activity, their Toggl "intent" entry, and how the models classified the switch.

Answer their questions about THIS switch concisely and concretely - reference what is actually visible on \
screen and in the activity log. If they ask, say which label you'd give and why, and note honestly when it's \
genuinely ambiguous. Don't lecture; be a sharp, brief thinking partner.

The four labels:
{labels}

About this person (their own words):
{about_me}
"""

_MAXLBL = ["continuation", "interruption", "distraction", "task_change"]


def _top_label(p: dict) -> str:
    probs = {k: p[f"p_{k}"] for k in _MAXLBL}
    return max(probs, key=probs.get)


class ReviewChat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        self.system = llm.with_preamble(SYSTEM.format(
            labels="\n".join(f"- {k}: {v}" for k, v in LABEL_HELP.items()),
            about_me=cfg.about_me.strip() or "(no description given)"))

    def _context_blocks(self, conn, switch_id: int) -> list[dict]:
        ctx = build_context(conn, switch_id)
        full = db.switch_full(conn, switch_id)
        blocks: list[dict] = []
        before = _b64(ctx.from_seg["last_screenshot"] if ctx.from_seg else None)
        after = _b64(ctx.to_seg["first_screenshot"])
        if before:
            blocks.append({"type": "text", "text": "Screenshot BEFORE the switch:"})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": before}})
        if after:
            blocks.append({"type": "text", "text": "Screenshot shortly AFTER the switch:"})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": after}})
        preds = "\n".join(f"  - {p['model']}: {_top_label(p)} — {p['rationale']}" for p in full["predictions"])
        ens = full["ensemble"]
        review = full["review"]
        summary = ctx.narrative()
        summary += f"\n\nHow the models classified this switch:\n{preds or '  (none yet)'}"
        if ens:
            summary += f"\nEnsemble: {ens['label']}" + (" (flagged uncertain)" if ens["uncertain"] else "")
        if review:
            summary += f"\nThe person already reviewed this as: {review['label']}"
        blocks.append({"type": "text", "text": summary})
        return blocks

    def reply(self, conn, switch_id: int, messages: list[dict]) -> str:
        text, cost = _run_chat(self.client, self.cfg, self.system, self._context_blocks(conn, switch_id), messages)
        log_exchange(conn, "switch", str(switch_id), messages, text, cost, self.cfg.claude_model)
        return text


def _api_messages(ctx_blocks, messages) -> list[dict]:
    """The thread as the API wants it: the context blocks ride along with the first question."""
    api_msgs: list[dict] = []
    for i, m in enumerate(messages):
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = (m.get("text") or "").strip()
        if i == 0 and role == "user":
            api_msgs.append({"role": "user",
                             "content": ctx_blocks + [{"type": "text", "text": "\n\nMy question: " + text}]})
        else:
            api_msgs.append({"role": role, "content": text})
    return api_msgs


def _run_chat(client, cfg, system, ctx_blocks, messages, max_tokens: int = 1024) -> str:
    api_msgs = _api_messages(ctx_blocks, messages)
    if not api_msgs:
        return "", 0.0
    system = llm.with_preamble(system)
    tools = [llm.CONTACT_TOOL]
    resp = client.messages.create(
        model=cfg.claude_model, max_tokens=max_tokens,
        system=system,
        messages=api_msgs,
        tools=tools,
        output_config={"effort": "low"},
    )
    from .classifiers.claude_vision import _cost
    cost = _cost(resp.usage)
    rec = llm.contacted("chat", cfg.claude_model, system, api_msgs, resp, tools)
    if rec:
        return llm.contact_notice(rec), cost
    if resp.stop_reason == "refusal":
        return "(Claude declined to answer that.)", cost
    return ("".join(b.text for b in resp.content if b.type == "text").strip() or "(no reply)"), cost


def log_exchange(conn, kind: str, ref: str, messages: list[dict], reply: str, cost: float,
                 model: str | None, question_id: int | None = None,
                 proposal: dict | None = None) -> int:
    """Persist the latest question and Claude's answer. The client sends the whole thread each
    turn; only the final user message is new, so that is what gets written (plus the reply).
    Returns the id of the stored reply (the handle for applying any proposal it carries)."""
    import json as _json
    import time as _t
    last = next((m for m in reversed(messages) if m.get("role") != "assistant"), None)
    now = _t.time()
    with conn:
        if last is not None and (last.get("text") or "").strip():
            conn.execute("INSERT INTO chat_messages(kind, ref, role, text, question_id, created) VALUES (?,?,?,?,?,?)",
                         (kind, ref, "user", last["text"].strip(), question_id, now))
        cur = conn.execute("INSERT INTO chat_messages(kind, ref, role, text, question_id, model, cost_usd, created, proposal)"
                           " VALUES (?,?,?,?,?,?,?,?,?)",
                           (kind, ref, "assistant", reply, question_id, model, cost, now + 0.001,
                            _json.dumps(proposal) if proposal else None))
        return int(cur.lastrowid)


def begin_exchange(conn, kind: str, ref: str, messages: list[dict], model: str | None) -> int:
    """Write the new user message and an EMPTY reply row first, returning the reply's id - so
    edits made while answering can carry it as provenance. finish_exchange fills the row in;
    abort_exchange removes both rows if the call fails."""
    import time as _t
    last = next((m for m in reversed(messages) if m.get("role") != "assistant"), None)
    now = _t.time()
    with db.tx(conn):
        if last is not None and (last.get("text") or "").strip():
            conn.execute("INSERT INTO chat_messages(kind, ref, role, text, created) VALUES (?,?,?,?,?)",
                         (kind, ref, "user", last["text"].strip(), now))
        cur = conn.execute("INSERT INTO chat_messages(kind, ref, role, text, model, cost_usd, created)"
                           " VALUES (?,?,?,?,?,?,?)", (kind, ref, "assistant", "", model, 0.0, now + 0.001))
        return int(cur.lastrowid)


def finish_exchange(conn, mid: int, text: str, cost: float, proposal: dict | None = None) -> None:
    import json as _json
    with db.tx(conn):
        conn.execute("UPDATE chat_messages SET text=?, cost_usd=?, proposal=? WHERE id=?",
                     (text, cost, _json.dumps(proposal) if proposal else None, mid))


def abort_exchange(conn, mid: int) -> None:
    with db.tx(conn):
        conn.execute("DELETE FROM chat_messages WHERE id=? OR (id=? AND role='user')", (mid, mid - 1))


def history(conn, kind: str, ref: str) -> list[dict]:
    import json as _json
    return [{"id": r["id"], "role": r["role"], "text": r["text"], "created": r["created"],
             "proposal": _json.loads(r["proposal"]) if r["proposal"] else None}
            for r in conn.execute(
        "SELECT id, role, text, created, proposal FROM chat_messages WHERE kind=? AND ref=? ORDER BY created, id", (kind, ref))]


SPAN_SYSTEM = """You are helping this person review their own attention-monitoring log. They are looking at ONE \
hypothesized focus span and deciding (a) whether it really was a focus span at all, and (b) which kind. \
You see the WHOLE DAY's activity log with the span under review marked in it - so you can reason about what \
led into the span, what followed it, and whether its boundaries are right - plus the day's other spans, the \
day's Toggl entries, the system's current call and prior summary of this span, and a few sampled screenshots \
from inside it.

Answer their questions about THIS span concisely and concretely - point at specific blocks, times, and \
what's visible on screen. If they ask for a verdict, say whether you'd call it a focus span, which subtype, \
and why - and note honestly when it's genuinely ambiguous. Their calibration feedback is that the system \
has been far too optimistic about what counts as focused work; do not repeat that mistake. Don't lecture; \
be a sharp, brief thinking partner.

Span subtypes: creative_work, focused_task, reading, planning, other_focused, meeting. (Correspondence - chat, email - \
is interactive, never a focus span on its own; only email processed inside a Toggl-covered task block belongs to that span.)

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""


class SpanChat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        from .classifiers.claude_vision import _rules_doc
        self.system = llm.with_preamble(SPAN_SYSTEM.format(
            rules=_rules_doc(), about_me=cfg.about_me.strip() or "(no description given)"))

    def _context_blocks(self, conn, start: float, end: float) -> list[dict]:
        import datetime as dt
        from . import stats, summaries, toggl
        from .config import logical_date
        t0, t1 = stats.day_bounds(logical_date(start))
        day_segs = stats.labelled_segments(conn, t0, t1)
        se = stats.spans_and_events(day_segs, conn=conn)
        mid = (start + end) / 2
        sp = next((s for s in se["focus_spans"] + se["disqualified"]
                   if s["start"] - 1 <= mid < s["end"] + 1), None)
        if sp is None:
            raise ValueError("no span at that time")
        blocks: list[dict] = []
        sp_ids = [s["id"] for r in sp["runs"] for s in r["segments"]]
        shot_rows = []
        if sp_ids:
            qs = ",".join("?" * len(sp_ids))
            shot_rows = [dict(r2) for r2 in conn.execute(
                f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) ORDER BY ts", sp_ids)]
        if not shot_rows:
            shot_rows = [{"segment_id": s["id"], "ts": s["clip_start"], "path": s["first_screenshot"], "display": 0}
                         for r in sp["runs"] for s in r["segments"] if s.get("first_screenshot")]
        step = max(1, len(shot_rows) // 4)
        for sh in shot_rows[::step][:4]:
            b = _b64(sh["path"])
            if b:
                blocks.append({"type": "text", "text": f"Screenshot at {dt.datetime.fromtimestamp(sh['ts']).strftime('%H:%M')}{' - EXTERNAL display' if sh.get('display') else ''}:"})
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        rats = {}
        if sp_ids:
            qs = ",".join("?" * len(sp_ids))
            rats = {r2["segment_id"]: r2["rationale"] for r2 in conn.execute(
                f"SELECT segment_id, rationale FROM seg_evals WHERE segment_id IN ({qs})", sp_ids)
                if r2["rationale"]}
        call = sp.get("disq_reason") and f"NOT focused work ({sp['disq_reason']})" or sp.get("subtype")
        row = summaries.lookup(conn, sp["start"], sp["end"])

        def _t(ts, fmt="%H:%M"):
            return dt.datetime.fromtimestamp(ts).strftime(fmt)

        overview = "\n".join(
            f"  {_t(s2['start'])}-{_t(s2['end'])}: "
            + (f"DISQUALIFIED ({s2.get('disq_reason')})" if s2.get("disq_reason") else s2.get("subtype", "?"))
            + f", {s2['focus_min']:.0f} min focused"
            + ("  <== THE SPAN UNDER REVIEW" if s2 is sp else "")
            for s2 in sorted(se["focus_spans"] + se["disqualified"], key=lambda s2: s2["start"]))
        tg = toggl.entries_between(conn, t0, t1)
        tgtxt = "\n".join(
            f"  {_t(e['start'])}-{_t(e['stop'] or t1)}: {e['description'] or '(unnamed)'}"
            for e in tg) or "  (no Toggl entries at all this day)"
        # Full-day log with the span marked; elide far-away stretches to bound the prompt.
        before = [s2 for s2 in day_segs if s2["clip_end"] <= sp["start"] and s2["duration"] > 0]
        inside = [s2 for s2 in day_segs if sp["start"] <= s2["clip_start"] < sp["end"] and s2["duration"] > 0]
        after = [s2 for s2 in day_segs if s2["clip_start"] >= sp["end"] and s2["duration"] > 0]
        lines = []
        if len(before) > 120:
            lines.append(f"  ... {len(before) - 120} earlier blocks elided ...")
            before = before[-120:]
        lines += [summaries.seg_line(s2) for s2 in before]
        lines.append(">>> START OF THE SPAN UNDER REVIEW <<<")
        lines += [summaries.seg_line(s2) + (f"  // {rats[s2['id']][:80]}" if s2["id"] in rats else "")
                  for s2 in inside]
        lines.append(">>> END OF THE SPAN UNDER REVIEW <<<")
        lines += [summaries.seg_line(s2) for s2 in after[:120]]
        if len(after) > 120:
            lines.append(f"  ... {len(after) - 120} later blocks elided ...")
        text = (f"The span under review: {_t(sp['start'], '%Y-%m-%d %H:%M')}-{_t(sp['end'])} "
                f"({sp['duration'] / 60:.0f} min, {sp['focus_min']:.0f} focused, "
                f"{len(sp.get('detours', []))} contained interruption(s), ended by {sp.get('ended_by')}).\n"
                f"System's current call: {call}.\n"
                f"Prior summary of this span: {row['summary'] if row else '(none yet)'}\n\n"
                f"All hypothesized spans this day:\n{overview}\n\n"
                f"Toggl entries this day:\n{tgtxt}\n\n"
                f"The whole day's activity log (every block; state and the switch label that led into it; "
                f"the span under review is marked):\n" + "\n".join(lines))
        blocks.append({"type": "text", "text": text})
        return blocks

    def reply(self, conn, start: float, end: float, messages: list[dict]) -> str:
        text, cost = _run_chat(self.client, self.cfg, self.system, self._context_blocks(conn, start, end), messages)
        log_exchange(conn, "span", str(round(start)), messages, text, cost, self.cfg.claude_model)
        return text


DAY_SYSTEM = """You are helping this person review their own attention-monitoring log. They are looking at ONE WHOLE \
DAY and want to talk it over with you: how the day went, where their attention held and where it broke, whether \
the system's calls look right, what the day's shape suggests for tomorrow.

You see the day's headline metrics, every hypothesized focus span (qualifying and disqualified) with the system's \
call and its summary, the self-distraction and interruption events, the day's Toggl "intent" entries, the whole \
day's activity log, and a few screenshots sampled across the day.

Answer concisely and concretely - point at specific times, spans, and blocks. When asked for an assessment, give \
one and say what it rests on; note honestly when the data is thin or a call is ambiguous. Their calibration \
feedback is that the system has been far too optimistic about what counts as focused work; do not repeat that \
mistake. Don't lecture; be a sharp, brief thinking partner.

You can also ACT, but only through the tools:
- set_segment_judgment: change the judgment of one or more activity blocks (blocks are numbered #id in the \
  log). This writes the person's authoritative review of each block, flagged as coming from this conversation.
- set_span_verdict: rule on a hypothesized span (by its start/end times from the span list).
- open_review_question: put a question in the person's review queue for later, with options that each map to \
  a concrete judgment - use this when a call needs their eyes and they do not want to settle it now.
- propose_rule: suggest a general rule for their rules document (they approve it with a click).
How a block's STATE (focus / interrupted / distracted) is derived from these labels - you must work with this: \
a 'continuation' block inherits the state of the block before it; a 'return' block is focus; an 'interruption' \
block starts a new thread and takes its kind (focus_start -> focus, detour -> interrupted, self_distraction -> \
distracted); passive content is always distracted; correspondence content with no Toggl entry is interrupted - \
UNLESS the person explicitly rules a block an exception (focus_exception=true: "this was correspondence and it \
still counts as independent focus"); never set that on your own judgment, only when they ask for it. \
So to make a stretch count as focused work: set its FIRST block to switch_label=interruption with \
interruption_kind=focus_start (only when the person has ruled it was focused work; otherwise leave the kind null \
and the app derives it), give the blocks inside it the right content with switch_label=continuation, and after \
any block you leave out as a diversion (a feed flip, a chat) mark the next block of the thread as 'return' with \
antecedent_id = an earlier block of that thread - otherwise it inherits the diversion's state. Hypothesized \
spans are then recomputed from the blocks; you cannot create a span directly.
Make a change only when the person has clearly asked for it or agreed to it; discuss first otherwise. Edit \
every block the agreed change covers - the person expects the timeline to match what was agreed. Afterwards, state exactly what you changed. NEVER say you changed, queued, flagged or applied anything \
unless a tool call in this reply actually did it; if a tool reports an error, say so. The log shows every block's \
CURRENT labels (a user ruling shows as <switch: ...> and the content tag): do not redo edits that are already in \
place, and do not open a review question that is already in the queue. A field you leave null on \
set_segment_judgment keeps the person's earlier ruling on that block. You cannot edit the rules \
document directly or change anything the tools do not cover - say so when asked.

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""

_LABEL_DOC = ("content = what the block was: creative | task | correspondence | reading | planning | other | passive "
              "| meeting | idle | transition. switch_label = how it relates to the thread before it: continuation "
              "(same thread) | interruption (a separate thread began) | return (back to an earlier thread; give "
              "antecedent_id). hybrid = the activity was carried on inside a live discussion. null leaves a field "
              "as judged.")

DAY_TOOLS = [
    {"name": "set_segment_judgment",
     "description": "Change the judgment of one or more activity blocks (their #ids from the log) to the SAME "
                    "labels. Writes the person's authoritative review of each, exactly as if they edited it "
                    "themselves, flagged as coming from this conversation. Only after they asked for or agreed to "
                    "the change. List every block the agreed change covers. " + _LABEL_DOC,
     "input_schema": {"type": "object", "properties": {
         "segment_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
         "content": {"type": ["string", "null"], "enum": ["creative", "task", "correspondence", "reading", "planning",
                                                         "other", "passive", "meeting", "idle", "transition", None]},
         "switch_label": {"type": ["string", "null"], "enum": ["continuation", "interruption", "return", None]},
         "interruption_kind": {"type": ["string", "null"], "enum": ["focus_start", "detour", "self_distraction", None],
                               "description": "With switch_label=interruption only: what the new thread was. Set it "
                                              "only when the person has explicitly ruled on it; null lets the app "
                                              "derive it from what followed."},
         "antecedent_id": {"type": ["integer", "null"]},
         "hybrid": {"type": ["boolean", "null"]},
         "focus_exception": {"type": ["boolean", "null"],
                             "description": "true ONLY when the person explicitly asked that this block count as "
                                            "independent focus although its content (correspondence with no Toggl "
                                            "entry) normally cannot. Never set it on your own judgment; null = no "
                                            "ruling; false withdraws an earlier ruling."},
         "note": {"type": "string", "description": "One sentence in the person's words: what was really going on and "
                                                   "why it is labelled this way. Stored as the precedent future "
                                                   "judgments see."}},
         "required": ["segment_ids", "note"]}},
    {"name": "set_span_verdict",
     "description": "Rule on a hypothesized span from the span list, by its start and end times (HH:MM as listed): "
                    "not a focus span, or a focus span of a given subtype. Only after the person asked for or agreed "
                    "to it.",
     "input_schema": {"type": "object", "properties": {
         "start_hhmm": {"type": "string"}, "end_hhmm": {"type": "string"},
         "verdict": {"type": "string", "enum": ["focus", "not_focus"]},
         "subtype": {"type": ["string", "null"], "enum": ["creative_work", "focused_task", "reading", "planning",
                                                         "other_focused", "meeting", None]},
         "note": {"type": "string"}},
         "required": ["start_hhmm", "end_hhmm", "verdict", "note"]}},
    {"name": "open_review_question",
     "description": "Put a question in the person's review queue to answer later. Each option maps to a concrete "
                    "judgment of a block (a neutral 'just passing through' option is added automatically). Use "
                    "when a call needs their eyes and they do not want to settle it now. " + _LABEL_DOC,
     "input_schema": {"type": "object", "properties": {
         "segment_id": {"type": "integer", "description": "The block whose judgment is in doubt."},
         "context": {"type": "string", "description": "1-2 sentences setting the scene (what happened around it)."},
         "question": {"type": "string"},
         "options": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
             "label": {"type": "string", "description": "The reading, in the person's own terms."},
             "segment_id": {"type": ["integer", "null"], "description": "The block this option edits (default: the question's)."},
             "content": {"type": ["string", "null"]}, "switch_label": {"type": ["string", "null"]},
             "antecedent_id": {"type": ["integer", "null"]}, "hybrid": {"type": ["boolean", "null"]}},
             "required": ["label"]}}},
         "required": ["segment_id", "context", "question", "options"]}},
    {"name": "propose_rule",
     "description": "Propose ONE general rule for the person's rules document, for them to approve with a click. "
                    "Written in the second person as the document is ('If I ...').",
     "input_schema": {"type": "object", "properties": {"rule": {"type": "string"}}, "required": ["rule"]}},
]


class DayChat:
    """Chat about a whole day. Context is the full day: metrics, spans, events, Toggl, the
    activity log (thinned when long) and screenshots sampled across the day."""

    MAX_LOG_LINES = 450

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        from .classifiers.claude_vision import _rules_doc
        self.system = llm.with_preamble(DAY_SYSTEM.format(
            rules=_rules_doc(), about_me=cfg.about_me.strip() or "(no description given)"))

    def _context_blocks(self, conn, day) -> list[dict]:
        import datetime as dt
        from . import stats, summaries, toggl
        t0, t1 = stats.day_bounds(day)
        segs = stats.labelled_segments(conn, t0, t1)
        se = stats.spans_and_events(segs, conn=conn)
        m = stats.daily_metrics(conn, day)

        def _t(ts, fmt="%H:%M"):
            return dt.datetime.fromtimestamp(ts).strftime(fmt)

        blocks: list[dict] = []
        # Screenshots: up to six, spread evenly over the active part of the day.
        shots = [s for s in segs if s.get("first_screenshot") and s.get("state") != "idle" and s["duration"] > 0]
        step = max(1, len(shots) // 6)
        for s in shots[::step][:6]:
            b = _b64(s["first_screenshot"])
            if b:
                blocks.append({"type": "text", "text": f"Screenshot at {_t(s['clip_start'])} ({s.get('app') or '?'}):"})
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})

        metrics = (f"  active {m['active_min']:.0f} min; independent focus {m['total_focus_min']:.0f} min "
                   f"(longest span {m['longest_focus_min']:.0f} min, longest fully-absorbed {m['longest_absorbed_min']:.0f} min, "
                   f"{m['fully_absorbed_spans']} fully-absorbed span(s)); meetings/hybrid {m['meeting_focus_min']:.0f} min; "
                   f"correspondence {m['correspondence_min']:.0f} min; passive consumption {m['passive_min']:.0f} min; "
                   f"distracted {m['distracted_min']:.0f} min; interruptions inside spans {m['interruption_count']} "
                   f"(self-distractions {m['distraction_count']}, bounce-aways all day {m['bounce_aways']}); "
                   f"{m['switches']} window switches; {m['short_breaks']} short inactivity breaks "
                   f"({m['short_break_min']:.0f} min).")

        def _span_line(sp):
            call = (f"DISQUALIFIED ({sp.get('disq_reason')})" if sp.get("disq_reason")
                    else (sp.get("subtype") or "?") + (" [hybrid]" if sp.get("hybrid") else "")
                    + (" [fully absorbed]" if sp.get("fully_absorbed") else ""))
            row = summaries.lookup(conn, sp["start"], sp["end"])
            line = (f"  {_t(sp['start'])}-{_t(sp['end'])}  {call}: {sp['duration'] / 60:.0f} min, "
                    f"{sp['focus_min']:.0f} focused, {len(sp.get('detours', []))} interruption(s), "
                    f"ended by {sp.get('ended_by')}")
            if row and row["summary"]:
                line += f"\n      summary: {row['summary']}"
            return line

        spans = sorted(se["focus_spans"] + se["disqualified"], key=lambda s: s["start"])
        spantxt = "\n".join(_span_line(sp) for sp in spans) or "  (no spans hypothesized)"

        def _ev_line(e, kind):
            where = ", ".join(e.get("where") or []) if isinstance(e.get("where"), list) else (e.get("where") or "")
            return (f"  {_t(e['start'])}  {kind}, {e['duration'] / 60:.1f} min"
                    + (" inside a focus span" if e.get("inside_span") else "") + (f": {where}" if where else ""))

        events = sorted([(e["start"], _ev_line(e, "self-distraction")) for e in se["distractions"]]
                        + [(e["start"], _ev_line(e, "interruption")) for e in se["interruptions"]])
        evtxt = "\n".join(l for _, l in events) or "  (none)"

        tg = toggl.entries_between(conn, t0, t1)
        tgtxt = "\n".join(f"  {_t(e['start'])}-{_t(e['stop'] or t1)}: {e['description'] or '(unnamed)'}"
                           for e in tg) or "  (no Toggl entries this day)"
        oq = conn.execute("""SELECT q.id, q.segment_id, q.source, q.question, g.start FROM review_questions q
                             JOIN segments g ON g.id = q.segment_id WHERE q.status = 'open' AND g.start >= ? AND g.start < ?
                             ORDER BY g.start""", (t0, t1)).fetchall()
        oqtxt = "\n".join(f"  #{q['id']} on block #{q['segment_id']} ({_t(q['start'])}, {q['source']}): {q['question']}"
                           for q in oq) or "  (none)"

        # Activity log: every block; when the day is long, drop the shortest blocks first.
        live = [s for s in segs if s["duration"] > 0]
        kept, floor = live, 0
        for floor in (0, 20, 45, 90, 180):
            kept = [s for s in live if s["duration"] >= floor or s.get("state") == "idle"]
            if len(kept) <= self.MAX_LOG_LINES:
                break
        note = (f"  (showing the {len(kept)} blocks of {len(live)} that lasted >= {floor}s; shorter ones omitted)\n"
                if len(kept) < len(live) else "")
        logtxt = note + "\n".join(f"#{s['id']} " + summaries.seg_line(s) for s in kept)

        text = (f"The day under discussion: {day.isoformat()} ({_t(t0, '%a %d %b')}; the logical day runs "
                f"{_t(t0)} to {_t(t1)} next morning).\n\n"
                f"Headline metrics:\n{metrics}\n\n"
                f"Hypothesized focus spans (the system's current calls):\n{spantxt}\n\n"
                f"Attention events:\n{evtxt}\n\n"
                f"Toggl entries (what they intended to work on):\n{tgtxt}\n\n"
                f"Open review questions already in their queue for this day (do not open these again):\n{oqtxt}\n\n"
                f"The whole day's activity log (#block id, time, length, state, app/site, title, <content label>, "
                f"<switch label into it>, [activity], (toggl)):\n{logtxt}")
        blocks.append({"type": "text", "text": text})
        return blocks

    def reply(self, conn, day, messages: list[dict]) -> dict:
        """One turn. Tool calls are executed as they come (each edit carries this reply's id as
        provenance), their results fed back, and the final text plus the list of applied changes
        is stored on the reply - so the person can see, open and undo exactly what happened."""
        from .classifiers.claude_vision import _cost
        mid = begin_exchange(conn, "day", day.isoformat(), messages, self.cfg.claude_model)
        applied: list[dict] = []
        proposal: dict = {}
        cost, text = 0.0, ""
        try:
            api_msgs = _api_messages(self._context_blocks(conn, day), messages)
            tools = DAY_TOOLS + [llm.CONTACT_TOOL]
            for _ in range(8):
                resp = self.client.messages.create(
                    model=self.cfg.claude_model, max_tokens=8192, system=self.system, messages=api_msgs,
                    tools=tools, output_config={"effort": "low"})
                cost += _cost(resp.usage)
                rec = llm.contacted("day_chat", self.cfg.claude_model, self.system, api_msgs, resp, tools)
                if rec:   # the model wants Eli: no tool runs, no further turns
                    text = llm.contact_notice(rec)
                    break
                if resp.stop_reason == "refusal":
                    text = "(Claude declined to answer that.)"
                    break
                text = "".join(b.text for b in resp.content if b.type == "text").strip() or text
                uses = [b for b in resp.content if b.type == "tool_use"]
                if not uses:
                    break
                results = [{"type": "tool_result", "tool_use_id": u.id,
                            "content": self._run_tool(conn, day, u.name, dict(u.input), mid, applied, proposal)}
                           for u in uses]
                api_msgs.append({"role": "assistant",
                                 "content": [b.model_dump(exclude_none=True) for b in resp.content]})
                api_msgs.append({"role": "user", "content": results})
        except Exception as e:
            if not applied:  # nothing happened: leave no trace of the failed turn
                abort_exchange(conn, mid)
                raise
            # Edits already carry this reply's id as provenance - keep the row, record the failure.
            finish_exchange(conn, mid, f"(error after {len(applied)} change(s) were applied: {e})", cost,
                            {**proposal, "applied_changes": applied})
            raise
        if not text:
            text = "(changes applied - see below)" if applied else "(no reply)"
        prop = ({**proposal, "applied_changes": applied} if (applied or proposal) else None)
        finish_exchange(conn, mid, text, cost, prop)
        log.info("day chat %s: %d change(s), $%.3f", day, len(applied), cost)
        return {"reply": text, "message_id": mid, "applied": applied, "proposal": prop}

    def _run_tool(self, conn, day, name: str, args: dict, mid: int, applied: list[dict], proposal: dict) -> str:
        import datetime as dt
        from . import edits, stats
        t0, t1 = stats.day_bounds(day)

        def _t(ts):
            return dt.datetime.fromtimestamp(ts).strftime("%H:%M")

        def _hhmm(s: str) -> float | None:
            try:
                h, m = (int(x) for x in s.strip().split(":")[:2])
            except (ValueError, AttributeError):
                return None
            base = dt.datetime.fromtimestamp(t0).replace(hour=h, minute=m, second=0, microsecond=0).timestamp()
            return base + 86400 if base < t0 else base  # after midnight belongs to the same logical day

        try:
            if name == "set_segment_judgment":
                ids = [int(x) for x in (args.get("segment_ids") or ([args["segment_id"]] if "segment_id" in args else []))]
                if not ids:
                    return "error: segment_ids is empty"
                parts = [f"{k} -> {args[k]}" for k in ("content", "switch_label", "interruption_kind", "antecedent_id",
                                                       "hybrid", "focus_exception") if args.get(k) is not None]
                note = (args.get("note") or "").strip() or None
                kind = args.get("interruption_kind") if args.get("switch_label") == "interruption" else None
                done, errs = [], []
                for sid in ids:
                    row = conn.execute("SELECT start, app FROM segments WHERE id=? AND start>=? AND start<?",
                                       (sid, t0, t1)).fetchone()
                    if row is None:
                        errs.append(f"#{sid} is not in this day's log")
                        continue
                    try:
                        derive = edits.write_seg_review(
                            conn, sid, content=args.get("content"), switch_label=args.get("switch_label"),
                            interruption_kind=kind, antecedent_id=args.get("antecedent_id"), hybrid=args.get("hybrid"),
                            note=note, chat_message_id=mid, focus_exception=args.get("focus_exception"),
                            merge_existing=True)
                    except edits.BadEdit as e:
                        errs.append(f"#{sid}: {e}")
                        continue
                    if derive:
                        try:
                            from .web import _kick_kind_derivation
                            _kick_kind_derivation(sid, note)
                        except Exception:  # not running under the server
                            pass
                    sw = conn.execute("SELECT id FROM switches WHERE to_segment=?", (sid,)).fetchone()
                    applied.append({"kind": "segment", "segment_id": sid, "switch_id": sw["id"] if sw else None,
                                    "when": _t(row["start"]), "app": row["app"],
                                    "summary": ", ".join(parts) or "note only", "note": note or ""})
                    done.append(f"#{sid} {_t(row['start'])}")
                out = f"ok: {len(done)} block(s) now {', '.join(parts) or 'unchanged labels'}: {', '.join(done)}" if done else "nothing changed"
                return out + (f"; errors: {'; '.join(errs)}" if errs else "")
            if name == "set_span_verdict":
                s0, e0 = _hhmm(args.get("start_hhmm", "")), _hhmm(args.get("end_hhmm", ""))
                if s0 is None or e0 is None:
                    return "error: start_hhmm/end_hhmm must be HH:MM as listed"
                segs = stats.labelled_segments(conn, t0, t1)
                se = stats.spans_and_events(segs, conn=conn)
                spans = se["focus_spans"] + se["disqualified"]
                sp = min(spans, key=lambda s: abs(s["start"] - s0) + abs(s["end"] - e0), default=None)
                if sp is None or abs(sp["start"] - s0) > 150 or abs(sp["end"] - e0) > 150:
                    return "error: no hypothesized span with those times; use a span from the list exactly"
                edits.write_span_review(conn, sp["start"], sp["end"], args["verdict"], args.get("subtype"),
                                        (args.get("note") or "").strip() or None, chat_message_id=mid)
                what = "not a focus span" if args["verdict"] == "not_focus" else f"focus span ({args.get('subtype') or 'subtype as judged'})"
                applied.append({"kind": "span", "start": sp["start"], "end": sp["end"], "summary": what,
                                "note": (args.get("note") or "").strip()})
                return f"ok: span {_t(sp['start'])}-{_t(sp['end'])} ruled {what}"
            if name == "open_review_question":
                sid = int(args["segment_id"])
                if not conn.execute("SELECT 1 FROM segments WHERE id=? AND start>=? AND start<?", (sid, t0, t1)).fetchone():
                    return f"error: block #{sid} is not in this day's log"
                qid = edits.open_question(conn, sid, "chat", args.get("context") or "", args.get("question") or "",
                                          list(args.get("options") or []), chat_message_id=mid)
                applied.append({"kind": "question", "question_id": qid, "segment_id": sid,
                                "question": (args.get("question") or "").strip()})
                return f"ok: review question #{qid} is in the queue (it shows as a flag on the timeline)"
            if name == "propose_rule":
                rule = (args.get("rule") or "").strip()
                if not rule:
                    return "error: empty rule"
                proposal["rule"] = rule
                return "ok: the rule is shown to the person with an 'add to my rules' button; it is NOT in the document until they click it"
            return f"error: unknown tool {name}"
        except edits.BadEdit as e:
            return f"error: {e}"
        except Exception as e:  # never let a tool failure kill the turn
            log.exception("day chat tool %s failed", name)
            return f"error: {type(e).__name__}: {e}"

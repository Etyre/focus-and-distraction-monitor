"""Interactive chats with Claude: about one switch, one focus span, or a whole day."""
from __future__ import annotations

import logging
import time

import anthropic

from . import LABEL_HELP, db, llm
from .seg_models import stored_switch
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


def _api_messages(ctx_blocks, messages, replays: dict | None = None) -> list[dict]:
    """The thread as the API wants it: the context blocks ride along with the first question.
    `replays` {message id: [api messages]} puts the tool calls and results of an earlier reply
    back in front of its text (chat_tools.replay)."""
    api_msgs: list[dict] = []
    for i, m in enumerate(messages):
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = (m.get("text") or "").strip()
        if i == 0 and role == "user":
            api_msgs.append({"role": "user",
                             "content": ctx_blocks + [{"type": "text", "text": "\n\nMy question: " + text}]})
        else:
            if role == "assistant" and replays and m.get("id") in replays:
                api_msgs += replays[m["id"]]
            api_msgs.append({"role": role, "content": text})
    return api_msgs


# 1024 was the old default. The model's thinking counts against max_tokens, so a question that
# made it think hard came back with no text at all: the span chat answered "(no reply)" on
# 2026-11-20 23:21 (message 84, $0.16 spent). 16,000 leaves room for thinking and an answer.
CHAT_MAX_TOKENS = 16000


def _run_chat(client, cfg, system, ctx_blocks, messages, max_tokens: int = CHAT_MAX_TOKENS) -> str:
    api_msgs = _api_messages(ctx_blocks, messages)
    if not api_msgs:
        return "", 0.0
    system = llm.with_preamble(system)
    tools = list(llm.CONTACT_TOOLS)
    from .classifiers.claude_vision import _cost
    costs, said = [], []   # one entry per request: a note_to_eli call is answered and the model asked again

    def call(msgs):
        resp = client.messages.create(
            model=cfg.claude_model, max_tokens=max_tokens,
            system=system,
            messages=msgs,
            tools=tools,
            output_config={"effort": cfg.claude_effort},   # was "low" written in (2026-11-20 audit)
        )
        costs.append(_cost(resp.usage))
        said.append("".join(b.text for b in resp.content if b.type == "text").strip())
        return resp

    api_msgs, resp = llm.with_notes("chat", cfg.claude_model, system, api_msgs, call(api_msgs), call, tools)
    cost = sum(costs)
    rec = llm.contacted("chat", cfg.claude_model, system, api_msgs, resp, tools)
    if rec:
        return llm.contact_notice(rec), cost
    if resp.stop_reason == "refusal":
        return "(Claude declined to answer that.)", cost
    # what it wrote in a turn that also left a note is part of the reply, not replaced by the next turn
    text = "\n\n".join(t for t in said if t)
    if not text:
        log.warning("chat call returned no text (stop_reason %s, output tokens %s)",
                    resp.stop_reason, getattr(resp.usage, "output_tokens", None))
        text = ("(Claude ran out of room before writing an answer. Please ask again.)"
                if resp.stop_reason == "max_tokens" else f"(no reply: stop reason {resp.stop_reason})")
    return text, cost


MAX_ROUNDS = 16      # tool rounds in one reply (8 until 2026-11-20, when every tool wrote and none read)


def _tool_turns(chat, conn, kind: str, ref: str, site: str, ctx_blocks, messages: list[dict], run_tool, day=None) -> dict:
    """One turn of a chat that can act. Tool calls are executed as they come (each edit carries
    this reply's id as provenance), their results fed back, and the final text plus the list of
    applied changes is stored on the reply - so the person can see, open and undo exactly what
    happened. Shared by the day chat and the span chat.

    Since 2026-11-20 the chats can also look things up (chat_tools.READ_TOOLS, run through the
    same `run_tool`), and every tool call with its result is stored with the reply
    (chat_tools.TRACE_KEY) and sent again on later turns: before, only the text of earlier
    turns was, so the model no longer had what it had read or the record of its own edits.
    `day` is the logical day the chat is about (for re-running a screenshot lookup on replay)."""
    from . import chat_tools as ct
    from .classifiers.claude_vision import _cost
    cfg = chat.cfg
    mid = begin_exchange(conn, kind, ref, messages, cfg.claude_model)
    applied: list[dict] = []
    proposal: dict = {}
    trace: list[dict] = []
    cost, text = 0.0, ""
    resp = None

    def stored():
        p = dict(proposal)
        if applied:
            p["applied_changes"] = applied
        if trace:
            p[ct.TRACE_KEY] = trace
        return p or None
    try:
        api_msgs = _api_messages(ctx_blocks, messages,
                                 ct.replay(conn, cfg, ct.stored_traces(conn, messages), day) if day is not None else None)
        tools = DAY_TOOLS + ct.READ_TOOLS + llm.CONTACT_TOOLS
        kept = ""   # text from turns whose only tool call was a note to Eli: still part of the reply
        read_chars = 0
        for _ in range(MAX_ROUNDS):
            resp = chat.client.messages.create(
                model=cfg.claude_model, max_tokens=CHAT_MAX_TOKENS, system=chat.system, messages=api_msgs,
                tools=tools, output_config={"effort": cfg.claude_effort})
            cost += _cost(resp.usage)
            rec = llm.contacted(site, cfg.claude_model, chat.system, api_msgs, resp, tools)
            if rec:   # the model wants Eli: no tool runs, no further turns
                text = llm.contact_notice(rec)
                break
            if resp.stop_reason == "refusal":
                text = "(Claude declined to answer that.)"
                break
            said = "".join(b.text for b in resp.content if b.type == "text").strip()
            text = "\n\n".join(t for t in (kept, said) if t) or text
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                break
            kept = text if all(u.name == llm.NOTE_TOOL_NAME for u in uses) else ""
            results, calls = [], []
            for u in uses:
                if u.name in ct.READ_NAMES and read_chars > ct.TURN_TEXT_MAX:
                    out = (f"not run: the lookups in this reply have already returned {read_chars:,} characters, the most one "
                           "reply may read. Answer from what you have, or say what you would look at next.")
                else:
                    out = run_tool(conn, u.name, dict(u.input), mid, applied, proposal)
                if u.name in ct.READ_NAMES:
                    read_chars += ct.size(out)
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
                if u.name != llm.NOTE_TOOL_NAME:    # (a note is in the inbox; a turn that only left one keeps its text)
                    calls.append(ct.trace_entry(u.id, u.name, dict(u.input), out))
            if calls:
                trace.append({"said": said, "calls": calls})
            api_msgs.append({"role": "assistant",
                             "content": [b.model_dump(exclude_none=True) for b in resp.content]})
            api_msgs.append({"role": "user", "content": results})
    except Exception as e:
        if not applied:  # nothing happened: leave no trace of the failed turn
            abort_exchange(conn, mid)
            raise
        # Edits already carry this reply's id as provenance - keep the row, record the failure.
        finish_exchange(conn, mid, f"(error after {len(applied)} change(s) were applied: {e})", cost, stored())
        raise
    if not text:
        if applied:
            text = "(changes applied - see below)"
        else:
            sr = getattr(resp, "stop_reason", None)
            log.warning("%s: no text (stop_reason %s)", site, sr)
            text = ("(Claude ran out of room before writing an answer. Please ask again.)"
                    if sr == "max_tokens" else f"(no reply: stop reason {sr})")
    prop = ({**proposal, "applied_changes": applied} if (applied or proposal) else None)
    finish_exchange(conn, mid, text, cost, stored())    # with the tool trace, which the page is not sent
    log.info("%s %s: %d change(s), %d tool round(s), $%.3f", site, ref, len(applied), len(trace), cost)
    return {"reply": text, "message_id": mid, "applied": applied, "proposal": prop}


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
    def shown(js):
        # what a reply looked up with its read tools (chat_tools) is for the next turn's replay, not for the page
        p = {k: v for k, v in _json.loads(js).items() if k != "tool_trace"} if js else None
        return p or None
    return [{"id": r["id"], "role": r["role"], "text": r["text"], "created": r["created"], "proposal": shown(r["proposal"])}
            for r in conn.execute(
        "SELECT id, role, text, created, proposal FROM chat_messages WHERE kind=? AND ref=? ORDER BY created, id", (kind, ref))]


SPAN_SYSTEM = """You are helping this person review their own attention-monitoring log. They are looking at ONE \
hypothesized focus span and deciding (a) whether it really was a focus span at all, and (b) which kind. \
You see the WHOLE DAY's activity log with the span under review marked in it - so you can reason about what \
led into the span, what followed it, and whether its boundaries are right - plus the day's other spans, the \
day's Toggl entries, what the system worked out about this span (its call, how it ended, the diversions inside \
it), its summary and coherence check, the person's verdict if they gave one, the review questions asked about its \
blocks, and screenshots from inside it: one of each app or site in it and of each block over a minute, as far as \
the allowance reaches. Every log line carries the evaluator's note in full, what the block continues, the \
person's own rulings and notes, and the signals the Mac recorded.

Answer their questions about THIS span concisely and concretely - point at specific blocks, times, and \
what's visible on screen. If they ask for a verdict, say whether you'd call it a focus span, which subtype, \
and why - and note honestly when it's genuinely ambiguous. Their calibration feedback is that the system \
has been far too optimistic about what counts as focused work; do not repeat that mistake. Don't lecture; \
be a sharp, brief thinking partner.

{act}
In this chat the blocks are the ones in the day's log below (numbered #id), and set_span_verdict takes the span under review by its start/end times as shown at the top. A verdict on the span is the most direct way to record "this was / was not a focus span"; use it when the person says so.

Span subtypes: creative_work, focused_task, reading, planning, other_focused, meeting. (Correspondence - chat, email - \
is interactive, never a focus span on its own; only email processed inside a Toggl-covered task block belongs to that span.)

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""


def _legend() -> str:
    from . import prompt_parts as pp
    return pp.LOG_LEGEND


class SpanChat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        from .classifiers.claude_vision import _rules_doc
        self.system = llm.with_preamble(SPAN_SYSTEM.format(
            act=ACT_DOC.format(legend=_legend()), rules=_rules_doc(), about_me=cfg.about_me.strip() or "(no description given)"))

    def _context_blocks(self, conn, start: float, end: float) -> list[dict]:
        """The span, its day, and what is on record about both. Rewritten 2026-11-20 with the day
        chat's: before, blocks further than 120 from the span were elided (291 earlier and 19
        later on 2026-11-19), lines were the short form, notes appeared only inside the span,
        there were four screenshots, and neither the coherence check nor the person's verdict
        nor any review question was shown."""
        import json as _json
        from . import evaluator as ev, prompt_parts as pp, stats, summaries
        from .config import logical_date
        day = logical_date(start)
        t0, t1 = stats.day_bounds(day)
        day_segs = stats.labelled_segments(conn, t0, t1)
        se = stats.spans_and_events(day_segs, conn=conn)
        mid = (start + end) / 2
        sp = next((s for s in se["focus_spans"] + se["disqualified"]
                   if s["start"] - 1 <= mid < s["end"] + 1), None)
        if sp is None:
            raise ValueError("no span at that time")
        _t = lambda ts, fmt="%H:%M": pp.t(ts, fmt)      # the zone the Mac was in then
        _, rows = pp.log_lines(conn, self.cfg, t0, t1, phone=False)
        inside = [g for g in rows if g["clip_end"] > sp["start"] and g["clip_start"] < sp["end"]]
        blocks, shots_note = summaries.span_images(conn, inside)
        away = {g["id"] for r in sp["runs"] if r["cat"] == "detour" for g in r["segments"]}
        marks = {g["id"]: (f"#{g['id']} [IN THE SPAN, a contained diversion] " if g["id"] in away else f"#{g['id']} [IN THE SPAN] ")
                 for g in inside}
        qrows = conn.execute("""SELECT q.*, g.start AS g_start FROM review_questions q JOIN segments g ON g.id = q.segment_id
                                WHERE g.start >= ? AND g.start < ? AND q.status IN ('open', 'answered', 'unsure', 'none')
                                ORDER BY g.start, q.id""", (sp["start"] - 600, sp["end"] + 600)).fetchall()
        qtxt, qnotes = [], {}
        for q in qrows:
            try:
                opts = _json.loads(q["options"] or "[]")
            except ValueError:
                opts = []
            qtxt.append(f"  [{q['status'].upper()}] on block #{q['segment_id']} ({_t(q['g_start'])}):\n"
                        + pp.question_text(q, opts, q["answer_index"], indent="    "))
            qnotes.setdefault(q["segment_id"], []).append(f"review question #{q['id']} ({q['status']}): {pp.one(q['question'])}"
                                                           + (f" -> answered “{pp.one(q['answer'])}”" if q["status"] == "answered" else ""))
        lines, shown = pp.log_lines(conn, self.cfg, t0, t1, marks=marks, notes=qnotes, default_mark="#{id} ", dated=True)
        row = summaries.lookup(conn, sp["start"], sp["end"])
        verdicts = []
        for r in conn.execute("SELECT * FROM span_reviews WHERE start < ? AND end > ? ORDER BY created", (sp["end"], sp["start"])):
            verdicts.append(f"  on {_t(r['start'])}-{_t(r['end'])}, made {_t(r['created'], '%Y-%m-%d %H:%M')}"
                            + (", through a chat" if r["chat_message_id"] else "") + ": "
                            + ("not a focus span" if r["verdict"] == "not_focus" else f"a focus span ({r['subtype'] or 'subtype as judged'})")
                            + (f"; their note: “{pp.one(r['note'])}”" if pp.one(r["note"]) else ""))
        from . import exceptions
        exc = exceptions.lines(conn)
        overview = "\n".join(
            f"  {_t(s2['start'])}-{_t(s2['end'])}: "
            + (f"DISQUALIFIED ({s2.get('disq_reason')})" if s2.get("disq_reason") else s2.get("subtype", "?"))
            + f", {s2['focus_min']:.0f} min focused"
            + ("  <== THE SPAN UNDER REVIEW" if s2 is sp else "")
            for s2 in sorted(se["focus_spans"] + se["disqualified"], key=lambda s2: s2["start"]))
        tgtxt = "\n".join(ev._toggl_lines(conn, t0, t1, sp["start"], sp["end"], mark="span under review")[0]) or "  (no Toggl entries at all this day)"
        text = (f"The span under review: {_t(sp['start'], '%a %Y-%m-%d %H:%M')}-{_t(sp['end'])} "
                f"(for set_span_verdict: start {_t(sp['start'])}, end {_t(sp['end'])}); times are in the zone the Mac was in "
                f"that day. This message is written {_t(time.time(), '%Y-%m-%d %H:%M')}.\n"
                "What the system worked out about it:\n" + "\n".join("  " + x for x in summaries.span_facts(sp)) + "\n"
                + (f"Its summary" + (f" (“{pp.one(row['title'])}”)" if row["title"] else "") + f": {pp.one(row['summary'])}\n"
                   + ("" if row["coherent"] else f"The summary's own check doubted that it is one object of attention: {pp.one(row['issue'])}\n")
                   if row else "Its summary: (none yet)\n")
                + "The person's verdict on it: " + ("\n" + "\n".join(verdicts) if verdicts else "none yet") + "\n"
                + ("Exceptions to the written rules the person has made so far (any day; say so when this span "
                   "resembles one):\n" + "\n".join("  " + x for x in exc) + "\n" if exc else "") + "\n"
                + shots_note + " get_screenshots shows any block.\n\n"
                f"All hypothesized spans this day:\n{overview}\n\n"
                f"Toggl entries this day (all of them, times to the second):\n{tgtxt}\n\n"
                "Review questions about blocks in or within ten minutes of the span, with the person's answers "
                "(do not open again one that is OPEN):\n" + ("\n".join(qtxt) or "  (none)") + "\n\n"
                f"The whole day's activity log: all {len(shown)} blocks, none left out; the span's own blocks are marked "
                "[IN THE SPAN]:\n" + "\n".join(lines))
        blocks.append({"type": "text", "text": text})
        return blocks

    def reply(self, conn, start: float, end: float, messages: list[dict]) -> dict:
        """One turn, with the day chat's tools (Eli, 2026-11-20: "The span chat should have edit
        access not just the ability to talk" - he told it a span should count as a focus span
        and nothing was recorded). Edits carry this reply's id as provenance, like the day chat's."""
        from .config import logical_date
        day = logical_date(start)
        doer = DayChat.__new__(DayChat)          # its tool handlers only need the config
        doer.cfg = self.cfg
        return _tool_turns(self, conn, "span", str(round(start)), "span_chat",
                           self._context_blocks(conn, start, end), messages,
                           lambda c, name, args, mid, applied, proposal:
                               doer._run_tool(c, day, name, args, mid, applied, proposal), day=day)


ACT_DOC = """You can LOOK THINGS UP, and should before asserting what a block was or why it was judged as it was:
- get_screenshots: the captures of a block, or of a stretch of time. The screenshots you start with are a \
sample; look at the block itself when it matters.
- get_evaluation: the evaluation run behind a block's judgment - what the evaluator was shown and its answer.
- get_segment_detail: everything on record about one block (probabilities, ruling, questions, earlier chat).
- get_log: the activity log of another day, or of a time range.
These only read. What you looked up in earlier turns of this conversation is still in it.

You can also ACT, but only through the tools:
- set_segment_judgment: change the judgment of one or more activity blocks (blocks are numbered #id in the \
  log). This writes the person's authoritative review of each block, flagged as coming from this conversation.
- set_span_verdict: rule on a hypothesized span (by its start/end times from the span list).
- open_review_question: put a question in the person's review queue for later, with options that each map to \
  a concrete judgment - use this when a call needs their eyes and they do not want to settle it now.
- propose_rule: suggest a general rule for their rules document (they approve it with a click).
EXCEPTIONS to the written rules: the person may rule against what the rules say ("this should be a focus \
span even though it did not follow the rule, because ...", or the reverse). Make the ruling they ask for; the \
rules do not outrank them. Pass exception_to_rule naming the rule, and put THEIR reason in the note; if they \
gave no reason, make the ruling and ask once why (a later ruling on the same span or blocks adds it). The app \
keeps a record of exceptions and drafts a general rule when several share a reason, so do not also \
propose_rule for a single exception unless they ask for a rule. Only what THEY call an exception is one: a \
ruling that corrects a misjudgment (the app read the screen wrong, or applied a rule wrongly) is an ordinary \
correction, so leave exception_to_rule null for it.
How the app turns these labels into focus spans - you must work with this:
- Every block has a focused/unfocused judgment (the QUALITY of attention in that block), separate from \
which thread it belongs to. A focused block counts as focus; an unfocused one does not. Passive content is \
never focus; correspondence content with no Toggl entry is not focus - UNLESS the person explicitly rules a \
block an exception (focus_exception=true: "this was correspondence and it still counts as independent \
focus"); never set that on your own judgment, only when they ask for it.
- Threads come from the switch labels: a 'shift' block starts a new thread, a 'continuation' or 'return' block \
belongs to the thread of its antecedent ("<continues id=N>" on its line). A focus span is focused time on ONE thread, and it has to start with \
10 continuous focused minutes. Time on a different thread inside a span is a diversion whether or not it was \
focused (a quick purposeful purchase is a focused block and still a diversion); under 5 minutes it is a \
contained interruption, at 5 minutes the span ended where it began.
- Unfocused blocks on the span's OWN thread count as focus time while the unfocused judgment is only the \
evaluator's ("I may still be thinking"). When the PERSON rules a block unfocused (focused=false set here), it \
becomes time away from the span.
- On a shift, self_distraction=true marks it as seeking stimulation without a clear intent (a planned break is \
not one), and next_task=true marks it as the next task in a focused task block (the app joins those threads \
into one block when a Toggl entry is running).
- Blocks judged before 2026-11-19 have no focused judgment; for them a 'continuation' inherits the state of \
the block before it, a 'return' is focus, and a 'shift' is focus, a detour or a distraction according to an \
older kind label. Setting focused on such a block switches it to the rules above.
So to make a stretch count as focused work: set focused=true on its blocks (only when the person has said it \
was focused work), give them the right content, link them into one thread (first block switch_label=shift, the \
rest continuation), and after any block you leave out as a diversion mark the next block of the thread as \
'return' with antecedent_id = an earlier block of that thread. Hypothesized spans are then recomputed from \
the blocks; you cannot create a span directly.
Make a change only when the person has clearly asked for it or agreed to it; discuss first otherwise. Edit \
every block the agreed change covers - the person expects the timeline to match what was agreed. Afterwards, state exactly what you changed. NEVER say you changed, queued, flagged or applied anything \
unless a tool call in this reply actually did it; if a tool reports an error, say so. The log shows every block's \
CURRENT labels (a block the person has ruled on says RULED BY THE PERSON, with the fields they set and their \
note): do not redo edits that are already in \
place, and do not open a review question that is already in the queue. A field you leave null on \
set_segment_judgment keeps the person's earlier ruling on that block. You cannot edit the rules \
document directly or change anything the tools do not cover - say so when asked.

{legend}
In these logs a block's mark is "#id". PHONE lines cannot be edited with the tools.
"""

DAY_SYSTEM = """You are helping this person review their own attention-monitoring log. They are looking at ONE WHOLE \
DAY and want to talk it over with you: how the day went, where their attention held and where it broke, whether \
the system's calls look right, what the day's shape suggests for tomorrow.

You see the day's headline metrics, every hypothesized focus span (qualifying and disqualified) with the system's \
call, its summary, its own coherence check and the person's verdict on it, the self-distraction and interruption \
events, the day's Toggl "intent" entries, every review question asked about the day with the person's answer, the \
whole day's activity log - every block, with the evaluator's note in full, what the block continues, the person's \
own rulings and notes, and the signals the Mac recorded - and a few screenshots sampled across the day.

Answer concisely and concretely - point at specific times, spans, and blocks. When asked for an assessment, give \
one and say what it rests on; note honestly when the data is thin or a call is ambiguous. Their calibration \
feedback is that the system has been far too optimistic about what counts as focused work; do not repeat that \
mistake. Don't lecture; be a sharp, brief thinking partner.

{act}
Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""

_LABEL_DOC = ("content = what the block was: creative | task | correspondence | reading | planning | other | passive "
              "| meeting | idle | transition. switch_label = how it relates to the thread before it: continuation "
              "(same thread) | shift (attention moved to something new: a separate thread began) | return (back to an earlier thread; give "
              "antecedent_id). focused = the quality of attention in the block: true focused, false unfocused. "
              "self_distraction (on a shift, or a return to a distraction) = sought out stimulation without a clear intent. "
              "next_task (on a shift) = the next task under the same task-processing intention. "
              "hybrid = the activity was carried on inside a live discussion. null leaves a field as judged.")

_EXC_DOC = ("Set ONLY when the person says this ruling goes against their written rules and should stand anyway "
            "(\"I know the rule says otherwise, but this one counts\", or the reverse): the rule it departs from, quoted "
            "or closely paraphrased from the rules document. The ruling is then also kept in the record of exceptions. "
            "null for an ordinary correction of a misjudgment.")

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
         "switch_label": {"type": ["string", "null"], "enum": ["continuation", "shift", "return", None]},
         "focused": {"type": ["boolean", "null"],
                     "description": "The person's ruling on the quality of attention in these blocks: true = "
                                    "focused, false = unfocused (which makes the block time away from a span "
                                    "even on its own thread). Set it only when they have said so; null leaves "
                                    "the evaluator's judgment."},
         "self_distraction": {"type": ["boolean", "null"],
                              "description": "With switch_label=shift only: true = they sought out stimulation "
                                             "or distraction without a clear intent; false = a deliberate move."},
         "next_task": {"type": ["boolean", "null"],
                       "description": "With switch_label=shift only: true = this was the next task under the "
                                      "same task-processing intention (part of a focused task block)."},
         "antecedent_id": {"type": ["integer", "null"]},
         "hybrid": {"type": ["boolean", "null"]},
         "focus_exception": {"type": ["boolean", "null"],
                             "description": "true ONLY when the person explicitly asked that this block count as "
                                            "independent focus although its content (correspondence with no Toggl "
                                            "entry) normally cannot. Never set it on your own judgment; null = no "
                                            "ruling; false withdraws an earlier ruling."},
         "note": {"type": "string", "description": "One sentence in the person's words: what was really going on and "
                                                   "why it is labelled this way. Stored as the precedent future "
                                                   "judgments see."},
         "exception_to_rule": {"type": ["string", "null"], "description": _EXC_DOC}},
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
         "note": {"type": "string", "description": "The person's reason, in their words. For an exception this "
                                                   "is the record a later general rule is drafted from."},
         "exception_to_rule": {"type": ["string", "null"], "description": _EXC_DOC}},
         "required": ["start_hhmm", "end_hhmm", "verdict", "note"]}},
    {"name": "open_review_question",
     "description": "Put a question in the person's review queue to answer later. Each option maps to a concrete "
                    "judgment of a block (a neutral 'just passing through' option is added automatically). Use "
                    "when a call needs their eyes and they do not want to settle it now. An answer changes the "
                    "record only through its option's fields, so set every field the option's wording asserts. " + _LABEL_DOC,
     "input_schema": {"type": "object", "properties": {
         "segment_id": {"type": "integer", "description": "The block whose judgment is in doubt."},
         "context": {"type": "string", "description": "1-2 sentences setting the scene (what happened around it)."},
         "question": {"type": "string"},
         "options": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
             "label": {"type": "string", "description": "The reading, in the person's own terms."},
             "segment_id": {"type": ["integer", "null"], "description": "The block this option edits (default: the question's)."},
             "content": {"type": ["string", "null"]}, "switch_label": {"type": ["string", "null"]},
             "antecedent_id": {"type": ["integer", "null"]}, "hybrid": {"type": ["boolean", "null"]},
             # 2026-11-20: an option could not say focused, drifted or next task, so "that was me
             # drifting" could not be offered as an answer (edits.open_question kept `focused` when
             # sent, but the schema never offered it).
             "focused": {"type": ["boolean", "null"], "description": "true when this reading says they were getting something specific done; false when it says they were not (a drift). null when it does not say."},
             "self_distraction": {"type": ["boolean", "null"], "description": "For a block that is, or that this option makes, a shift or a return: true = this reading says they drifted there without deciding to; false = it says they went on purpose. null when it does not say."},
             "next_task": {"type": ["boolean", "null"], "description": "With switch_label=shift: true = this reading says it was the next task under the same task-processing intention."}},
             "required": ["label"]}}},
         "required": ["segment_id", "context", "question", "options"]}},
    {"name": "propose_rule",
     "description": "Propose ONE general rule for the person's rules document, for them to approve with a click. "
                    "Written in the second person as the document is ('If I ...').",
     "input_schema": {"type": "object", "properties": {"rule": {"type": "string"}}, "required": ["rule"]}},
]


class DayChat:
    """Chat about a whole day. Context is the full day: metrics, spans, events, Toggl, the review
    questions and their answers, the activity log (every block) and screenshots sampled across
    the day; the read tools of chat_tools fetch the rest."""

    START_IMAGES = 12    # screenshots attached up front; get_screenshots shows any block

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        from .classifiers.claude_vision import _rules_doc
        self.system = llm.with_preamble(DAY_SYSTEM.format(act=ACT_DOC.format(legend=_legend()),
            rules=_rules_doc(), about_me=cfg.about_me.strip() or "(no description given)"))

    def _context_blocks(self, conn, day) -> list[dict]:
        """What every turn starts from. Rewritten 2026-11-20 (Eli: every model call is given all
        the information that could be relevant). Before: when a day had more than 450 blocks the
        shortest were dropped - on 2026-11-18 that hid 334 of 665 (everything under 45 s),
        among them 100 shifts and returns, 24 self-distractions and 31 blocks with an open
        review question, while the instructions said to edit every block a change covers; no
        line carried the evaluator's note, the person's note or what the block continues; only
        open questions were listed; times were in the Mac's zone of today."""
        import json as _json
        from . import evaluator as ev, prompt_parts as pp, shots as _shots, stats, summaries
        t0, t1 = stats.day_bounds(day)
        segs = stats.labelled_segments(conn, t0, t1)
        se = stats.spans_and_events(segs, conn=conn)
        m = stats.daily_metrics(conn, day)
        _t = pp.t

        # Screenshots: a sample spread over the active part of the day, labelled with the block
        # they belong to (before: six first frames labelled with a time and an app only).
        real = [s for s in stats.labelled_segments(conn, t0, t1, phone=False) if s["duration"] > 0 and not s["idle"]]
        mom = _shots.moments(conn, [s["id"] for s in real])
        have = [s for s in real if mom.get(s["id"])]
        images = [im for s in _shots.spread(have, self.START_IMAGES)
                  for im in ev._moment_images(mom[s["id"]][0], "block", s["app"] or "", both=False)]
        blocks, meta = pp.image_blocks(images)

        metrics = (f"  active {m['active_min']:.0f} min; independent focus {m['total_focus_min']:.0f} min "
                   f"(longest span {m['longest_focus_min']:.0f} min, longest fully-absorbed {m['longest_absorbed_min']:.0f} min, "
                   f"{m['fully_absorbed_spans']} fully-absorbed span(s)); meetings/hybrid {m['meeting_focus_min']:.0f} min; "
                   f"correspondence {m['correspondence_min']:.0f} min; passive consumption {m['passive_min']:.0f} min; "
                   f"distracted {m['distracted_min']:.0f} min; interruptions inside spans {m['interruption_count']} "
                   f"(bounce-aways, i.e. leaving a focus span by a self-distraction: {m['bounce_aways']}; self-distraction events all day: {m['self_distraction_events']}); "
                   f"{m['switches']} window switches; {m['short_breaks']} short inactivity breaks "
                   f"({m['short_break_min']:.0f} min).")

        reviews = conn.execute("SELECT * FROM span_reviews WHERE start < ? AND end > ? ORDER BY start", (t1, t0)).fetchall()

        def _span_line(sp):
            call = (f"DISQUALIFIED ({sp.get('disq_reason')})" if sp.get("disq_reason")
                    else (sp.get("subtype") or "?") + (" [hybrid]" if sp.get("hybrid") else "")
                    + (" [fully absorbed]" if sp.get("fully_absorbed") else ""))
            row = summaries.lookup(conn, sp["start"], sp["end"])
            line = (f"  {_t(sp['start'], '%H:%M')}-{_t(sp['end'], '%H:%M')}  {call}: {summaries.dur_text(sp['duration'])}, "
                    f"{sp['focus_min']:.0f} min focused, {len(sp.get('detours', []))} contained diversion(s)"
                    + (f" ({sp.get('detour_min', 0):.1f} min)" if sp.get("detours") else "")
                    + f", ended by {sp.get('ended_by')}")
            for txt in summaries.span_facts(sp)[1:]:
                line += f"\n      {txt}"
            if row and row["summary"]:
                line += "\n      summary" + (f" (“{pp.one(row['title'])}”)" if row["title"] else "") + f": {pp.one(row['summary'])}"
                if not row["coherent"]:
                    line += f"\n      the summary's own check doubted this is one object of attention: {pp.one(row['issue'])}"
            for r in reviews:
                ov = min(sp["end"], r["end"]) - max(sp["start"], r["start"])
                if ov > 0 and ov >= 0.5 * min(sp["end"] - sp["start"], r["end"] - r["start"]):
                    line += (f"\n      THE PERSON'S VERDICT on {_t(r['start'], '%H:%M')}-{_t(r['end'], '%H:%M')} (made {_t(r['created'], '%Y-%m-%d %H:%M')}"
                             + (", through this chat" if r["chat_message_id"] else "") + "): "
                             + ("not a focus span" if r["verdict"] == "not_focus" else f"a focus span ({r['subtype'] or 'subtype as judged'})")
                             + (f"; their note: “{pp.one(r['note'])}”" if pp.one(r["note"]) else ""))
            return line

        spans = sorted(se["focus_spans"] + se["disqualified"], key=lambda s: s["start"])
        spantxt = "\n".join(_span_line(sp) for sp in spans) or "  (no spans hypothesized)"

        def _ev_line(e, kind):
            where = ", ".join(e.get("where") or []) if isinstance(e.get("where"), list) else (e.get("where") or "")
            return (f"  {_t(e['start'])}  {kind}, {summaries.dur_text(e['duration'])}"
                    + (" inside a focus span" if e.get("inside_span") else "") + (f": {where}" if where else ""))

        events = sorted([(e["start"], _ev_line(e, "self-distraction")) for e in se["distractions"]]
                        + [(e["start"], _ev_line(e, "interruption")) for e in se["interruptions"]])
        evtxt = "\n".join(l for _, l in events) or "  (none)"

        tgtxt = "\n".join(ev._toggl_lines(conn, t0, t1, None, None)[0]) or "  (no Toggl entries this day)"

        # Every question about the day, with what the person answered (before: the open ones only,
        # so the chat could not see that something had been asked and settled, or left as "can't tell").
        qrows = conn.execute("""SELECT q.*, g.start AS g_start FROM review_questions q JOIN segments g ON g.id = q.segment_id
                                WHERE g.start >= ? AND g.start < ? ORDER BY g.start, q.id""", (t0, t1)).fetchall()
        qtxt, others, qnotes = [], {}, {}
        for q in qrows:
            if q["status"] in ("open", "answered", "unsure", "none"):
                try:
                    opts = _json.loads(q["options"] or "[]")
                except ValueError:
                    opts = []
                qtxt.append(f"  [{q['status'].upper()}] on block #{q['segment_id']} ({_t(q['g_start'], '%H:%M')}):\n"
                            + pp.question_text(q, opts, q["answer_index"], indent="    "))
                qnotes.setdefault(q["segment_id"], []).append(
                    f"review question #{q['id']} ({q['status']}): {pp.one(q['question'])}"
                    + (f" -> answered “{pp.one(q['answer'])}”" if q["status"] == "answered" else ""))
            else:
                others.setdefault(q["status"], []).append(q)
        for st in ("dismissed", "archived"):
            for q in others.get(st, []):
                qtxt.append(f"  [{st.upper()}, never answered] #{q['id']} on block #{q['segment_id']} ({_t(q['g_start'], '%H:%M')}): {pp.one(q['question'])}"
                            + (f" ({pp.one(q['note'])})" if pp.one(q["note"]) else ""))
        if others.get("declined"):
            qtxt.append(f"  ({len(others['declined'])} more doubt(s) were looked at by the question writer and no question was asked: blocks "
                        + ", ".join(f"#{q['segment_id']}" for q in others["declined"]) + ")")

        lines, shown = pp.log_lines(conn, self.cfg, t0, t1, notes=qnotes, default_mark="#{id} ", dated=True)

        text = (f"The day under discussion: {day.isoformat()} ({_t(t0, '%a %d %b')}; the logical day runs "
                f"{_t(t0, '%H:%M')} to {_t(t1, '%H:%M')} next morning; times are in the zone the Mac was in that day). "
                f"This message is written {_t(time.time(), '%Y-%m-%d %H:%M')}.\n\n"
                f"Screenshots attached above: {len(meta)}, a sample spread over the day's {len(have)} blocks that have a "
                f"capture. get_screenshots shows any block.\n\n"
                f"Headline metrics:\n{metrics}\n\n"
                f"Hypothesized focus spans (the system's current calls):\n{spantxt}\n\n"
                f"Attention events:\n{evtxt}\n\n"
                f"Toggl entries (what they intended to work on; all of the day's, times to the second):\n{tgtxt}\n\n"
                f"Review questions about this day's blocks, with the person's answers (do not open again one that is OPEN):\n"
                + ("\n".join(qtxt) or "  (none)") + "\n\n"
                f"The whole day's activity log: all {len(shown)} blocks, none left out:\n" + "\n".join(lines))
        blocks.append({"type": "text", "text": text})
        return blocks

    def reply(self, conn, day, messages: list[dict]) -> dict:
        return _tool_turns(self, conn, "day", day.isoformat(), "day_chat", self._context_blocks(conn, day), messages,
                           lambda c, name, args, mid, applied, proposal:
                               self._run_tool(c, day, name, args, mid, applied, proposal), day=day)

    def _run_tool(self, conn, day, name: str, args: dict, mid: int, applied: list[dict], proposal: dict):
        import json as _json
        from . import chat_tools as ct, edits, prompt_parts as pp, stats
        t0, t1 = stats.day_bounds(day)
        _t = lambda ts: pp.t(ts, "%H:%M")       # the zone the Mac was in then
        _hhmm = lambda s: ct.clock(day, s)

        try:
            if name == llm.NOTE_TOOL_NAME:   # already recorded by llm.contacted (check); the turn carries on
                return llm.NOTE_RESULT_TEXT
            if name in ct.READ_NAMES:
                return ct.run(conn, self.cfg, name, args, day)
            if name == "set_segment_judgment":
                # the model is shown 'shift'; the stored name is unchanged
                args = dict(args, switch_label=stored_switch(args.get("switch_label")))
                ids = [int(x) for x in (args.get("segment_ids") or ([args["segment_id"]] if "segment_id" in args else []))]
                if not ids:
                    return "error: segment_ids is empty"
                parts = [f"{k} -> {args[k]}" for k in ("content", "switch_label", "focused", "self_distraction",
                                                       "next_task", "antecedent_id", "hybrid", "focus_exception")
                         if args.get(k) is not None]
                note = (args.get("note") or "").strip() or None
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
                            antecedent_id=args.get("antecedent_id"), hybrid=args.get("hybrid"),
                            note=note, chat_message_id=mid, focus_exception=args.get("focus_exception"),
                            focused=args.get("focused"), self_distraction=args.get("self_distraction"),
                            next_task=args.get("next_task"), merge_existing=True)
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
                exc_rule = (args.get("exception_to_rule") or "").strip()
                if exc_rule and done:
                    from . import exceptions
                    exceptions.record_segments(conn, [int(d.split()[0][1:]) for d in done],
                                               ", ".join(parts) or "note only", note, exc_rule, chat_message_id=mid)
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
                from . import exceptions
                eid = exceptions.record_span(conn, sp["start"], sp["end"], args["verdict"], args.get("subtype"),
                                             sp.get("own_call"), bool((args.get("exception_to_rule") or "").strip()),
                                             reason=args.get("note"),
                                             rule=args.get("exception_to_rule"), chat_message_id=mid)
                what = "not a focus span" if args["verdict"] == "not_focus" else f"focus span ({args.get('subtype') or 'subtype as judged'})"
                applied.append({"kind": "span", "start": sp["start"], "end": sp["end"], "summary": what,
                                "note": (args.get("note") or "").strip()})
                return (f"ok: span {_t(sp['start'])}-{_t(sp['end'])} ruled {what}"
                        + (" (kept in the record of exceptions to the rules)" if eid else ""))
            if name == "open_review_question":
                sid = int(args["segment_id"])
                if not conn.execute("SELECT 1 FROM segments WHERE id=? AND start>=? AND start<?", (sid, t0, t1)).fetchone():
                    return f"error: block #{sid} is not in this day's log"
                asked = list(args.get("options") or [])
                qid = edits.open_question(conn, sid, "chat", args.get("context") or "", args.get("question") or "",
                                          asked, chat_message_id=mid)
                # edits.open_question keeps content, switch, antecedent, hybrid and focused of an
                # option and drops the rest. The two shift judgments are put back here, as the
                # question writer stores them (self_distraction not on a continuation; next_task
                # only on a shift), so the answer writes what the option's wording says.
                extra = {}
                for o in asked:
                    sw = stored_switch(o.get("switch_label"))
                    e = {"self_distraction": o.get("self_distraction") if sw != "continuation" else None,
                         "next_task": o.get("next_task") if sw == "interruption" else None}
                    if any(v is not None for v in e.values()):
                        extra[((o.get("label") or "").strip(), int(o.get("segment_id") or sid))] = e
                if extra:
                    row = conn.execute("SELECT options FROM review_questions WHERE id=?", (qid,)).fetchone()
                    opts = _json.loads(row["options"])
                    for o in opts:
                        o.update({k: bool(v) for k, v in extra.get((o.get("label"), o.get("segment_id")), {}).items() if v is not None})
                    with db.tx(conn):
                        conn.execute("UPDATE review_questions SET options=? WHERE id=?", (_json.dumps(opts), qid))
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

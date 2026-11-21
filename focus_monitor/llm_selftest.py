"""Checks of the LLM inbox wiring at every call site, with a fake Anthropic client: no network, a
throwaway data dir, inbox and usage log. A note_to_eli call is recorded and answered and the task
carries on; contact_eli and an @ELI: line still stop it. Run: python -m focus_monitor.llm_selftest"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import types


class _Block(types.SimpleNamespace):
    def model_dump(self, **_):
        return dict(vars(self))


class _Resp:
    """Just enough of an SDK message: attributes for the app, model_dump() for llm_contact."""
    def __init__(self, *blocks, parsed=None, stop="end_turn"):
        self.content, self.parsed_output, self.stop_reason = list(blocks), parsed, stop
        self.usage = types.SimpleNamespace(input_tokens=1000, output_tokens=100,
                                           cache_creation_input_tokens=0, cache_read_input_tokens=0)

    def model_dump(self, **_):
        return {"role": "assistant", "content": [dict(vars(b)) for b in self.content], "stop_reason": self.stop_reason}


def _text(s):
    return _Block(type="text", text=s)


def _tool(name, message, i="tu_1"):
    return _Block(type="tool_use", id=i, name=name, input={"message": message})


class _Client:
    """Each request gets the next scripted response; every request's arguments are kept."""
    def __init__(self, *script):
        self.script, self.calls = list(script), []
        door = types.SimpleNamespace(create=self._next, parse=self._next)
        self.messages, self.beta = door, types.SimpleNamespace(messages=door)

    def _next(self, **kw):
        self.calls.append(dict(kw, messages=list(kw["messages"])))
        return self.script.pop(0)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="fm-llm-")
    # Set before the imports below read them: nothing here may reach the real data, inbox or usage log.
    os.environ.update(FOCUS_MONITOR_DATA=tmp + "/data", LLM_INBOX_DIR=tmp + "/inbox", CLAUDE_APP_USAGE_DIR=tmp + "/usage")
    from . import chat, db, evaluator, llm
    lc = llm.llm_contact
    assert str(lc.INBOX_DIR) == tmp + "/inbox", "the throwaway inbox is not in use"
    fails = 0

    def check(name, got, want=True):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f"\n          got  {got!r}\n          want {want!r}"))

    def inbox():
        """(via, halts, message) of what has been recorded since the last look."""
        p = lc.MESSAGES_PATH
        recs = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        p.unlink(missing_ok=True)
        return [(r["via"], r.get("halts", True), r["message"]) for r in recs]

    def answered(call):
        """The note call in the request before was answered with the shared acknowledgement."""
        last = call["messages"][-1]
        return last["role"] == "user" and last["content"] == [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": llm.NOTE_RESULT_TEXT}]

    note = lambda: _tool("note_to_eli", "The rules document names a label that is not in the list.")
    stop = lambda i="tu_1": _tool("contact_eli", "I would rather not continue with this one.", i)
    names = lambda call: [t["name"] for t in call["tools"]]
    cfg = types.SimpleNamespace(llm_backend="api", claude_model="fake-model", claude_effort="high")
    ask = [{"role": "user", "content": "judge this"}]

    # ---- structured calls (llm.parse, API backend): no tool loop, so with_notes answers the note
    c = _Client(_Resp(note(), stop="tool_use"), _Resp(_text('{"ok": true}'), parsed={"ok": True}))
    r = llm.parse(c, site="selftest", cfg=cfg, model="fake-model", system="RULES", messages=ask)
    check("parse: a note leads to a second request and its answer", (r.parsed_output, r.stop_reason, len(c.calls)), ({"ok": True}, "end_turn", 2))
    check("parse: both tools offered, each once", names(c.calls[0]), ["contact_eli", "note_to_eli"])
    check("parse: the note call is answered, then the same request again", answered(c.calls[1]) and len(c.calls[1]["messages"]) == 3
          and names(c.calls[1]) == names(c.calls[0]))
    check("parse: one note recorded, nothing that stops", inbox(), [("note", False, "The rules document names a label that is not in the list.")])
    check("parse: the cost covers both requests", round(r.cost_usd, 9), round(2 * llm.call_cost(_Resp().usage), 9))
    check("parse: the system prompt starts with the shared preamble", c.calls[0]["system"].startswith(lc.preamble()))

    c = _Client(_Resp(stop(), stop="tool_use"))
    r = llm.parse(c, site="selftest", cfg=cfg, model="fake-model", system="RULES", messages=ask, tools=list(llm.CONTACT_TOOLS))
    check("parse: contact_eli stops the task, no second request", (r.parsed_output, r.stop_reason, len(c.calls)), (None, "contact_eli", 1))
    check("parse: tools the caller already passed are not doubled", names(c.calls[0]), ["contact_eli", "note_to_eli"])
    check("parse: the contact is recorded", inbox(), [("tool", True, "I would rather not continue with this one.")])

    c = _Client(_Resp(note(), stop("tu_2"), stop="tool_use"))
    r = llm.parse(c, site="selftest", cfg=cfg, model="fake-model", system="RULES", messages=ask)
    check("parse: a note next to contact_eli still stops", (r.stop_reason, len(c.calls), [v for v, _, _ in inbox()]), ("contact_eli", 1, ["note", "tool"]))

    c = _Client(_Resp(note(), stop="tool_use"), _Resp(_text("@ELI: please look at this before I go on."), stop="end_turn"))
    r = llm.parse(c, site="selftest", cfg=cfg, model="fake-model", system="RULES", messages=ask)
    check("parse: an @ELI: line after a note stops", (r.parsed_output, r.stop_reason, [v for v, _, _ in inbox()]), (None, "contact_eli", ["note", "marker"]))

    # ---- switch and span chats (_run_chat): one request, plain text
    said = [{"role": "user", "text": "why was this a distraction?"}]
    c = _Client(_Resp(_text("It followed a notification."), note(), stop="tool_use"), _Resp(_text("That is all I can see.")))
    reply, cost = chat._run_chat(c, cfg, "RULES", [], said)
    check("chat: text written beside a note stays in the reply", reply, "It followed a notification.\n\nThat is all I can see.")
    check("chat: note answered, second request, one note recorded", (answered(c.calls[1]), len(c.calls), [v for v, _, _ in inbox()]), (True, 2, ["note"]))
    check("chat: both tools offered", names(c.calls[0]), ["contact_eli", "note_to_eli"])
    c = _Client(_Resp(stop(), stop="tool_use"))
    reply, cost = chat._run_chat(c, cfg, "RULES", [], said)
    check("chat: contact_eli shows the notice and stops", (reply.startswith("(Claude asked to talk to Eli"), len(c.calls), [v for v, _, _ in inbox()]), (True, 1, ["tool"]))

    # ---- day chat: its own tool loop, so the dispatcher answers the note like any other tool
    conn = db.connect()
    day = dt.date(2026, 11, 19)
    dc = object.__new__(chat.DayChat)
    dc.cfg, dc.system, dc._context_blocks = cfg, llm.with_preamble("RULES"), lambda conn, day: []
    dc.client = c = _Client(_Resp(_text("Looking at the morning."), note(), stop="tool_use"), _Resp(_text("Nothing needs changing.")))
    out = dc.reply(conn, day, [{"role": "user", "text": "anything wrong this morning?"}])
    check("day chat: the turn carries on after a note", (out["reply"], out["applied"], len(c.calls)), ("Looking at the morning.\n\nNothing needs changing.", [], 2))
    check("day chat: the dispatcher answered the note", answered(c.calls[1]))
    check("day chat: one note recorded, both tools offered", ([v for v, _, _ in inbox()], names(c.calls[0])[-2:]), (["note"], ["contact_eli", "note_to_eli"]))
    dc.client = c = _Client(_Resp(note(), stop("tu_2"), stop="tool_use"))
    out = dc.reply(conn, day, [{"role": "user", "text": "anything wrong this morning?"}])
    check("day chat: contact_eli stops the turn, no tool runs", (out["reply"].startswith("(Claude asked to talk to Eli"), len(c.calls), [v for v, _, _ in inbox()]), (True, 1, ["note", "tool"]))

    # ---- evaluator chat: one request whose proposal tools are read from the answer
    propose = _Block(type="tool_use", id="tu_3", name="propose_rule", input={"rule": "Slack during a call is part of the call."})
    evaluator._ev = types.SimpleNamespace(_system_blocks=lambda conn: [{"type": "text", "text": "RULES"}])
    evaluator._ev.client = c = _Client(_Resp(_text("You are right."), note(), stop="tool_use"), _Resp(_text("Here is the rule."), propose, stop="tool_use"))
    out = evaluator.chat_reply(conn, cfg, 1, [{"role": "user", "text": "this was part of the call"}])
    check("evaluator chat: note answered, the proposal comes from the answer", (out["reply"], out["proposal"], answered(c.calls[1])),
          ("You are right.\n\nHere is the rule.", {"rule": "Slack during a call is part of the call."}, True))
    check("evaluator chat: one note recorded, both tools offered", ([v for v, _, _ in inbox()], names(c.calls[0])[-2:]), (["note"], ["contact_eli", "note_to_eli"]))
    evaluator._ev.client = c = _Client(_Resp(stop(), stop="tool_use"))
    out = evaluator.chat_reply(conn, cfg, 1, [{"role": "user", "text": "this was part of the call"}])
    check("evaluator chat: contact_eli stops", (out["reply"].startswith("(Claude asked to talk to Eli"), out["proposal"], len(c.calls), [v for v, _, _ in inbox()]), (True, None, 1, ["tool"]))

    print(f"llm self-test: {'FAILED' if fails else 'passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

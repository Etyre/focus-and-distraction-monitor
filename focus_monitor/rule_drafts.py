"""Repeated answers become rules. Each review ruling is one precedent; a rules-doc line covers
every later case of its kind. So when several of the person's rulings say the same thing and
the Rules doc does not, the app drafts the line and asks: add it to your rules? (Eli,
2026-11-19: one of the ways the app is meant to converge on his judgments.)

One model call per run, at most every MIN_GAP_S and only once MIN_NEW rulings have come in
since the last run. Drafts wait in `rule_drafts` (status open) and are shown on the Rules
page, each with the rulings it rests on; nothing reaches the Rules doc until he clicks add (web.api_rule_draft_decide), and he
can edit the wording first. Rejected drafts are shown to later runs so they are not re-proposed."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import Config, load_config

log = logging.getLogger("rule_drafts")

MIN_NEW = 8             # new rulings needed before another run
MIN_GAP_S = 6 * 3600.0
MAX_TOKENS = 8000       # three drafts, and thinking counts against it (3,000 before)
MAX_OPEN = 5            # no new drafts while this many wait for an answer

SYSTEM = """You help one person keep the written rules of their focus-tracking app up to date. \
The app judges each stretch of their computer activity (content type, focused or unfocused, and \
whether it continued what came before, shifted to something new, or returned to something \
earlier). The person reviews some of those judgments and rules on them. Their rules document is \
below; every model that judges their activity reads it as authoritative.

Your task: read their recent rulings and find PATTERNS that the rules document does not state. \
A pattern qualifies only when at least three rulings, made on different occasions, say the same \
general thing - most usefully where the app had judged otherwise. For each one, draft a line \
for the document.

- Write the line as the person would: first person ("I", "my"), plain words, one or two \
sentences, specific enough to decide a case (name the kind of app, site or situation).
- Stay inside the evidence. Do not generalise past what the rulings show, and do not restate, \
reword or contradict anything the document already says.
- At most 3 drafts. An empty list is the right answer when nothing qualifies - a weak rule \
costs more than no rule.
- Do not repeat a draft from the "already drafted" list, whatever its status.

EXCEPTIONS. Sometimes the person rules against what the document says, knowing it, and gives a \
reason ("this should be a focus span even though it did not follow the rule, because ..."). Those \
are listed separately, numbered X<id>, each with what they ruled, what the app had under the rules, \
and their reason. One exception changes nothing in the document: it was a one-off. When three or \
more exceptions, made on different occasions, share a reason, the rule is what needs changing: draft \
the line that would have made them unnecessary (a carve-out or an amendment), and say in "why" \
which existing rule it qualifies. Such a draft may qualify what the document says; that is its \
purpose. Give the exceptions it rests on in exception_ids. A draft needs at least three occasions \
in all, counting entries and exceptions together. Exceptions with no reason given, or whose \
reasons do not match, are not a pattern.

How the rulings are listed: one ENTRY per act of ruling. An entry that covers several segments \
(one edit or one chat reply applied to a run of blocks, with one note) is ONE occasion, however \
many segments it lists. Each segment shows what it was, what the person ruled (every field they \
set, and what they say it continues), and what the app had judged before they ruled, with the \
app's reasoning; "differs on" names the fields they changed. Where the ruling was an answer to a \
review question, the question, the context they were shown and the options they did NOT choose \
are given; where it came out of a chat, what they wrote there. Times are in the zone they were \
in that day. In segment_ids give the segment ids of the rulings a draft rests on, from at least \
three different entries.

Their rules document:
{rules}"""


class Draft(BaseModel):
    rule: str = Field(description="The line for the rules document, in the person's first-person voice.")
    why: str = Field(description="One sentence for the person: what their rulings had in common.")
    segment_ids: list[int] = Field(default_factory=list, description="The segment ids of the rulings that support it.")
    exception_ids: list[int] = Field(default_factory=list, description="The exceptions (the number after X) that support it.")


class Drafts(BaseModel):
    drafts: list[Draft] = Field(default_factory=list)


def _new_since_last(conn) -> tuple[int, float]:
    last = conn.execute("SELECT COALESCE(MAX(created), 0) FROM rule_draft_runs").fetchone()[0]
    n = conn.execute("SELECT COUNT(*) FROM seg_reviews WHERE created > ?", (last,)).fetchone()[0]
    try:   # phone rulings count too (2026-11-21: the phone questions exist to surface rules)
        n += conn.execute("SELECT COUNT(*) FROM phone_reviews WHERE created > ?", (last,)).fetchone()[0]
    except Exception:
        pass
    return n, last


def run(conn, cfg: Config | None = None, force: bool = False) -> int:
    """Draft rules if a run is due. Returns how many drafts were stored."""
    from .classifiers.claude_vision import _rules_doc
    from .questions import _client
    from .config import local_dt
    cfg = cfg or load_config()
    new, last = _new_since_last(conn)
    from . import exceptions
    # Exceptions to the rules are rare and each one is deliberate, so three standing ones that no
    # draft has rested on are enough for a run, without waiting for MIN_NEW rulings (2026-11-21).
    if not force and not exceptions.due(conn, last) and (new < MIN_NEW or time.time() - last < MIN_GAP_S):
        return 0
    if conn.execute("SELECT COUNT(*) FROM rule_drafts WHERE status='open'").fetchone()[0] >= MAX_OPEN:
        return 0
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    # Every ruling, not the newest 150 (2026-11-20). The 150 lines held about 64 occasions: 96
    # of them repeated another line's note (one chat reply or bulk edit writes one note on many
    # segments, the largest group 32), while the rule asks for "three rulings made on different
    # occasions" and neither the ruling's date nor its group was printed; 176 of 326 rulings,
    # 105 of them with a note in the person's own words, were never shown. Thread pointers (30
    # of the 150) and the focus exception (82) were not printed; the app's side was missing for
    # 18 because the segment had been judged again since; and 44 notes said only "(answered
    # review question: <label>)" with the question itself not given.
    rows = conn.execute("""SELECT r.*, g.app, g.domain, g.title, g.url, g.start AS g_start, g.end AS g_end
                           FROM seg_reviews r JOIN segments g ON g.id = r.segment_id ORDER BY r.created""").fetchall()
    xlines = exceptions.lines(conn)
    if len(rows) < 3 and conn.execute("SELECT COUNT(*) FROM phone_reviews").fetchone()[0] < 3 and len(xlines) < 3:
        return 0
    lines = _ruling_entries(conn, cfg, rows)
    # Their rulings on phone app use (Eli, 2026-11-21: the phone questions are there to surface the
    # rules about which phone use is passive consumption and which is not).
    plines = []
    try:
        for r in conn.execute("""SELECT r.*, p.app, p.start, p.end, p.content AS pc, p.created AS p_at
                                 FROM phone_reviews r JOIN phone_evals p ON p.seg_key = r.seg_key
                                 ORDER BY r.created""").fetchall():
            from .config import local_dt
            secs = (r["end"] or r["start"]) - r["start"]
            plines.append(f"phone {r['app']} ({local_dt(r['start']).strftime('%b %d %H:%M')}, "
                          f"{secs / 60:.0f} min) -> content={r['content'] or '(unchanged)'}"
                          + (f", focused={'yes' if r['focused'] else 'no'}" if r["focused"] is not None else "")
                          + (f" [the app had said content={r['pc']}]" if r["pc"] and r["pc"] != r["content"] else "")
                          + (f" - their note: {r['note']}" if r["note"] and not r["note"].startswith("(") else ""))
    except Exception:
        pass
    spans = conn.execute("SELECT * FROM span_reviews ORDER BY created").fetchall()
    slines = [f"- {local_dt(r['start']).strftime('%Y-%m-%d %H:%M')}-{local_dt(r['end']).strftime('%H:%M')}, ruled "
              f"{local_dt(r['created']).strftime('%Y-%m-%d')}: "
              + ("not a focus span" if r["verdict"] == "not_focus" else f"a focus span ({r['subtype'] or 'kind as the app had it'})")
              + (f" - their note: “{' '.join(r['note'].split())}”" if r["note"] else "") for r in spans]
    prior = conn.execute("SELECT * FROM rule_drafts ORDER BY created").fetchall()
    text = (f"Their rulings: {len(rows)} segments ruled on, in {len(lines)} entries, oldest first:\n\n" + "\n\n".join(lines)
            + ("\n\nTheir verdicts on whole spans the app had put together:\n" + "\n".join(slines) if slines else "")
            + ("\n\nEXCEPTIONS they made to the written rules:\n" + "\n".join(xlines) if xlines else "")
            + ("\n\nTheir rulings on phone app use (only the app name and timing are known; a rule here "
               "says which use of which app is passive consumption and which is not):\n" + "\n".join(plines) if plines else "")
            + ("\n\nAlready drafted (do not repeat):\n" + "\n".join(
                f"- [{p['status']}] {p['rule']}" + (f"  (why it was drafted: {p['why']}; it rested on segments {p['support']})" if p["why"] else "")
                for p in prior) if prior else ""))
    resp = llm.parse(_client(), site="rule_drafts",
                     model=cfg.claude_model, max_tokens=MAX_TOKENS,
                     system=[{"type": "text", "text": SYSTEM.format(rules=_rules_doc())}],
                     messages=[{"role": "user", "content": text}],
                     output_format=Drafts,
                     output_config={"effort": "medium"},
                     betas=["server-side-fallback-2026-07-01"],
                     fallbacks="default")
    cost = float(getattr(resp, "cost_usd", 0.0) or 0.0) or llm.call_cost(resp.usage)
    now = time.time()
    n = 0
    with db.tx(conn):
        if resp.parsed_output is not None:   # None: the model stopped (contact_eli) or returned nothing usable
            known = {r["segment_id"] for r in rows}
            xknown = {x["id"] for x in exceptions.active(conn)} if xlines else set()
            for d in resp.parsed_output.drafts[:3]:
                support = [i for i in d.segment_ids if i in known]
                exc = sorted({i for i in d.exception_ids if i in xknown})
                if not d.rule.strip() or len(support) + len(exc) < 3:
                    log.warning("dropping a drafted rule with %d valid supporting ruling(s) and %d exception(s): %r",
                                len(support), len(exc), d.rule[:80])
                    continue
                conn.execute("INSERT INTO rule_drafts(rule, why, support, exceptions, created, status) VALUES (?,?,?,?,?,'open')",
                             (d.rule.strip(), d.why.strip(), json.dumps(support), json.dumps(exc) if exc else None, now))
                n += 1
        conn.execute("INSERT INTO rule_draft_runs(created, reviews_seen, n_drafts, cost_usd) VALUES (?,?,?,?)",
                     (now, len(rows), n, cost))
    log.info("rule drafting: %d ruling(s) read, %d draft(s) stored ($%.3f)", len(rows), n, cost)
    return n


def _ruling_entries(conn, cfg, rows) -> list[str]:
    """The person's rulings as the drafting model reads them: one entry per act of ruling
    (rulings sharing a chat reply, or the same second and note: evaluator._group_key), each
    segment with every field ruled and the app's judgment from BEFORE the ruling
    (prompt_parts.judged_before), then the review question answered and what the person wrote
    in the chat it came out of."""
    from . import prompt_parts as pp
    from .evaluator import FOLLOWUP_ASK, ChunkEvaluator, _seg_brief
    from .summaries import verdict_text
    groups: dict = {}
    for r in rows:
        groups.setdefault(ChunkEvaluator._group_key(r), []).append(r)
    out = []
    for members in groups.values():
        members = sorted(members, key=lambda r: r["g_start"])
        first = members[0]
        note = pp.one(first["note"])
        head = (f"ENTRY: ruling on segment id={first['segment_id']}" if len(members) == 1 else
                f"ENTRY: one ruling covering {len(members)} segments (ids {', '.join(str(r['segment_id']) for r in members)})")
        lines = [f"{head}, made {pp.t(first['created'], '%a %Y-%m-%d %H:%M')}"
                 + (" through a chat with the app's model" if first["chat_message_id"] else "") + ":"]
        for r in members:
            end = r["g_end"] if r["g_end"] is not None else r["g_start"]
            lines.append(f"  - segment id={r['segment_id']}: {pp.t(r['g_start'], '%a %Y-%m-%d')} " + _seg_brief(
                {"start": r["g_start"], "end": end, "idle": 0, "app": r["app"], "domain": r["domain"], "title": r["title"], "url": r["url"]}))
            lines.append("      they ruled: " + (verdict_text(r) or "(no field set; a note only)"))
            if r["antecedent_id"]:
                g2 = conn.execute("SELECT * FROM segments WHERE id=?", (r["antecedent_id"],)).fetchone()
                if g2 is not None:
                    lines.append(f"      what they say it continues (id={g2['id']}): " + _seg_brief(g2))
            j = pp.judged_before(conn, r["segment_id"], r["created"])
            cmp_ = pp.compare(conn, r, j)
            lines.append("      the app had judged: " + pp.judged_text(j))
            if j is not None:
                lines.append("      differs on: " + ("; ".join(cmp_["differ"]) or "nothing (they confirmed what the app had)")
                             + (". Also ruled: " + "; ".join(cmp_["extra"]) if cmp_["extra"] else ""))
        lines.append("  their note: " + (f"“{note}”" if note else "(none)"))
        q = pp.ruling_question(conn, first)
        if q is not None:
            lines.append("  it was their answer to this review question:\n" + pp.question_text(*q, indent="    "))
        # what they wrote in the chat the ruling came out of, and to the evaluator about these segments
        said, seen = [], set()
        if first["chat_message_id"]:
            m = conn.execute("SELECT kind, ref, created FROM chat_messages WHERE id=?", (first["chat_message_id"],)).fetchone()
            if m is not None:
                said += conn.execute("SELECT id, text, created FROM chat_messages WHERE kind=? AND ref=? AND role='user' AND created <= ?"
                                     " ORDER BY created", (m["kind"], m["ref"], m["created"])).fetchall()
        for r in members:
            said += conn.execute("SELECT id, text, created FROM chat_messages WHERE kind='evaluator' AND ref=? AND role='user' ORDER BY created",
                                 (str(r["segment_id"]),)).fetchall()
        for m in said:
            if m["id"] in seen or m["text"].strip() == FOLLOWUP_ASK:
                continue
            seen.add(m["id"])
            lines.append(f"  they wrote in the chat ({pp.t(m['created'], '%Y-%m-%d %H:%M')}): “{pp.one(m['text'])}”")
        out.append("\n".join(lines))
    return out


def _verdict(c, sw, f, extra=()):
    from .seg_models import shown_switch
    return " / ".join(v for v in (c, shown_switch(sw), None if f is None else ("focused" if f else "unfocused"),
                                  *extra) if v) or "(nothing)"


def evidence(conn, segment_ids: list[int]) -> list[dict]:
    """The rulings a draft rests on, for the Rules page: what it was, what the person ruled, what
    the app had said, their note, the question they answered (if one), and a capture to look at."""
    out = []
    for sid in segment_ids:
        r = conn.execute("""
            SELECT r.segment_id, r.content, r.switch_label, r.focused, r.self_distraction, r.next_task, r.hybrid,
                   r.note, r.created, g.app, g.domain, g.title, g.start, g.end, g.first_screenshot,
                   e.content AS ec, e.switch_label AS esw, e.focused AS ef, e.created AS e_at, e.rationale
            FROM seg_reviews r JOIN segments g ON g.id = r.segment_id
            LEFT JOIN seg_evals e ON e.segment_id = r.segment_id
            WHERE r.segment_id = ?""", (sid,)).fetchone()
        if r is None:   # the ruling was cleared since the draft was written
            continue
        d = {k: r[k] for k in ("segment_id", "app", "domain", "title", "start", "end", "note")}
        d["ruling"] = _verdict(r["content"], r["switch_label"], r["focused"],
                               ("self-distraction" if r["self_distraction"] else None,
                                "next task in the block" if r["next_task"] else None,
                                "hybrid" if r["hybrid"] else None))
        # only a judgment made before the ruling shows what the app thought on its own
        d["app_said"] = _verdict(r["ec"], r["esw"], r["ef"]) if r["e_at"] is not None and r["e_at"] < r["created"] else None
        d["rationale"] = r["rationale"] if d["app_said"] else None
        # Compared field by field, on the fields the person set (as run() does): a ruling on one
        # axis that agrees with the app is not a correction, though the two strings differ.
        d["corrected"] = bool(d["app_said"]) and bool(
            (r["content"] and r["content"] != r["ec"]) or (r["switch_label"] and r["switch_label"] != r["esw"])
            or (r["focused"] is not None and r["ef"] is not None and bool(r["focused"]) != bool(r["ef"])))
        # The question whose answer made THIS ruling. An option can edit a different segment
        # than the one its question is anchored on, and a spot-check "No" on the kind or the
        # switch makes no ruling at all, so "the latest answered question on this segment" could
        # show one that had nothing to do with it.
        q = None
        for c in conn.execute("""SELECT segment_id, question, answer, note, options, answer_index FROM review_questions
                                 WHERE status='answered' AND (segment_id=? OR options LIKE ?)
                                 ORDER BY answered_at DESC""", (sid, f'%"segment_id": {sid}%')):
            try:
                opts = json.loads(c["options"] or "[]")
            except ValueError:
                opts = []
            if c["answer_index"] is not None and 0 <= c["answer_index"] < len(opts):
                o = opts[c["answer_index"]]
            else:   # answers from before the index was stored: find the option by its label
                o = next((x for x in opts if x.get("label") == c["answer"]), {})
            if (o.get("segment_id") or c["segment_id"]) != sid:
                continue
            if o.get("sample_verdict") == "disagree" and o.get("focused") is None:
                continue
            q = c
            break
        d["question"], d["answer"] = (q["question"], q["answer"]) if q else (None, None)
        if q and q["note"] and not d["note"]:
            d["note"] = q["note"]
        shot = conn.execute("SELECT path FROM segment_shots WHERE segment_id=? AND display=0 ORDER BY ts LIMIT 1", (sid,)).fetchone()
        path = shot["path"] if shot else r["first_screenshot"]
        d["shot"] = path.rsplit("/", 1)[-1] if path else None
        sw = conn.execute("SELECT COALESCE(group_id, id) FROM switches WHERE to_segment=?", (sid,)).fetchone()
        d["switch_id"] = sw[0] if sw else None
        out.append(d)
    out.sort(key=lambda d: d["start"])
    return out


def drafts(conn, decided: int = 20) -> dict:
    """Open drafts with their evidence, and the most recently decided ones (for the record)."""
    def row(r, with_evidence):
        d = dict(r)
        d["support"] = json.loads(d["support"] or "[]")
        d["exceptions"] = json.loads(d.get("exceptions") or "[]")
        if with_evidence:
            d["evidence"] = evidence(conn, d["support"])
            d["exception_rows"] = [dict(x) for i in d["exceptions"]
                                   for x in conn.execute("SELECT * FROM rule_exceptions WHERE id=?", (i,))]
        return d
    return {"open": [row(r, True) for r in conn.execute("SELECT * FROM rule_drafts WHERE status='open' ORDER BY created")],
            "decided": [row(r, False) for r in conn.execute(
                "SELECT * FROM rule_drafts WHERE status!='open' ORDER BY decided_at DESC LIMIT ?", (decided,))]}

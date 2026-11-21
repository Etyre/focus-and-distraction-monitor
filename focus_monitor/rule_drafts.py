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
MAX_REVIEWS = 150       # the newest rulings shown to the model
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

Their rules document:
{rules}"""


class Draft(BaseModel):
    rule: str = Field(description="The line for the rules document, in the person's first-person voice.")
    why: str = Field(description="One sentence for the person: what their rulings had in common.")
    segment_ids: list[int] = Field(description="The ids of the rulings (at least 3) that support it.")


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
    cfg = cfg or load_config()
    new, last = _new_since_last(conn)
    if not force and (new < MIN_NEW or time.time() - last < MIN_GAP_S):
        return 0
    if conn.execute("SELECT COUNT(*) FROM rule_drafts WHERE status='open'").fetchone()[0] >= MAX_OPEN:
        return 0
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    rows = conn.execute("""
        SELECT r.segment_id, r.content, r.switch_label, r.focused, r.self_distraction, r.next_task, r.hybrid,
               r.note, r.created, g.app, g.domain, g.title, g.start,
               e.content AS ec, e.switch_label AS esw, e.focused AS ef, e.created AS e_at
        FROM seg_reviews r JOIN segments g ON g.id = r.segment_id
        LEFT JOIN seg_evals e ON e.segment_id = r.segment_id
        ORDER BY r.created DESC LIMIT ?""", (MAX_REVIEWS,)).fetchall()
    if len(rows) < 3 and conn.execute("SELECT COUNT(*) FROM phone_reviews").fetchone()[0] < 3:
        return 0

    verdict = _verdict
    lines = []
    for r in reversed(rows):
        mine = verdict(r["content"], r["switch_label"], r["focused"],
                       ("self-distraction" if r["self_distraction"] else None,
                        "next task in the block" if r["next_task"] else None,
                        "hybrid (inside a live discussion)" if r["hybrid"] else None))
        # the app's judgment as it stood when they ruled, where that is still on record
        app = ""
        if r["e_at"] is not None and r["e_at"] < r["created"]:
            differs = ((r["content"] and r["content"] != r["ec"]) or (r["switch_label"] and r["switch_label"] != r["esw"])
                       or (r["focused"] is not None and r["ef"] is not None and bool(r["focused"]) != bool(r["ef"])))
            app = f" [the app had said: {verdict(r['ec'], r['esw'], r['ef'])}]" if differs else " [the app had said the same]"
        when = dt.datetime.fromtimestamp(r["start"]).strftime("%b %d %H:%M")
        lines.append(f"id={r['segment_id']} ({when}) {r['app']} {r['domain'] or ''} | {(r['title'] or '')[:70]}"
                     f" -> {mine}{app}" + (f" - their note: {r['note'][:300]}" if r["note"] else ""))
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
    prior = conn.execute("SELECT rule, status FROM rule_drafts ORDER BY created DESC LIMIT 60").fetchall()
    text = ("Their rulings, oldest first:\n" + "\n".join(lines)
            + ("\n\nTheir rulings on phone app use (only the app name and timing are known; a rule here "
               "says which use of which app is passive consumption and which is not):\n" + "\n".join(plines) if plines else "")
            + ("\n\nAlready drafted (do not repeat):\n" + "\n".join(f"- [{p['status']}] {p['rule']}" for p in prior) if prior else ""))
    resp = llm.parse(_client(), site="rule_drafts",
                     model=cfg.claude_model, max_tokens=3000,
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
            for d in resp.parsed_output.drafts[:3]:
                support = [i for i in d.segment_ids if i in known]
                if not d.rule.strip() or len(support) < 3:
                    log.warning("dropping a drafted rule with %d valid supporting ruling(s): %r", len(support), d.rule[:80])
                    continue
                conn.execute("INSERT INTO rule_drafts(rule, why, support, created, status) VALUES (?,?,?,?,'open')",
                             (d.rule.strip(), d.why.strip(), json.dumps(support), now))
                n += 1
        conn.execute("INSERT INTO rule_draft_runs(created, reviews_seen, n_drafts, cost_usd) VALUES (?,?,?,?)",
                     (now, len(rows), n, cost))
    log.info("rule drafting: %d ruling(s) read, %d draft(s) stored ($%.3f)", len(rows), n, cost)
    return n


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
        if with_evidence:
            d["evidence"] = evidence(conn, d["support"])
        return d
    return {"open": [row(r, True) for r in conn.execute("SELECT * FROM rule_drafts WHERE status='open' ORDER BY created")],
            "decided": [row(r, False) for r in conn.execute(
                "SELECT * FROM rule_drafts WHERE status!='open' ORDER BY decided_at DESC LIMIT ?", (decided,))]}

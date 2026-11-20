"""Repeated answers become rules. Each review ruling is one precedent; a rules-doc line covers
every later case of its kind. So when several of the person's rulings say the same thing and
the Rules doc does not, the app drafts the line and asks: add it to your rules? (Eli,
2026-11-19: one of the ways the app is meant to converge on his judgments.)

One model call per run, at most every MIN_GAP_S and only once MIN_NEW rulings have come in
since the last run. Drafts wait in `rule_drafts` (status open) and are shown on the Review
page; nothing reaches the Rules doc until he clicks add (web.api_rule_draft_decide), and he
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
    return n, last


def run(conn, cfg: Config | None = None, force: bool = False) -> int:
    """Draft rules if a run is due. Returns how many drafts were stored."""
    from .classifiers.claude_vision import _rules_doc
    from .questions import _client
    from .seg_models import shown_switch
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
    if len(rows) < 3:
        return 0

    def verdict(c, sw, f, extra=()):
        return " / ".join(v for v in (c, shown_switch(sw), None if f is None else ("focused" if f else "unfocused"),
                                      *extra) if v) or "(nothing)"
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
    prior = conn.execute("SELECT rule, status FROM rule_drafts ORDER BY created DESC LIMIT 60").fetchall()
    text = ("Their rulings, oldest first:\n" + "\n".join(lines)
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


def open_drafts(conn) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM rule_drafts WHERE status='open' ORDER BY created"):
        d = dict(r)
        d["support"] = json.loads(d["support"] or "[]")
        out.append(d)
    return out

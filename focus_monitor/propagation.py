"""Feedback propagation: when the user corrects a mislabeled segment, a judgment role
considers whether that feedback applies to similar segments from the past several days -
re-labeling the ones where it clearly does, flagging for review the ones where it is
uncertain, and leaving the rest. Spans re-derive automatically from the updated judgments.

User rule (2026-09-02): "when I correct something that's mislabeled, that should spin up a
process that considers if that feedback is relevant to other segments from the past several
days, and recomputes them. If the model is uncertain whether the feedback applies to a
particular segment, it should flag that segment for review."
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import Config, load_config
from .seg_models import CONTENTS, KINDS, SWITCHES, shown_switch, stored_switch

log = logging.getLogger("propagation")

LOOKBACK_DAYS = 4
MAX_CANDIDATES = 80


def _foc(row, unset: str = "?") -> str:
    """A row's focused/unfocused judgment in words (rows judged before the redesign have none)."""
    if row is None or "focused" not in row.keys() or row["focused"] is None:
        return unset
    return "focused" if row["focused"] else "unfocused"


class PropVerdict(BaseModel):
    segment_id: int
    verdict: str = Field(description="'applies' (the correction clearly covers this segment too), 'does_not_apply', or 'uncertain' (plausible but the person should decide - it will be flagged for their review).")
    content: str | None = Field(default=None, description="When applies: the corrected content label.")
    switch_label: str | None = Field(default=None, description="When applies and the switch judgment changes: continuation, shift (attention moved to something new), or return.")
    focused: bool | None = Field(default=None, description="When applies and the focus judgment changes: true = attention was focused in the segment, false = unfocused. null leaves it as judged.")
    note: str = Field(description="One short sentence.")


class PropJudgement(BaseModel):
    verdicts: list[PropVerdict]


SYSTEM = """You are the feedback-propagation judge for a personal attention monitor. The person \
corrected one segment's judgment. Decide, for each CANDIDATE segment (similar app/site, past few \
days), whether the same correction logically applies.

Be conservative in both directions: mark 'applies' only when the candidate is clearly the same \
kind of case as the corrected one (same activity, same pattern - not merely the same app); mark \
'uncertain' when it plausibly applies but a human should decide; otherwise 'does_not_apply'. \
The person's note, when present, states the general rule - apply the RULE, not surface similarity.

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        from .config import load_api_key
        load_api_key()
        c = anthropic.Anthropic()
        if not (getattr(c, "api_key", None) or getattr(c, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials")
        _client = c
    return _client


def note_event(conn, segment_id: int) -> None:
    """Queue a user correction for propagation (called by the edit endpoints)."""
    conn.execute("INSERT INTO feedback_events(segment_id, created) VALUES (?,?)",
                 (segment_id, time.time()))



def _retry_later(conn, event_id: int) -> None:
    """Put a feedback event back in the queue after a failed call, at most three times."""
    n = int(db.kv_get(conn, f"prop.fail.{event_id}") or 0) + 1
    db.kv_set(conn, f"prop.fail.{event_id}", str(n))
    if n < 3:
        conn.execute("UPDATE feedback_events SET processed=0 WHERE id=?", (event_id,))


def _apply(conn, rec) -> bool:
    """Write one propagated verdict (a seg_propagated row, or the same fields as a dict) onto the
    segment's stored judgment. False when it changes nothing."""
    sid, src, note = rec["segment_id"], rec["from_segment_id"], rec["note"]
    had = (conn.execute("SELECT rationale FROM seg_evals WHERE segment_id=?", (sid,)).fetchone() or [None])[0] or ""
    if rec["verdict"] == "uncertain":
        mark = f" | does your correction of segment {src} apply here? {note}"
        conn.execute("UPDATE seg_evals SET uncertain=1, rationale=COALESCE(rationale,'') || ? WHERE segment_id=?",
                     ("" if mark in had else mark, sid))
        return True
    sets, args = [], []
    if rec["content"] in CONTENTS:
        sets.append("content=?"); args.append(rec["content"])
    switch_label = stored_switch(rec["switch_label"])  # tolerate the shown name 'shift'
    focused = None if rec["focused"] is None else bool(rec["focused"])
    cur_row = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (sid,)).fetchone()
    foc = focused if focused is not None else (
        None if cur_row is None or cur_row["focused"] is None else bool(cur_row["focused"]))
    if focused is not None:
        sets.append("focused=?"); args.append(int(focused))
    sw_new = switch_label if switch_label in SWITCHES else (cur_row["switch_label"] if cur_row else None)
    if switch_label in SWITCHES:
        sets.append("switch_label=?"); args.append(switch_label)
        if switch_label != "interruption":
            sets.append("next_task=0")
            if switch_label != "return":   # a return keeps its own self-distraction judgment
                sets.append("self_distraction=0")
    if (switch_label in SWITCHES or focused is not None) and foc is not None:
        # the older per-shift kind follows from the fields (evaluator.kind_of)
        from .evaluator import kind_of
        sd = bool(cur_row["self_distraction"]) if cur_row is not None and sw_new in ("interruption", "return") else False
        sets.append("interruption_kind=?"); args.append(kind_of(sw_new, foc, sd))
    elif switch_label in SWITCHES and switch_label != "interruption":
        sets.append("interruption_kind=NULL")
    if not sets:
        return False
    sets.append("uncertain=0")
    sets.append("rationale = COALESCE(rationale,'') || ?")
    mark = f" | corrected via your feedback on segment {src}: {note}"
    args.append("" if mark in had else mark)   # (a re-judgment that kept the old row would stack the mark)
    args.append(sid)
    conn.execute(f"UPDATE seg_evals SET {', '.join(sets)} WHERE segment_id=?", args)
    return True


def reapply(conn, segment_ids) -> int:
    """Put propagated corrections back on segments the evaluator has just judged again. A new
    judgment replaces the whole stored row, and with it went the correction carried over from
    the person's ruling on a similar segment (about 170 of 1,444 by 2026-11-20). The person's
    ruling outranks the evaluator, so it is applied again - unless that ruling has since been
    cleared."""
    n = 0
    for sid in segment_ids:
        for rec in conn.execute("""SELECT p.* FROM seg_propagated p
                                   JOIN seg_reviews r ON r.segment_id = p.from_segment_id
                                   WHERE p.segment_id=? ORDER BY p.created""", (sid,)).fetchall():
            if rec["verdict"] == "uncertain" and conn.execute(
                    "SELECT 1 FROM review_questions WHERE segment_id=? AND status != 'open' AND COALESCE(answered_at, created) >= ?",
                    (sid, rec["created"])).fetchone():
                continue   # the person has dealt with that doubt already: it is not raised again
            n += int(_apply(conn, rec))
    return n


def process_pending(conn, cfg: Config | None = None, max_events: int = 1) -> int:
    from . import efforts, stats, summaries
    from .classifiers.claude_vision import _b64, _rules_doc
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    events = conn.execute("SELECT * FROM feedback_events WHERE processed=0 ORDER BY id LIMIT ?",
                          (max_events,)).fetchall()
    done = 0
    for ev in events:
        # Marked processed here, and un-marked below if the call fails: a failed call is not a
        # propagated correction (event 501 on 2026-11-14 was lost that way).
        conn.execute("UPDATE feedback_events SET processed=1 WHERE id=?", (ev["id"],))
        seg = conn.execute("SELECT * FROM segments WHERE id=?", (ev["segment_id"],)).fetchone()
        rv = conn.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (ev["segment_id"],)).fetchone()
        if not seg or not rv:
            continue
        old = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (ev["segment_id"],)).fetchone()
        # A ruling that sets nothing different from the model's judgment is a confirmation.
        # Calling it a correction told the judge that something had been wrong.
        same = old is not None and all(
            rv[k] is None or rv[k] == old[k] for k in ("content", "switch_label", "focused"))
        correction = (f"The person {'confirmed the judgment of' if same else 'corrected'} segment id={seg['id']} "
                      f"({seg['app']} {seg['domain'] or ''} | {(seg['title'] or '')[:70]}).\n"
                      f"Model had said: content={old['content'] if old else '?'}, "
                      f"switch={shown_switch(old['switch_label']) if old else '?'}, focus={_foc(old)}.\n"
                      f"The person set: content={rv['content'] or '(unchanged)'}, "
                      f"switch={shown_switch(rv['switch_label']) or '(unchanged)'}, focus={_foc(rv, '(unchanged)')}.\n"
                      f"Their note: {rv['note'] or '(none)'}")
        cands = conn.execute("""
            SELECT g.*, e.content AS e_content, e.switch_label AS e_switch,
                   e.interruption_kind AS e_kind, e.rationale AS e_rationale
            FROM segments g JOIN seg_evals e ON e.segment_id = g.id
            LEFT JOIN seg_reviews r ON r.segment_id = g.id
            WHERE r.segment_id IS NULL AND g.id != ? AND g.start >= ?
              AND (g.app = ? OR (g.domain != '' AND g.domain = ?))
            ORDER BY g.start DESC LIMIT ?""",
            (seg["id"], time.time() - LOOKBACK_DAYS * 86400, seg["app"], seg["domain"],
             MAX_CANDIDATES)).fetchall()
        if not cands:
            continue
        lines = []
        for c2 in cands:
            d = dict(c2)
            d["clip_start"], d["duration"] = d["start"], (d["end"] or d["start"]) - d["start"]
            lines.append(f"id={d['id']} " + summaries.seg_line(d)
                         + f"  [current: {c2['e_content']}/{c2['e_switch']}/{c2['e_kind']}]"
                         + (f"  // {c2['e_rationale'][:70]}" if c2["e_rationale"] else ""))
        content_blocks: list = []
        from . import shots as _shots
        sh = next(iter(_shots.pick(conn, [seg["id"]], 1)), None)
        if sh:
            b = _b64(sh["path"])
            if b:
                content_blocks.append({"type": "text", "text": "Screenshot of the CORRECTED segment:"})
                content_blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        prompt = (correction + "\n\nCANDIDATE segments (same app/site, past few days) with their "
                  "current judgments:\n" + "\n".join(lines)
                  + ("\n\nThe person agreed with the model here, so nothing was wrong in this segment. A candidate "
                     "that is clearly the same kind of case but is judged differently should get the confirmed "
                     "judgment ('applies'); one already judged the same way is 'does_not_apply'."
                     if same else "")
                  + "\n\nFor every candidate: does the correction apply, not apply, or uncertain?")
        content_blocks.append({"type": "text", "text": prompt})
        system = SYSTEM.format(rules=_rules_doc(), about_me=cfg.about_me.strip() or "(none)")
        client = _get_client()

        def _call(eff):
            return llm.parse(client, site="propagation", 
                model=cfg.claude_model, max_tokens=6000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content_blocks}],
                output_format=PropJudgement,
                output_config={"effort": eff},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")

        try:
            resp = _call(cfg.claude_effort)
        except Exception:
            log.exception("propagation call failed for event %d", ev["id"])
            _retry_later(conn, ev["id"])
            continue
        if resp.parsed_output is None:
            if resp.stop_reason != "contact_eli":   # (a contact stops the task for good: no retry)
                _retry_later(conn, ev["id"])
            continue
        cost = efforts.call_cost(resp.usage)
        cand_ids = {c2["id"] for c2 in cands}
        n_applied = n_flagged = 0
        now = time.time()
        with db.tx(conn):
            # this ruling's earlier verdicts are replaced by the ones just given: a target it
            # no longer applies to must not get the old correction back at its next re-judgment
            conn.execute("DELETE FROM seg_propagated WHERE from_segment_id=?", (seg["id"],))
            for v in resp.parsed_output.verdicts:
                if v.segment_id not in cand_ids:
                    continue
                verdict = v.verdict.strip().lower()
                if verdict not in ("applies", "uncertain"):
                    continue
                rec = {"segment_id": v.segment_id, "from_segment_id": seg["id"], "verdict": verdict,
                       "content": v.content, "switch_label": v.switch_label, "focused": v.focused, "note": v.note}
                if not _apply(conn, rec):
                    continue
                # Kept in a table of its own: the evaluator replaces the whole seg_evals row
                # when it judges a segment again, and reapply() puts the correction back.
                conn.execute("INSERT OR REPLACE INTO seg_propagated(segment_id, from_segment_id, verdict, content,"
                             " switch_label, focused, note, created) VALUES (?,?,?,?,?,?,?,?)",
                             (v.segment_id, seg["id"], verdict, v.content, v.switch_label,
                              None if v.focused is None else int(v.focused), v.note, now))
                if verdict == "applies":
                    n_applied += 1
                else:
                    n_flagged += 1
            conn.execute("UPDATE feedback_events SET n_applied=?, n_flagged=?, cost_usd=? WHERE id=?",
                         (n_applied, n_flagged, cost, ev["id"]))
        log.info("propagated feedback on segment %d: %d applied, %d flagged (of %d candidates, $%.3f)",
                 seg["id"], n_applied, n_flagged, len(cands), cost)
        # effort-ladder shadows for this judgment role too
        def _cmp(a, b):
            pa = {x.segment_id: x.verdict for x in a.verdicts}
            pb = {x.segment_id: x.verdict for x in b.verdicts}
            keys = set(pa) & set(pb)
            if not keys:
                return 0.0, "no overlap"
            same = sum(1 for k in keys if pa[k] == pb[k])
            return same / len(keys), f"{same}/{len(keys)} verdicts match"
        efforts.run_shadows(conn, cfg, client, "feedback_propagation", f"edit:{seg['id']}",
                            cfg.claude_effort, resp, _call, _cmp)
        done += 1
    return done

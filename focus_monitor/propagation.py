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

from . import db
from .config import Config, load_config
from .seg_models import CONTENTS, KINDS, SWITCHES

log = logging.getLogger("propagation")

LOOKBACK_DAYS = 4
MAX_CANDIDATES = 80


class PropVerdict(BaseModel):
    segment_id: int
    verdict: str = Field(description="'applies' (the correction clearly covers this segment too), 'does_not_apply', or 'uncertain' (plausible but the person should decide - it will be flagged for their review).")
    content: str | None = Field(default=None, description="When applies: the corrected content label.")
    switch_label: str | None = Field(default=None, description="When applies and the switch judgment changes.")
    interruption_kind: str | None = Field(default=None, description="When applies and switch_label is 'interruption'.")
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
        conn.execute("UPDATE feedback_events SET processed=1 WHERE id=?", (ev["id"],))
        seg = conn.execute("SELECT * FROM segments WHERE id=?", (ev["segment_id"],)).fetchone()
        rv = conn.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (ev["segment_id"],)).fetchone()
        if not seg or not rv:
            continue
        old = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (ev["segment_id"],)).fetchone()
        correction = (f"The person corrected segment id={seg['id']} "
                      f"({seg['app']} {seg['domain'] or ''} | {(seg['title'] or '')[:70]}).\n"
                      f"Model had said: content={old['content'] if old else '?'}, "
                      f"switch={old['switch_label'] if old else '?'}, kind={old['interruption_kind'] if old else '?'}.\n"
                      f"The person set: content={rv['content'] or '(unchanged)'}, "
                      f"switch={rv['switch_label'] or '(unchanged)'}, kind={rv['interruption_kind'] or '(unchanged)'}.\n"
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
        sh = conn.execute("SELECT path FROM segment_shots WHERE segment_id=? ORDER BY ts LIMIT 1",
                          (seg["id"],)).fetchone()
        if sh:
            b = _b64(sh["path"])
            if b:
                content_blocks.append({"type": "text", "text": "Screenshot of the CORRECTED segment:"})
                content_blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        prompt = (correction + "\n\nCANDIDATE segments (same app/site, past few days) with their "
                  "current judgments:\n" + "\n".join(lines)
                  + "\n\nFor every candidate: does the correction apply, not apply, or uncertain?")
        content_blocks.append({"type": "text", "text": prompt})
        system = SYSTEM.format(rules=_rules_doc(), about_me=cfg.about_me.strip() or "(none)")
        client = _get_client()

        def _call(eff):
            return client.beta.messages.parse(
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
            continue
        if resp.parsed_output is None:
            continue
        cost = efforts.call_cost(resp.usage)
        cand_ids = {c2["id"] for c2 in cands}
        n_applied = n_flagged = 0
        now = time.time()
        with db.tx(conn):
            for v in resp.parsed_output.verdicts:
                if v.segment_id not in cand_ids:
                    continue
                verdict = v.verdict.strip().lower()
                if verdict == "applies":
                    sets, args = [], []
                    if v.content in CONTENTS:
                        sets.append("content=?"); args.append(v.content)
                    if v.switch_label in SWITCHES:
                        sets.append("switch_label=?"); args.append(v.switch_label)
                        sets.append("interruption_kind=?")
                        args.append(v.interruption_kind if (v.switch_label == "interruption"
                                    and v.interruption_kind in KINDS) else None)
                    if not sets:
                        continue
                    sets.append("uncertain=0")
                    sets.append("rationale = COALESCE(rationale,'') || ?")
                    args.append(f" | corrected via your feedback on segment {seg['id']}: {v.note}")
                    args.append(v.segment_id)
                    conn.execute(f"UPDATE seg_evals SET {', '.join(sets)} WHERE segment_id=?", args)
                    n_applied += 1
                elif verdict == "uncertain":
                    conn.execute("UPDATE seg_evals SET uncertain=1, rationale=COALESCE(rationale,'') || ? WHERE segment_id=?",
                                 (f" | does your correction of segment {seg['id']} apply here? {v.note}", v.segment_id))
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

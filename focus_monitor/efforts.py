"""Effort-ladder experimentation: every judgment-role LLM call is repeated at the lower effort
levels with the SAME prompt and context; all outputs are stored, mechanically compared, and
then judged by a dedicated effort-judge role - so the system learns, per role, where low and
medium effort get the same result as high, and cost can later be scaled down with evidence.

The user's framing: "I bet we don't actually need high effort very often and I want the whole
system to be learning where low and medium effort get the same result."
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from pydantic import BaseModel, Field

from . import db
from .config import (PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, Config)

log = logging.getLogger("efforts")

LADDER = ["low", "medium", "high"]


def lower_efforts(primary: str) -> list[str]:
    """Up to two effort levels below the primary, nearest first."""
    try:
        i = LADDER.index(primary)
    except ValueError:
        return []
    return list(reversed(LADDER[max(0, i - 2):i]))


def call_cost(usage) -> float:
    return ((usage.input_tokens or 0) * PRICE_INPUT + (usage.output_tokens or 0) * PRICE_OUTPUT
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)


def record_trial(conn, role: str, ref: str, model: str, effort: str, is_primary: bool,
                 response_json: str, cost: float, prompt: str | None = None,
                 agree: float | None = None, agree_detail: str | None = None) -> int:
    return conn.execute(
        "INSERT INTO effort_trials(role, ref, model, effort, is_primary, prompt, response, agree, agree_detail, cost_usd, created)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (role, ref, model, effort, int(is_primary), prompt, response_json, agree, agree_detail,
         cost, time.time())).lastrowid


class EffortVerdict(BaseModel):
    effort: str
    verdict: str = Field(description="'equivalent' (same conclusions), 'minor' (differences that would not change any downstream decision), or 'material' (a decision would differ).")
    note: str = Field(description="One sentence: what differs and whether it matters.")


class EffortJudgement(BaseModel):
    verdicts: list[EffortVerdict]


JUDGE_SYSTEM = """You are the effort-comparison judge for a personal attention monitor. The same \
prompt was answered by the same model at multiple effort levels. Compare each LOWER-effort output \
against the PRIMARY output and say whether the differences would change any downstream decision \
(labels, boundaries, flags - not phrasing). Judge substance, not style."""


def judge(conn, cfg: Config, client, role: str, primary_effort: str, primary_json: str,
          shadows: list[tuple[int, str, str]]) -> None:
    """shadows: [(trial_id, effort, response_json)]. Writes judge verdicts onto the trial rows."""
    if not shadows:
        return
    parts = [f"ROLE: {role}", f"PRIMARY OUTPUT (effort={primary_effort}):\n{primary_json[:6000]}"]
    for _tid, eff, rj in shadows:
        parts.append(f"OUTPUT at effort={eff}:\n{rj[:6000]}")
    try:
        resp = client.beta.messages.parse(
            model=cfg.claude_model, max_tokens=1000,
            system=[{"type": "text", "text": JUDGE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
            output_format=EffortJudgement,
            output_config={"effort": "low"})
    except Exception:
        log.exception("effort judge failed")
        return
    if resp.parsed_output is None:
        return
    cost = call_cost(resp.usage)
    per = cost / max(len(shadows), 1)
    by_eff = {v.effort.strip().lower(): v for v in resp.parsed_output.verdicts}
    for tid, eff, _rj in shadows:
        v = by_eff.get(eff)
        if v is None:
            continue
        verdict = v.verdict.strip().lower()
        if verdict not in ("equivalent", "minor", "material"):
            verdict = "minor"
        conn.execute("UPDATE effort_trials SET judge_verdict=?, "
                     "agree_detail=COALESCE(agree_detail,'') || ?, cost_usd=cost_usd+? WHERE id=?",
                     (verdict, f" | judge: {v.note}", per, tid))


def run_shadows(conn, cfg: Config, client, role: str, ref: str, primary_effort: str,
                primary_resp, call_fn, compare_fn, prompt_text: str | None = None) -> None:
    """The generic ladder experiment around one already-made primary call.
    call_fn(effort) -> parse response (same prompt/context); compare_fn(primary_parsed,
    shadow_parsed) -> (agreement_fraction, detail). Never raises."""
    try:
        primary_json = primary_resp.parsed_output.model_dump_json()
        record_trial(conn, role, ref, cfg.claude_model, primary_effort, True,
                     primary_json, call_cost(primary_resp.usage), prompt=prompt_text)
        shadows = []
        for eff in lower_efforts(primary_effort):
            try:
                r = call_fn(eff)
            except Exception:
                log.exception("shadow call failed (%s @ %s)", role, eff)
                continue
            if r.parsed_output is None:
                continue
            rj = r.parsed_output.model_dump_json()
            agree, detail = compare_fn(primary_resp.parsed_output, r.parsed_output)
            tid = record_trial(conn, role, ref, cfg.claude_model, eff, False, rj,
                               call_cost(r.usage), agree=agree, agree_detail=detail)
            shadows.append((tid, eff, rj))
        judge(conn, cfg, client, role, primary_effort, primary_json, shadows)
    except Exception:
        log.exception("effort shadows failed for %s", role)


# ---- reporting --------------------------------------------------------------------------------

def summary(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT role, model, effort, is_primary, COUNT(*) n, AVG(agree) avg_agree,
               AVG(cost_usd) avg_cost,
               SUM(CASE WHEN judge_verdict='equivalent' THEN 1 ELSE 0 END) * 1.0 /
                   MAX(SUM(CASE WHEN judge_verdict IS NOT NULL THEN 1 ELSE 0 END), 1) equiv_rate,
               SUM(CASE WHEN judge_verdict='material' THEN 1 ELSE 0 END) n_material
        FROM effort_trials GROUP BY role, model, effort, is_primary
        ORDER BY role, is_primary DESC, effort DESC""").fetchall()
    return [dict(r) for r in rows]


def trials(conn, role: str | None = None, effort: str | None = None, limit: int = 50) -> list[dict]:
    q = "SELECT id, role, ref, model, effort, is_primary, agree, judge_verdict, cost_usd, created FROM effort_trials WHERE 1=1"
    args: list = []
    if role:
        q += " AND role=?"; args.append(role)
    if effort:
        q += " AND effort=?"; args.append(effort)
    q += " ORDER BY created DESC LIMIT ?"; args.append(limit)
    return [dict(r) for r in conn.execute(q, args)]


def trial_detail(conn, trial_id: int) -> dict | None:
    t = conn.execute("SELECT * FROM effort_trials WHERE id=?", (trial_id,)).fetchone()
    if not t:
        return None
    siblings = [dict(r) for r in conn.execute(
        "SELECT * FROM effort_trials WHERE role=? AND ref=? ORDER BY is_primary DESC, effort DESC",
        (t["role"], t["ref"]))]
    prompt = t["prompt"]
    if prompt is None and t["role"] == "segment_evaluator" and t["ref"]:
        run = conn.execute("SELECT prompt FROM eval_runs WHERE id=?", (t["ref"],)).fetchone()
        prompt = run["prompt"] if run else None
    return {"trial": dict(t), "siblings": siblings, "prompt": prompt}

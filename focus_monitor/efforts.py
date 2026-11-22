"""Effort comparison: is "high" effort buying accuracy?

Since 2026-11-21 the judgment calls run at "medium" (config.toml: claude_effort) and a random
share of them (effort_check_fraction) is repeated at "high" with the SAME prompt and context
(Eli: "switch to medium, with 'high' spot-checks to see if they're more accurate"). Both
outputs are stored, compared mechanically, and read by an effort-judge call. That says how
often they DIFFER; which one is RIGHT comes from the person's own rulings (accuracy() below).

Before that the calls ran at "high" and every one was repeated at "medium" and "low". That
still happens if claude_effort is set back to "high". (The evaluator's rows from before
2026-11-20 03:34 are labelled high but ran low: evaluator.LOW_EFFORT_FIXED_AT.)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import (PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, Config)

log = logging.getLogger("efforts")

LADDER = ["low", "medium", "high"]
CHECK_EFFORT = "high"


def comparison_efforts(primary: str, cfg, always: bool = False) -> list[str]:
    """The other effort levels to run beside this primary call. At "high": the two below it,
    every time (the old ladder). Below "high": one check at "high", on a random share of calls
    (`always` for a role that runs once a day)."""
    if primary == CHECK_EFFORT:
        return lower_efforts(primary)
    if primary not in LADDER:
        return []
    return [CHECK_EFFORT] if always or random.random() < getattr(cfg, "effort_check_fraction", 0.1) else []


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
prompt was answered by the same model at different effort levels. Compare each OTHER output \
against the PRIMARY output and say whether the differences would change any downstream decision \
(labels, boundaries, flags - not phrasing). Judge substance, not style."""


def judge(conn, cfg: Config, client, role: str, primary_effort: str, primary_json: str,
          shadows: list[tuple[int, str, str]]) -> None:
    """shadows: [(trial_id, effort, response_json)]. Writes judge verdicts onto the trial rows."""
    if not shadows:
        return
    # The outputs whole. Each was cut at 6,000 characters until 2026-11-20: 996 of 1,043 stored
    # feedback-propagation outputs are longer (up to 14,881), so for that role the judge
    # compared the first part of two verdict lists and called the rest equivalent unseen.
    parts = [f"ROLE: {role}", f"PRIMARY OUTPUT (effort={primary_effort}):\n{primary_json}"]
    for _tid, eff, rj in shadows:
        parts.append(f"OUTPUT at effort={eff}:\n{rj}")
    try:
        resp = llm.parse(client, site="effort_judge", 
            model=cfg.claude_model, max_tokens=4000,   # thinking counts against it (1,000 before)
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
                primary_resp, call_fn, compare_fn, prompt_text: str | None = None, always: bool = False) -> None:
    """The comparison around one already-made primary call (comparison_efforts says at which
    other levels; most medium calls get none, and then nothing is recorded).
    call_fn(effort) -> parse response (same prompt/context); compare_fn(primary_parsed,
    shadow_parsed) -> (agreement_fraction, detail). Never raises."""
    try:
        others = comparison_efforts(primary_effort, cfg, always)
        if not others:
            return
        primary_json = primary_resp.parsed_output.model_dump_json()
        record_trial(conn, role, ref, cfg.claude_model, primary_effort, True,
                     primary_json, call_cost(primary_resp.usage), prompt=prompt_text)
        shadows = []
        for eff in others:
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


def accuracy(conn) -> dict:
    """High against medium, scored on the person's rulings: over the evaluator chunks judged at
    both levels (whichever was the primary), each block the person ruled on AFTERWARDS, one count
    per axis they ruled. `differ` is the part that answers the question: blocks where the two
    levels disagreed and the person said which was right. Counted from LOW_EFFORT_FIXED_AT (older
    "high" rows ran at low). The rulings lean toward the primary's mistakes (a question is asked
    where the primary was unsure, and the person saw its answer), so read `differ` by era."""
    from .evaluator import LOW_EFFORT_FIXED_AT
    from .seg_models import stored_switch
    rev = {r["segment_id"]: r for r in conn.execute("SELECT segment_id, content, switch_label, focused, created FROM seg_reviews")}
    runs: dict = {}
    for t in conn.execute("SELECT ref, effort, is_primary, response, created FROM effort_trials WHERE role='segment_evaluator'"
                          " AND effort IN ('high','medium') AND created > ? ORDER BY id", (LOW_EFFORT_FIXED_AT,)):
        runs.setdefault(t["ref"], {})[t["effort"]] = t
    out = {}
    for primary in ("high", "medium"):
        ax = {a: {"ruled": 0, "high_right": 0, "medium_right": 0, "differ": 0, "differ_high_right": 0, "differ_medium_right": 0}
              for a in ("content", "switch", "focused")}
        n_runs = n_blocks = 0
        for d in runs.values():
            if len(d) < 2 or not d[primary]["is_primary"]:
                continue
            n_runs += 1
            try:
                js = {e: {j["segment_id"]: j for j in json.loads(d[e]["response"])["segments"]} for e in d}
            except (ValueError, KeyError, TypeError):
                continue
            for sid in set(js["high"]) & set(js["medium"]):
                r = rev.get(sid)
                if r is None or r["created"] < d[primary]["created"]:
                    continue
                n_blocks += 1
                truth = {"content": r["content"], "switch": r["switch_label"],
                         "focused": None if r["focused"] is None else bool(r["focused"])}
                for a, c in ax.items():
                    if truth[a] is None:
                        continue
                    got = {e: (stored_switch(js[e][sid].get("switch")) if a == "switch" else bool(js[e][sid].get("focused"))
                               if a == "focused" else js[e][sid].get("content")) for e in js}
                    c["ruled"] += 1
                    c["high_right"] += got["high"] == truth[a]
                    c["medium_right"] += got["medium"] == truth[a]
                    if got["high"] != got["medium"]:
                        c["differ"] += 1
                        c["differ_high_right"] += got["high"] == truth[a]
                        c["differ_medium_right"] += got["medium"] == truth[a]
        out[primary] = {"runs": n_runs, "blocks": n_blocks, "axes": ax}
    return out


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

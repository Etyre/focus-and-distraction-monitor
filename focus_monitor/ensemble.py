"""Combine model predictions, weighted by each model's log-loss on recent human reviews."""
from __future__ import annotations

import json
import math

from . import LABELS
from .classifiers.base import Probs, normalize

DEFAULT_WEIGHTS = {"claude": 1.0, "heuristic": 0.35, "learned": 0.35, "trivial": 3.0}
MIN_SCORED = 8          # reviewed predictions needed before a model's weight is data-driven
WINDOW = 200            # most recent reviewed switches used for scoring


def model_scores(conn) -> dict[str, dict]:
    """Per model: n, accuracy, mean log-loss over the most recent reviewed switches."""
    rows = conn.execute(f"""
        SELECT p.model, p.p_continuation, p.p_interruption, p.p_distraction, p.p_task_change, r.label,
               COALESCE(e.importance, 1.0) AS importance
        FROM predictions p JOIN reviews r ON r.switch_id = p.switch_id
        LEFT JOIN ensemble e ON e.switch_id = p.switch_id
        WHERE p.created < r.created AND p.model != 'trivial'
        ORDER BY r.created DESC LIMIT {WINDOW * 4}""").fetchall()
    acc: dict[str, dict] = {}
    for r in rows:
        s = acc.setdefault(r["model"], {"n": 0, "w": 0.0, "correct": 0.0, "logloss": 0.0, "n_focus": 0, "correct_focus": 0})
        if s["n"] >= WINDOW:
            continue
        probs = {"continuation": r["p_continuation"], "interruption": r["p_interruption"],
                 "distraction": r["p_distraction"], "task_change": r["p_task_change"]}
        pred = max(probs, key=probs.get)
        w = float(r["importance"])  # switches leaving a focus span count for more
        hit = int(pred == r["label"])
        s["n"] += 1
        s["w"] += w
        s["correct"] += w * hit
        s["logloss"] += w * -math.log(max(probs.get(r["label"], 1e-6), 1e-6))
        if w >= 1.0:
            s["n_focus"] += 1
            s["correct_focus"] += hit
    for s in acc.values():
        s["accuracy"] = s["correct"] / s["w"] if s["w"] else None          # importance-weighted
        s["logloss"] = s["logloss"] / s["w"] if s["w"] else None
        s["accuracy_focus"] = s["correct_focus"] / s["n_focus"] if s["n_focus"] else None
    return acc


def weights_from_scores(scores: dict[str, dict], models: list[str]) -> dict[str, float]:
    w = {}
    for m in models:
        s = scores.get(m)
        if s and s["n"] >= MIN_SCORED:
            w[m] = math.exp(-2.0 * s["logloss"])  # perfect => 1, uniform-ish (ll≈1.39) => 0.06
        else:
            w[m] = DEFAULT_WEIGHTS.get(m, 0.3)
    return w


def combine(preds: dict[str, Probs], weights: dict[str, float]) -> tuple[Probs, dict[str, float]]:
    used = {m: weights.get(m, 0.3) for m in preds}
    total = sum(used.values()) or 1.0
    out = {k: sum(used[m] * preds[m][k] for m in preds) / total for k in LABELS}
    return normalize(out), used


def is_uncertain(probs: Probs, preds: dict[str, Probs], threshold: float) -> bool:
    top = max(probs.values())
    if top < threshold:
        return True
    label = max(probs, key=probs.get)
    # Disagreement between Claude and the ensemble label is worth a human look.
    if "claude" in preds and max(preds["claude"], key=preds["claude"].get) != label:
        return True
    return False


def dumps_weights(w: dict[str, float]) -> str:
    return json.dumps({k: round(v, 4) for k, v in w.items()})

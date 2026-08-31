"""Segment-evaluation ensemble: several models output probabilities over the three judgment
heads (content / switch / interruption kind), are trained and weighted by the user's edits
(seg_reviews), and are combined into the resolved judgment. When the cheap local models earn a
good enough track record on a head, the LLM call is skipped for confident segments (deferral);
a once-a-day LLM audit sweep re-checks the whole day and flags suspected errors for review.

Models:
  heuristic - probability priors from app/domain/Toggl categories (no LLM, no training).
  learned   - logistic regression per head, trained on seg_reviews (no LLM).
  claude    - the chunk evaluator's judgment, converted to (pseudo-)probabilities.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time

from . import db

log = logging.getLogger("seg_models")

CONTENTS = ("creative", "task", "reading", "planning", "other", "passive", "meeting", "idle")
SWITCHES = ("continuation", "interruption", "return")
KINDS = ("self_distraction", "focus_start", "detour")
HEADS = {"content": CONTENTS, "switch": SWITCHES, "kind": KINDS}

DEFAULT_WEIGHTS = {"claude": 1.0, "heuristic": 0.35, "learned": 0.35}
MIN_SCORED = 8
WINDOW = 300

MEETING_APPS = ("zoom.us", "zoom", "Google Meet", "FaceTime", "Microsoft Teams", "Around", "Otter")
MEETING_DOMAINS = ("meet.google.com", "zoom.us", "whereby.com", "otter.ai")


def _norm(p: dict) -> dict:
    t = sum(p.values()) or 1.0
    return {k: v / t for k, v in p.items()}


def _uniformish(labels, top: str | None = None, top_p: float = 0.4) -> dict:
    if top is None:
        return {l: 1.0 / len(labels) for l in labels}
    rest = (1.0 - top_p) / (len(labels) - 1)
    return {l: (top_p if l == top else rest) for l in labels}


def _dom_in(dom: str, domains) -> bool:
    return any(dom == d or dom.endswith("." + d) for d in domains)


# ---- feature extraction (shared by heuristic + learned) ---------------------------------------

def features(cfg, seg: dict, prev: dict | None) -> dict:
    dom = (seg.get("domain") or "").lower()
    app = seg.get("app") or ""
    dur = seg.get("duration") or ((seg.get("end") or seg["start"]) - seg["start"])
    f = {
        "app=" + app: 1.0,
        "dom=" + dom: 1.0,
        "dur=" + ("<30s" if dur < 30 else "<2m" if dur < 120 else "<10m" if dur < 600 else "10m+"): 1.0,
        "hour=" + str(dt.datetime.fromtimestamp(seg["start"]).hour // 4): 1.0,
        "toggl": 1.0 if seg.get("toggl_id") else 0.0,
        "same_ctx_prev": 1.0 if prev is not None and (prev.get("app"), prev.get("domain")) == (app, seg.get("domain")) else 0.0,
        "authoring": 1.0 if app in cfg.authoring_apps or _dom_in(dom, cfg.authoring_domains) else 0.0,
        "notes": 1.0 if app in cfg.notes_apps or _dom_in(dom, cfg.focus_domains) else 0.0,
        "passive_dom": 1.0 if _dom_in(dom, cfg.passive_domains) else 0.0,
        "distraction_dom": 1.0 if _dom_in(dom, cfg.distraction_domains) else 0.0,
        "meeting": 1.0 if any(m.lower() in app.lower() for m in MEETING_APPS) or _dom_in(dom, MEETING_DOMAINS) else 0.0,
        "mail": 1.0 if "mail" in dom or app == "Mail" else 0.0,
    }
    if prev is not None:
        f["prev_dom=" + (prev.get("domain") or prev.get("app") or "")] = 1.0
    return f


# ---- heuristic model --------------------------------------------------------------------------

def heuristic_predict(cfg, seg: dict, prev: dict | None) -> dict:
    """{head: {label: prob}} from category priors. Weak by design - it earns weight only if
    its track record justifies it."""
    f = features(cfg, seg, prev)
    if f["meeting"]:
        content = _uniformish(CONTENTS, "meeting", 0.75)
    elif f["passive_dom"]:
        content = _norm({"passive": 0.6, "reading": 0.2, "other": 0.1, "task": 0.04,
                         "creative": 0.02, "planning": 0.02, "meeting": 0.02})
    elif f["mail"]:
        content = _uniformish(CONTENTS, "task", 0.6)
    elif f["notes"]:
        content = _norm({"planning": 0.3, "creative": 0.25, "reading": 0.2, "task": 0.1,
                         "other": 0.1, "passive": 0.03, "meeting": 0.02})
    elif f["authoring"]:
        content = _uniformish(CONTENTS, "creative", 0.55)
    elif f["distraction_dom"]:
        content = _norm({"passive": 0.4, "reading": 0.3, "other": 0.15, "task": 0.05,
                         "creative": 0.04, "planning": 0.03, "meeting": 0.03})
    else:
        content = _norm({"reading": 0.25, "other": 0.2, "task": 0.2, "creative": 0.15,
                         "planning": 0.1, "passive": 0.07, "meeting": 0.03})
    if f["same_ctx_prev"]:
        switch = _uniformish(SWITCHES, "continuation", 0.85)
    elif f["passive_dom"] or f["distraction_dom"]:
        switch = _norm({"interruption": 0.55, "continuation": 0.35, "return": 0.10})
    else:
        switch = _norm({"continuation": 0.55, "interruption": 0.3, "return": 0.15})
    if f["passive_dom"] or f["distraction_dom"]:
        kind = _norm({"self_distraction": 0.65, "detour": 0.25, "focus_start": 0.10})
    elif f["toggl"]:
        kind = _norm({"focus_start": 0.5, "detour": 0.4, "self_distraction": 0.1})
    else:
        kind = _norm({"detour": 0.55, "focus_start": 0.25, "self_distraction": 0.2})
    return {"content": content, "switch": switch, "kind": kind}


# ---- learned model ----------------------------------------------------------------------------

class LearnedSegModel:
    """One logistic regression per head, trained on the user's seg_reviews."""

    HEAD_COL = {"content": "content", "switch": "switch_label", "kind": "interruption_kind"}

    def __init__(self):
        self.models: dict = {}
        self._trained_on = -1

    def learn(self, conn, cfg) -> None:
        n_reviews = conn.execute("SELECT COUNT(*) FROM seg_reviews").fetchone()[0]
        if n_reviews == self._trained_on:
            return
        self._trained_on = n_reviews
        self.models = {}
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
        except ImportError:
            return
        rows = conn.execute("""
            SELECT r.content, r.switch_label, r.interruption_kind, g.*
            FROM seg_reviews r JOIN segments g ON g.id = r.segment_id""").fetchall()
        if not rows:
            return
        for head, col in self.HEAD_COL.items():
            X, y = [], []
            for r in rows:
                label = r[col]
                if not label or label not in HEADS[head]:
                    continue
                seg = dict(r)
                seg["duration"] = (seg.get("end") or seg["start"]) - seg["start"]
                prev = conn.execute("SELECT * FROM segments WHERE start < ? AND idle=0 ORDER BY start DESC LIMIT 1",
                                    (seg["start"],)).fetchone()
                X.append(features(cfg, seg, dict(prev) if prev else None))
                y.append(label)
            if len(X) >= 12 and len(set(y)) >= 2:
                m = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=2000, C=1.0))
                m.fit(X, y)
                self.models[head] = m

    def predict(self, cfg, seg: dict, prev: dict | None) -> dict | None:
        if not self.models:
            return None
        out = {}
        f = features(cfg, seg, prev)
        for head, labels in HEADS.items():
            m = self.models.get(head)
            if m is None:
                out[head] = _uniformish(labels)
                continue
            probs = m.predict_proba([f])[0]
            classes = m.classes_
            d = {l: 1e-4 for l in labels}
            for c, p in zip(classes, probs):
                d[c] = float(p)
            out[head] = _norm(d)
        return out


_learned = LearnedSegModel()


# ---- storage, scoring, weighting, combination -------------------------------------------------

def store_prediction(conn, segment_id: int, model: str, preds: dict, cost: float = 0.0) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO seg_predictions(segment_id, model, p_content, p_switch, p_kind, cost_usd, created)"
        " VALUES (?,?,?,?,?,?,?)",
        (segment_id, model, json.dumps(preds["content"]), json.dumps(preds["switch"]),
         json.dumps(preds.get("kind") or {}), cost, time.time()))


def scores(conn) -> dict:
    """{(model, head): {n, accuracy, logloss}} vs the user's seg_reviews; also a virtual
    'local' model = the heuristic+learned combination, used for the deferral decision."""
    rows = conn.execute(f"""
        SELECT p.segment_id, p.model, p.p_content, p.p_switch, p.p_kind,
               r.content, r.switch_label, r.interruption_kind
        FROM seg_predictions p JOIN seg_reviews r ON r.segment_id = p.segment_id
        ORDER BY r.created DESC LIMIT {WINDOW * 4}""").fetchall()
    by_seg: dict = {}
    for r in rows:
        by_seg.setdefault(r["segment_id"], {"gold": {"content": r["content"], "switch": r["switch_label"],
                                                     "kind": r["interruption_kind"]}, "preds": {}})
        by_seg[r["segment_id"]]["preds"][r["model"]] = {
            "content": json.loads(r["p_content"]), "switch": json.loads(r["p_switch"]),
            "kind": json.loads(r["p_kind"] or "{}")}
    out: dict = {}
    for seg in by_seg.values():
        preds = dict(seg["preds"])
        local = {m: preds[m] for m in ("heuristic", "learned") if m in preds}
        if local:
            preds["local"] = {h: combine_head({m: p[h] for m, p in local.items()},
                                              {m: DEFAULT_WEIGHTS.get(m, 0.35) for m in local})
                              for h in HEADS}
        for model, p in preds.items():
            for head in HEADS:
                gold = seg["gold"][head]
                if not gold or gold not in HEADS[head]:
                    continue
                d = p.get(head) or {}
                if not d:
                    continue
                s = out.setdefault((model, head), {"n": 0, "correct": 0, "logloss": 0.0})
                if s["n"] >= WINDOW:
                    continue
                s["n"] += 1
                s["correct"] += int(max(d, key=d.get) == gold)
                s["logloss"] += -math.log(max(d.get(gold, 1e-6), 1e-6))
    for s in out.values():
        s["accuracy"] = s["correct"] / s["n"] if s["n"] else None
        s["logloss"] = s["logloss"] / s["n"] if s["n"] else None
    return out


def weights_for(sc: dict, head: str, models: list[str]) -> dict:
    w = {}
    for m in models:
        s = sc.get((m, head))
        if s and s["n"] >= MIN_SCORED:
            w[m] = math.exp(-2.0 * s["logloss"])
        else:
            w[m] = DEFAULT_WEIGHTS.get(m, 0.35)
    return w


def combine_head(preds: dict, weights: dict) -> dict:
    """preds: {model: {label: prob}} -> weighted average distribution."""
    out: dict = {}
    total = 0.0
    for m, d in preds.items():
        w = weights.get(m, 0.35)
        total += w
        for l, p in d.items():
            out[l] = out.get(l, 0.0) + w * p
    return _norm({l: p / (total or 1.0) for l, p in out.items()})


def local_predict(conn, cfg, seg: dict, prev: dict | None) -> dict:
    """heuristic + learned predictions for one segment: {model: {head: {label: prob}}}."""
    _learned.learn(conn, cfg)
    preds = {"heuristic": heuristic_predict(cfg, seg, prev)}
    lp = _learned.predict(cfg, seg, prev)
    if lp is not None:
        preds["learned"] = lp
    return preds


def can_defer(sc: dict, combined: dict, cfg) -> bool:
    """Defer to the local models when their combined judgment is confident AND their track
    record on each needed head has earned it (enough user feedback, high accuracy)."""
    heads = ["content", "switch"]
    if max(combined["switch"], key=combined["switch"].get) == "interruption":
        heads.append("kind")
    for head in heads:
        s = sc.get(("local", head))
        if not s or s["n"] < cfg.defer_min_feedback or (s["accuracy"] or 0) < cfg.defer_accuracy:
            return False
        if max(combined[head].values()) < cfg.defer_confidence:
            return False
    return True

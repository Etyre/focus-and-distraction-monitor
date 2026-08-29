"""Logistic-regression classifier on switch features, trained on human reviews."""
from __future__ import annotations

import datetime as dt
import math
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .. import LABELS, db
from .base import Classifier, SwitchContext, normalize

MIN_REVIEWS = 15


def _bucket(seconds: float) -> str:
    if seconds < 15: return "<15s"
    if seconds < 60: return "<1m"
    if seconds < 300: return "<5m"
    if seconds < 1200: return "<20m"
    return ">20m"


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]{3,}", (s or "").lower())][:12]


def features_from_row(r: dict, to_end: float | None = None) -> dict:
    """r has from_app/from_domain/from_start/from_end/to_app/to_domain/to_title/to_url/to_start/to_end/ts."""
    f = {}
    fa, fd = r.get("from_app") or "", r.get("from_domain") or ""
    ta, td = r["to_app"], r["to_domain"] or ""
    f[f"to_app={ta}"] = 1
    f[f"to_dom={td}"] = 1
    f[f"from_app={fa}"] = 1
    f[f"from_dom={fd}"] = 1
    f[f"pair={fa}|{fd}->{ta}|{td}"] = 1
    f["same_app"] = int(fa == ta)
    f["same_dom"] = int(fd == td and td != "")
    hour = dt.datetime.fromtimestamp(r["ts"]).hour
    f[f"hour={hour // 3}"] = 1
    if r.get("from_start") is not None:
        f[f"from_dur={_bucket((r.get('from_end') or r['ts']) - r['from_start'])}"] = 1
    end = r.get("to_end") or to_end or r["to_start"]
    f[f"to_dur={_bucket(end - r['to_start'])}"] = 1
    for t in _tokens(r.get("to_title", "")):
        f[f"tok={t}"] = 1
    f["toggl_running"] = int(bool(r.get("toggl_before")))
    f["toggl_started"] = int(bool(r.get("toggl_started")))
    for t in _tokens((r.get("toggl_before") or {}).get("tags", "")):
        f[f"toggl_tag={t}"] = 1
    return f


class LearnedClassifier(Classifier):
    name = "learned"

    def __init__(self):
        self.model = None
        self.classes: list[str] = []
        self.n_train = 0

    def learn(self, conn) -> None:
        rows = db.reviews_with_context(conn)
        labels = [r["label"] for r in rows]
        if len(rows) < MIN_REVIEWS or len(set(labels)) < 2:
            self.model = None
            return
        from .. import toggl
        for r in rows:
            try:
                r["toggl_before"] = toggl.entry_at(conn, r["ts"] - 5)
                r["toggl_started"] = toggl.entry_started_near(conn, r["ts"])
            except Exception:
                r["toggl_before"] = r["toggl_started"] = None
        X = [features_from_row(r) for r in rows]
        self.model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=2000, C=1.0))
        self.model.fit(X, labels)
        self.classes = list(self.model.classes_)
        self.n_train = len(rows)

    def predict(self, ctx: SwitchContext):
        if self.model is None:
            return None
        fr, to = ctx.from_seg, ctx.to_seg
        row = {
            "ts": ctx.ts,
            "from_app": fr["app"] if fr else "", "from_domain": fr["domain"] if fr else "",
            "from_start": fr["start"] if fr else None, "from_end": fr["end"] if fr else None,
            "to_app": to["app"], "to_domain": to["domain"], "to_title": to["title"],
            "to_url": to["url"], "to_start": to["start"], "to_end": to["end"],
            "toggl_before": ctx.toggl_before, "toggl_started": ctx.toggl_started,
        }
        pr = self.model.predict_proba([features_from_row(row)])[0]
        probs = {k: 0.02 for k in LABELS}
        for c, p in zip(self.classes, pr):
            probs[c] = float(p)
        return normalize(probs), f"logistic regression trained on {self.n_train} reviews", 0.0

"""Local dashboard: timeline, daily metrics, review queue, model accuracy."""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import LABEL_HELP, LABEL_NAMES, LABELS, db, ensemble, stats, toggl
from .config import DATA_DIR, load_config, logical_date

app = FastAPI(title="Focus Monitor")
cfg = load_config()
HTML = (Path(__file__).parent / "dashboard.html").read_text()


def conn():
    return db.connect()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/day")
def api_day(day: str | None = None):
    d = dt.date.fromisoformat(day) if day else logical_date()
    c = conn()
    t0, t1 = stats.day_bounds(d)
    segs = stats.labelled_segments(c, t0, t1)
    se = stats.spans_and_events(segs, conn=c)
    # "Needs review" marks the PRIMARY review stream: segments with an open question, or an
    # unresolved uncertainty flag (evaluator self-doubt, audit, feedback propagation). Not
    # gated behind span confirmation - questions are tier one.
    qsegs = {r[0] for r in c.execute("SELECT segment_id FROM review_questions WHERE status='open'")}
    usegs = {r[0] for r in c.execute("""SELECT e.segment_id FROM seg_evals e
        LEFT JOIN seg_reviews rv ON rv.segment_id = e.segment_id
        WHERE e.uncertain = 1 AND rv.segment_id IS NULL""")}
    for s in segs:
        s["needs_review"] = (s["id"] in qsegs or s["id"] in usegs) and s.get("source") != "review"
    selected = {s["id"] for s in segs if s["needs_review"]}
    reviewed = c.execute("SELECT COUNT(*) FROM reviews r JOIN switches s ON s.id=r.switch_id WHERE s.ts>=? AND s.ts<?",
                         (t0, t1)).fetchone()[0]
    slim = [{k: s.get(k) for k in ("id", "app", "title", "url", "domain", "clip_start", "clip_end", "duration",
                                   "state", "label", "switch_id", "uncertain", "needs_review", "source", "probs", "neutral", "first_screenshot", "activity", "content", "toggl_id", "toggl_desc")} for s in segs]
    from . import summaries as _summ
    def _summary(r):
        row = _summ.lookup(c, r["start"], r["end"])
        return row["summary"] if row else None
    def _coherence(r):
        row = _summ.lookup(c, r["start"], r["end"])
        return row["issue"] if row is not None and not row["coherent"] else None
    def _title(r):
        row = _summ.lookup(c, r["start"], r["end"])
        return row["title"] if row is not None else None
    return {"day": d.isoformat(), "t0": t0, "t1": t1, "metrics": stats.daily_metrics(c, d), "segments": slim,
            "spend_total": db.spend_total(c), "budget_total": cfg.total_budget_usd,
            "to_review": len(selected), "reviewed": reviewed,  # open questions + unresolved flags today
            "short_breaks": stats.short_breaks(c, t0, t1),
            "toggl": [{"start": max(e["start"], t0), "end": min(e["stop"] or time.time(), t1), "description": e["description"],
                       "project": e["project"], "tags": e["tags"]} for e in toggl.entries_between(c, t0, t1)],
            "count_events": cfg.count_events, "min_focus_span_min": cfg.min_focus_span_min,
            "focus_spans": [{"start": r["start"], "end": r["end"], "duration": r["duration"], "ended_by": r["ended_by"],
                             "focus_min": r["focus_min"], "detours": len(r["detours"]), "detour_min": r["detour_min"],
                             "subtype": r.get("subtype", "focus"), "fully_absorbed": r.get("fully_absorbed", False),
                             "mixed": r.get("mixed_read_create", False), "summary": _summary(r),
                             "coherence_issue": _coherence(r), "title": _title(r),
                             "events": [{"start": e["start"], "end": e["end"], "kind": e["state"]} for e in r["detours"]],
                             "apps": _apps_by_time(r["segments"]),
                             "toggl": _toggl_with_coverage(c, r["start"], r["end"])}
                            for r in se["focus_spans"]],
            "distractions": [{"start": r["start"], "end": r["end"], "duration": r["duration"], "is_event": r["is_event"],
                              "where": sorted({s["domain"] or s["app"] for s in r["segments"]})} for r in se["distractions"]],
            "disqualified": [{"start": r["start"], "end": r["end"], "duration": r["duration"],
                              "reason": r.get("disq_reason", "does not meet the focused-work standard"),
                              "summary": _summary(r)}
                             for r in se.get("disqualified", [])]}


@app.get("/api/metrics")
def api_metrics(days: int = 30):
    return {"count_events": cfg.count_events, "days": stats.metrics_range(conn(), days)}


@app.get("/api/accuracy")
def api_accuracy(days: int = 60):
    c = conn()
    return {"daily": stats.accuracy_over_time(c, days), "scores": ensemble.model_scores(c),
            "spend_today": db.spend_today(c), "budget": cfg.daily_budget_usd,
            "spend_total": db.spend_total(c), "budget_total": cfg.total_budget_usd,
            "reviews": c.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]}


def _unconfirmed_ranges(c, t0: float, t1: float) -> list[tuple[float, float]]:
    se = stats.spans_and_events(stats.labelled_segments(c, t0, t1), conn=c)
    return [(sp["start"], sp["end"]) for sp in se["focus_spans"] + se["disqualified"]
            if not sp.get("confirmed")]


CTX_S = 600.0  # timeline context shown on each side of a reviewed span


def _toggl_with_coverage(c, start: float, end: float) -> list[str]:
    """Toggl descriptions annotated with how much of the span each actually covered, so a
    briefly-running entry cannot masquerade as describing the whole span."""
    cov: dict = {}
    for e in toggl.entries_between(c, start, end):
        if not e["description"]:
            continue
        ov = min(e["stop"] or time.time(), end) - max(e["start"], start)
        if ov > 0:
            cov[e["description"]] = cov.get(e["description"], 0.0) + ov
    dur = end - start
    return [d if ov >= 0.9 * dur else f"{d} ({ov / 60:.0f}m of {dur / 60:.0f}m)"
            for d, ov in sorted(cov.items(), key=lambda kv: -kv[1])]


def _apps_by_time(segments) -> list[str]:
    from collections import Counter
    c = Counter()
    for s in segments:
        c[s["app"]] += s.get("duration", 0.0)
    return [a for a, _ in c.most_common()]


def _span_payload(c, sp, day: str, day_segs: list[dict] | None = None) -> dict:
    from . import summaries as _summ
    row = _summ.lookup(c, sp["start"], sp["end"])
    coherence_issue = (row["issue"] if row is not None and not row["coherent"] else None)
    w0, w1 = sp["start"] - CTX_S, sp["end"] + CTX_S
    keys = ("clip_start", "clip_end", "duration", "state", "label", "app", "domain", "title",
            "url", "activity", "content", "switch_id", "source", "first_screenshot", "toggl_id", "toggl_desc")
    in_span_ids = {id(s) for s in sp["segments"]}
    segs = []
    for s in (day_segs or sp["segments"]):
        if s["clip_end"] <= w0 or s["clip_start"] >= w1 or s["duration"] <= 0:
            continue
        d = {k: s.get(k) for k in keys}
        d["in_span"] = id(s) in in_span_ids or (day_segs is None)
        segs.append(d)
    return {"day": day, "start": sp["start"], "end": sp["end"], "duration": sp["duration"],
            "focus_min": sp["focus_min"], "subtype": sp.get("subtype"),
            "disq_reason": sp.get("disq_reason"), "detours": len(sp.get("detours", [])),
            "detour_min": sp.get("detour_min", 0), "ended_by": sp.get("ended_by"),
            "summary": row["summary"] if row else None,
            "title": row["title"] if row else None,
            "coherence_issue": coherence_issue,
            "apps": _apps_by_time(sp["segments"]),
            "toggl": _toggl_with_coverage(c, sp["start"], sp["end"]),
            "toggl_entries": [{"start": max(e["start"], w0), "end": min(e["stop"] or time.time(), w1),
                               "description": e["description"], "project": e["project"]}
                              for e in toggl.entries_between(c, w0, w1)],
            "segments": segs}


@app.get("/api/span/queue")
def api_span_queue(days: int = 7):
    """Every finished, unconfirmed hypothesized span of the last N days: today's first
    (chronological within the day), then yesterday's, and so on back."""
    c = conn()
    out = []
    today = logical_date()
    for i in range(days):
        day = today - dt.timedelta(days=i)
        t0, t1 = stats.day_bounds(day)
        day_segs = stats.labelled_segments(c, t0, t1)
        se = stats.spans_and_events(day_segs, conn=c)
        for sp in sorted(se["focus_spans"] + se["disqualified"], key=lambda s: s["start"]):
            if sp.get("confirmed") or sp.get("ended_by") in ("ongoing", "awaiting evaluation"):
                continue
            out.append(_span_payload(c, sp, day.isoformat(), day_segs))
    return out


@app.get("/api/span/at")
def api_span_at(ts: float):
    """The hypothesized span (qualifying or disqualified) containing ts, as a review payload."""
    c = conn()
    d = logical_date(ts)
    t0, t1 = stats.day_bounds(d)
    day_segs = stats.labelled_segments(c, t0, t1)
    se = stats.spans_and_events(day_segs, conn=c)
    for sp in se["focus_spans"] + se["disqualified"]:
        if sp["start"] - 1 <= ts < sp["end"] + 1:
            p = _span_payload(c, sp, d.isoformat(), day_segs)
            p["confirmed"] = bool(sp.get("confirmed"))
            rv = stats._span_review_for(c.execute("SELECT * FROM span_reviews").fetchall(), sp)
            p["review"] = {"verdict": rv["verdict"], "subtype": rv["subtype"]} if rv else None
            return p
    raise HTTPException(404, "no span at that time")


class SpanReview(BaseModel):
    start: float
    end: float
    verdict: str
    subtype: str | None = None
    note: str | None = None


@app.post("/api/span/review")
def api_span_review(r: SpanReview):
    if r.verdict not in ("focus", "not_focus"):
        raise HTTPException(400, "bad verdict")
    if r.subtype is not None and r.subtype not in stats.SUBTYPES:
        raise HTTPException(400, "bad subtype")
    c = conn()
    with db.tx(c):
        c.execute("DELETE FROM span_reviews WHERE start >= ? AND end <= ?", (r.start - 60, r.end + 60))
        c.execute("INSERT INTO span_reviews(start, end, verdict, subtype, note, created) VALUES (?,?,?,?,?,?)",
                  (r.start, r.end, r.verdict, r.subtype, r.note, time.time()))
    return {"ok": True}


@app.get("/api/efficiency")
def api_efficiency():
    from . import seg_models
    return seg_models.efficiency_report(conn())


@app.get("/api/efforts/summary")
def api_efforts_summary():
    from . import efforts
    return efforts.summary(conn())


@app.get("/api/efforts/trials")
def api_efforts_trials(role: str | None = None, effort: str | None = None, limit: int = 50):
    from . import efforts
    return efforts.trials(conn(), role, effort, limit)


@app.get("/api/efforts/trial/{trial_id}")
def api_efforts_trial(trial_id: int):
    from . import efforts
    d = efforts.trial_detail(conn(), trial_id)
    if not d:
        raise HTTPException(404)
    return d


@app.get("/api/review/queue")
def api_review_queue(limit: int = 30, all: bool = False):
    c = conn()
    if all:
        rows = c.execute("""
            SELECT s.id FROM switches s JOIN ensemble e ON e.switch_id=s.id
            LEFT JOIN reviews r ON r.switch_id=s.id
            WHERE r.switch_id IS NULL AND s.status!='transit' ORDER BY s.ts DESC LIMIT ?""", (limit,)).fetchall()
        ids = [r[0] for r in rows]
    else:
        # The ranked selection: top-N per day (highest stakes x uncertainty), most recent day first.
        # Two-tier review: a switch inside a not-yet-confirmed span is held back until the
        # span itself has been reviewed.
        ids = []
        today = logical_date()
        for i in range(14):
            t0, t1 = stats.day_bounds(today - dt.timedelta(days=i))
            day_ids = stats.select_for_review(c, t0, t1, cfg.max_reviews_per_day)
            if day_ids:
                unconf = _unconfirmed_ranges(c, t0, t1)
                if unconf:
                    q = ",".join("?" * len(day_ids))
                    ts_of = dict(c.execute(f"SELECT id, ts FROM switches WHERE id IN ({q})", day_ids).fetchall())
                    day_ids = [i2 for i2 in day_ids
                               if not any(a - 1 <= ts_of.get(i2, 0) < b + 1 for a, b in unconf)]
            ids += day_ids
            if len(ids) >= limit:
                break
        ids = ids[:limit]
    return [db.switch_full(c, i) for i in ids]


@app.get("/api/switch/latest")
def api_latest():
    r = conn().execute("SELECT id FROM switches WHERE status!='transit' ORDER BY ts DESC LIMIT 1").fetchone()
    return {"id": r[0] if r else None}


@app.get("/api/switch/{switch_id}/adjacent")
def api_adjacent(switch_id: int, dir: str = "next"):
    c = conn()
    sw = c.execute("SELECT ts FROM switches WHERE id=?", (switch_id,)).fetchone()
    if not sw:
        raise HTTPException(404)
    if dir == "prev":
        r = c.execute("SELECT id FROM switches WHERE ts<? AND status!='transit' ORDER BY ts DESC LIMIT 1", (sw["ts"],)).fetchone()
    else:
        r = c.execute("SELECT id FROM switches WHERE ts>? AND status!='transit' ORDER BY ts ASC LIMIT 1", (sw["ts"],)).fetchone()
    return {"id": r[0] if r else None}


@app.get("/api/switch/{switch_id}")
def api_switch(switch_id: int):
    c = conn()
    full = db.switch_full(c, switch_id)
    if not full:
        raise HTTPException(404)
    full["before"] = [dict(r) for r in db.segments_before(c, (full["from"] or full["to"])["start"], 5)]
    full["after"] = [dict(r) for r in db.segments_after(c, full["to"]["start"], 4)]
    ts = full["switch"]["ts"]
    to_id = full["to"]["id"]
    import json as _json
    qrow = c.execute("SELECT * FROM review_questions WHERE segment_id=? AND status='open' LIMIT 1", (to_id,)).fetchone()
    full["question"] = None
    if qrow:
        full["question"] = dict(qrow)
        full["question"]["options"] = _json.loads(full["question"]["options"])
    full["to_shots"] = [dict(r) for r in c.execute(
        "SELECT ts, path, display FROM segment_shots WHERE segment_id=? ORDER BY ts", (to_id,))]
    full["to_eval"] = (lambda r: dict(r) if r else None)(
        c.execute("SELECT * FROM seg_evals WHERE segment_id=?", (to_id,)).fetchone())
    full["to_seg_review"] = (lambda r: dict(r) if r else None)(
        c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (to_id,)).fetchone())
    te = next((e for e in toggl.entries_between(c, ts - 1, ts + 1)
               if e["start"] <= ts and (e["stop"] is None or e["stop"] > ts)), None)
    full["toggl"] = {"description": te["description"], "project": te["project"], "tags": te["tags"]} if te else None
    # The span this switch lands in (qualifying or disqualified), for audit context.
    from . import summaries as _summ
    full["span"] = None
    full["timeline"] = []
    try:
        ctx_segs = stats.labelled_segments(c, ts - 6 * 3600, ts + 6 * 3600)
        keys = ("id", "clip_start", "clip_end", "duration", "state", "label", "app", "domain", "title",
                "url", "activity", "content", "switch_id", "source", "first_screenshot", "toggl_id", "toggl_desc")
        full["timeline"] = [{k: s.get(k) for k in keys} for s in ctx_segs
                            if s["clip_end"] > ts - 900 and s["clip_start"] < ts + 900 and s["duration"] > 0]
        tl_ids = [s["id"] for s in full["timeline"]]
        full["timeline_shots"] = []
        if tl_ids:
            qs2 = ",".join("?" * len(tl_ids))
            full["timeline_shots"] = [dict(r) for r in c.execute(
                f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs2}) ORDER BY ts",
                tl_ids)]
        if not full["timeline_shots"]:  # pre-shot-history segments
            full["timeline_shots"] = [{"segment_id": s["id"], "ts": s["clip_start"],
                                       "path": s["first_screenshot"], "display": 0}
                                      for s in full["timeline"] if s.get("first_screenshot")]
        se = stats.spans_and_events(ctx_segs, conn=c)
        spans = se["focus_spans"] + se["disqualified"]
        spans_sorted = sorted(spans, key=lambda s2: s2["start"])
        def _first_switch(sp2):
            for sg in sp2["segments"]:
                r2 = c.execute("SELECT id FROM switches WHERE to_segment=? LIMIT 1", (sg["id"],)).fetchone()
                if r2:
                    return r2["id"]
            return None
        nxt = next((sp2 for sp2 in spans_sorted if sp2["start"] > ts + 1), None)
        prv = next((sp2 for sp2 in reversed(spans_sorted) if sp2["start"] < ts - 1), None)
        full["next_span_switch"] = _first_switch(nxt) if nxt else None
        full["prev_span_switch"] = _first_switch(prv) if prv else None
        cand = next((sp for sp in spans if sp["start"] - 1 <= ts < sp["end"] + 1), None)
        rel = "inside"
        if cand is None:  # a span retroactively shortened ends just before the switch that left it
            ended = [sp for sp in spans if 0 <= ts - sp["end"] <= 300]
            if ended:
                cand, rel = max(ended, key=lambda s: s["end"]), "leaving"
        if cand is not None:
            row = _summ.lookup(c, cand["start"], cand["end"])
            full["span"] = {"start": cand["start"], "end": cand["end"], "subtype": cand.get("subtype"),
                            "disq_reason": cand.get("disq_reason"), "relation": rel,
                            "is_start": rel == "inside" and abs(full["to"]["start"] - cand["start"]) < 1,
                            "summary": row["summary"] if row else None}
    except Exception:
        pass
    return full


_chat = None
def get_chat():
    global _chat
    if _chat is None:
        from .chat import ReviewChat
        _chat = ReviewChat(cfg)
    return _chat


_span_chat = None
def get_span_chat():
    global _span_chat
    if _span_chat is None:
        from .chat import SpanChat
        _span_chat = SpanChat(cfg)
    return _span_chat


class SpanChatReq(BaseModel):
    start: float
    end: float
    messages: list[dict]


QCTX_S = 900.0  # timeline context on each side of a question's segment (matches ~what the generator saw)


@app.get("/api/questions/queue")
def api_questions_queue(limit: int = 30):
    c = conn()
    import json as _json
    out = []
    keys = ("id", "clip_start", "clip_end", "duration", "state", "label", "app", "domain", "title",
            "content", "switch_id", "source", "first_screenshot", "toggl_id", "toggl_desc")
    for r in c.execute("""SELECT q.*, g.start AS seg_start, COALESCE(g.end, g.start) AS seg_end,
                                 g.app, g.title, g.domain,
                                 (SELECT id FROM switches WHERE to_segment = q.segment_id LIMIT 1) AS switch_id
                          FROM review_questions q JOIN segments g ON g.id = q.segment_id
                          WHERE q.status = 'open' ORDER BY q.created DESC LIMIT ?""", (limit,)):
        d = dict(r)
        d["options"] = _json.loads(d["options"])
        segs = stats.labelled_segments(c, d["seg_start"] - QCTX_S, d["seg_end"] + QCTX_S)
        d["timeline"] = [{k: s.get(k) for k in keys} for s in segs if s["duration"] > 0]
        out.append(d)
    return out


class QAnswer(BaseModel):
    option_index: int
    note: str | None = None


@app.post("/api/questions/{qid}/answer")
def api_question_answer(qid: int, a: QAnswer):
    import json as _json
    c = conn()
    q = c.execute("SELECT * FROM review_questions WHERE id=?", (qid,)).fetchone()
    if not q:
        raise HTTPException(404)
    opts = _json.loads(q["options"])
    if not (0 <= a.option_index < len(opts)):
        raise HTTPException(400, "bad option")
    o = opts[a.option_index]
    with db.tx(c):
        if o.get("switch_label") or o.get("content"):
            prev = c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (o["segment_id"],)).fetchone()
            c.execute("INSERT OR REPLACE INTO seg_reviews(segment_id, content, switch_label, interruption_kind, note, created)"
                      " VALUES (?,?,?,?,?,?)",
                      (o["segment_id"],
                       o.get("content") or (prev["content"] if prev else None),
                       o.get("switch_label") or (prev["switch_label"] if prev else None),
                       o.get("interruption_kind") if o.get("switch_label") == "interruption" else (None if o.get("switch_label") else (prev["interruption_kind"] if prev else None)),
                       (a.note.strip() if a.note and a.note.strip() else f"(answered review question: {o['label']})"), time.time()))
            from . import propagation
            propagation.note_event(c, o["segment_id"])
            old = _NEW2OLD.get((o.get("switch_label"),
                                o.get("interruption_kind") if o.get("switch_label") == "interruption" else None))
            sw = c.execute("SELECT id FROM switches WHERE to_segment=?", (o["segment_id"],)).fetchone()
            if old and sw:
                c.execute("INSERT OR REPLACE INTO reviews(switch_id, label, note, created) VALUES (?,?,?,?)",
                          (sw["id"], old, (a.note.strip() if a.note and a.note.strip() else o["label"]), time.time()))
                c.execute("UPDATE switches SET status='reviewed' WHERE id=?", (sw["id"],))
        c.execute("UPDATE review_questions SET status='answered', answer=?, note=?, answered_at=? WHERE id=?",
                  (o["label"], (a.note.strip() or None) if a.note else None, time.time(), qid))
    return {"ok": True}


@app.post("/api/questions/{qid}/dismiss")
def api_question_dismiss(qid: int):
    c = conn()
    with db.tx(c):
        c.execute("UPDATE review_questions SET status='dismissed', answered_at=? WHERE id=?",
                  (time.time(), qid))
    return {"ok": True}


class EvalChatReq(BaseModel):
    messages: list[dict]


@app.post("/api/segment/{segment_id}/evaluator-chat")
def api_evaluator_chat(segment_id: int, req: EvalChatReq):
    from . import evaluator
    try:
        return {"reply": evaluator.chat_reply(conn(), cfg, segment_id, req.messages)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/span/chat")
def api_span_chat(req: SpanChatReq):
    try:
        return {"reply": get_span_chat().reply(conn(), req.start, req.end, req.messages)}
    except Exception as e:
        raise HTTPException(500, str(e))


class ChatReq(BaseModel):
    messages: list[dict]


@app.post("/api/switch/{switch_id}/chat")
def api_chat(switch_id: int, req: ChatReq):
    c = conn()
    if not db.switch_full(c, switch_id):
        raise HTTPException(404)
    try:
        return {"reply": get_chat().reply(c, switch_id, req.messages)}
    except Exception as e:
        raise HTTPException(500, str(e))


class SegEvalReview(BaseModel):
    content: str | None = None
    switch_label: str | None = None
    interruption_kind: str | None = None
    note: str | None = None


# The user's edit of the evaluator's judgment maps onto the legacy 4-label review scheme so the
# older per-switch models keep learning from the same corrections.
_NEW2OLD = {("continuation", None): "continuation", ("return", None): "task_change",
            ("interruption", "self_distraction"): "distraction",
            ("interruption", "focus_start"): "task_change",
            ("interruption", "detour"): "interruption"}


@app.post("/api/segment/{segment_id}/eval-review")
def api_seg_eval_review(segment_id: int, r: SegEvalReview):
    from .evaluator import CONTENTS, KINDS, SWITCHES
    if r.content is not None and r.content not in CONTENTS:
        raise HTTPException(400, "bad content label")
    if r.switch_label is not None and r.switch_label not in SWITCHES:
        raise HTTPException(400, "bad switch label")
    if r.interruption_kind is not None and r.interruption_kind not in KINDS:
        raise HTTPException(400, "bad interruption kind")
    c = conn()
    if not c.execute("SELECT 1 FROM segments WHERE id=?", (segment_id,)).fetchone():
        raise HTTPException(404)
    with db.tx(c):
        c.execute("INSERT OR REPLACE INTO seg_reviews(segment_id, content, switch_label, interruption_kind, note, created)"
                  " VALUES (?,?,?,?,?,?)",
                  (segment_id, r.content, r.switch_label, r.interruption_kind, r.note, time.time()))
        if r.content or r.switch_label:
            from . import propagation
            propagation.note_event(c, segment_id)
        old = _NEW2OLD.get((r.switch_label, r.interruption_kind if r.switch_label == "interruption" else None))
        sw = c.execute("SELECT id FROM switches WHERE to_segment=?", (segment_id,)).fetchone()
        if old and sw:
            c.execute("INSERT OR REPLACE INTO reviews(switch_id, label, note, created) VALUES (?,?,?,?)",
                      (sw["id"], old, r.note, time.time()))
            c.execute("UPDATE switches SET status='reviewed' WHERE id=?", (sw["id"],))
    return {"ok": True}


@app.delete("/api/segment/{segment_id}/eval-review")
def api_seg_eval_unreview(segment_id: int):
    c = conn()
    with db.tx(c):
        c.execute("DELETE FROM seg_reviews WHERE segment_id=?", (segment_id,))
    return {"ok": True}


class Review(BaseModel):
    label: str
    note: str | None = None


@app.post("/api/switch/{switch_id}/review")
def api_review(switch_id: int, review: Review):
    if review.label not in LABELS:
        raise HTTPException(400, "bad label")
    c = conn()
    if not c.execute("SELECT 1 FROM switches WHERE id=?", (switch_id,)).fetchone():
        raise HTTPException(404)
    with db.tx(c):
        c.execute("INSERT OR REPLACE INTO reviews(switch_id, label, note, created) VALUES (?,?,?,?)",
                  (switch_id, review.label, review.note, time.time()))
        c.execute("UPDATE switches SET status='reviewed' WHERE id=?", (switch_id,))
    return {"ok": True}


@app.delete("/api/switch/{switch_id}/review")
def api_unreview(switch_id: int):
    c = conn()
    with db.tx(c):
        c.execute("DELETE FROM reviews WHERE switch_id=?", (switch_id,))
        c.execute("UPDATE switches SET status='classified' WHERE id=?", (switch_id,))
    return {"ok": True}


@app.get("/api/labels")
def api_labels():
    return {k: {"name": LABEL_NAMES[k], "help": LABEL_HELP[k]} for k in LABELS}


@app.get("/shot/{name}")
def shot(name: str):
    p = DATA_DIR / "screenshots" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


def serve():
    import uvicorn
    uvicorn.run(app, host=cfg.web_host, port=cfg.web_port, log_level="warning")

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
    se = stats.spans_and_events(segs)
    selected = set(stats.select_for_review(c, t0, t1, cfg.max_reviews_per_day))
    for s in segs:
        s["needs_review"] = s.get("switch_id") in selected and s.get("source") != "review"
    reviewed = c.execute("SELECT COUNT(*) FROM reviews r JOIN switches s ON s.id=r.switch_id WHERE s.ts>=? AND s.ts<?",
                         (t0, t1)).fetchone()[0]
    slim = [{k: s.get(k) for k in ("id", "app", "title", "url", "domain", "clip_start", "clip_end", "duration",
                                   "state", "label", "switch_id", "uncertain", "needs_review", "source", "probs", "neutral", "first_screenshot")} for s in segs]
    return {"day": d.isoformat(), "t0": t0, "t1": t1, "metrics": stats.daily_metrics(c, d), "segments": slim,
            "to_review": len(selected), "reviewed": reviewed,
            "short_breaks": stats.short_breaks(c, t0, t1),
            "toggl": [{"start": max(e["start"], t0), "end": min(e["stop"] or time.time(), t1), "description": e["description"],
                       "project": e["project"], "tags": e["tags"]} for e in toggl.entries_between(c, t0, t1)],
            "count_events": cfg.count_events, "min_focus_span_min": cfg.min_focus_span_min,
            "focus_spans": [{"start": r["start"], "end": r["end"], "duration": r["duration"], "ended_by": r["ended_by"],
                             "focus_min": r["focus_min"], "detours": len(r["detours"]), "detour_min": r["detour_min"],
                             "subtype": r.get("subtype", "focus"), "fully_absorbed": r.get("fully_absorbed", False),
                             "events": [{"start": e["start"], "end": e["end"], "kind": e["state"]} for e in r["detours"]],
                             "apps": sorted({s["app"] for s in r["segments"]}),
                             "toggl": sorted({e["description"] for e in toggl.entries_between(c, r["start"], r["end"]) if e["description"]})}
                            for r in se["focus_spans"]],
            "distractions": [{"start": r["start"], "end": r["end"], "duration": r["duration"], "is_event": r["is_event"],
                              "where": sorted({s["domain"] or s["app"] for s in r["segments"]})} for r in se["distractions"]]}


@app.get("/api/metrics")
def api_metrics(days: int = 30):
    return {"count_events": cfg.count_events, "days": stats.metrics_range(conn(), days)}


@app.get("/api/accuracy")
def api_accuracy(days: int = 60):
    c = conn()
    return {"daily": stats.accuracy_over_time(c, days), "scores": ensemble.model_scores(c),
            "spend_today": db.spend_today(c), "budget": cfg.daily_budget_usd,
            "reviews": c.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]}


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
        ids = []
        today = logical_date()
        for i in range(14):
            t0, t1 = stats.day_bounds(today - dt.timedelta(days=i))
            ids += stats.select_for_review(c, t0, t1, cfg.max_reviews_per_day)
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
    return full


_chat = None
def get_chat():
    global _chat
    if _chat is None:
        from .chat import ReviewChat
        _chat = ReviewChat(cfg)
    return _chat


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

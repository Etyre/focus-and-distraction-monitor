"""Local dashboard: timeline, daily metrics, review queue, model accuracy."""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import LABEL_HELP, LABEL_NAMES, LABELS, db, edits, ensemble, stats, toggl
from .config import CONFIG_PATH, DATA_DIR, load_config, logical_date, screenshot_file

app = FastAPI(title="Focus Monitor")
cfg = load_config()
HTML = (Path(__file__).parent / "dashboard.html").read_text()
log = logging.getLogger("web")
STARTED = time.time()
# Set by the menu bar app: a callable that restarts the process. None when serving headless.
restart_hook = None


def conn():
    return db.connect()


_qgen_busy = threading.Lock()


def _kick_question_generation():
    """Generate questions for flagged-but-questionless segments right now, in the background -
    opening the review queue must not wait minutes for the classifier loop to catch up."""
    if not _qgen_busy.acquire(blocking=False):
        return  # a generation pass is already running
    def _run():
        try:
            from . import questions
            c2 = db.connect()
            try:
                for _ in range(3):  # drain a burst; stop as soon as a pass yields nothing
                    if questions.generate_pending(c2, cfg) == 0:
                        break
                questions.score_questions(c2, limit=200)
            finally:
                c2.close()
        except Exception:
            log.exception("on-demand question generation failed")
        finally:
            _qgen_busy.release()
    threading.Thread(target=_run, daemon=True, name="qgen-on-demand").start()


def _kick_kind_derivation(segment_id: int, note: str | None = None):
    """The person asserted 'separate thread' without a kind - the kind is the app's to derive
    (from what the thread became). Run the evaluator's derivation pass in the background."""
    def _run():
        try:
            from . import evaluator
            c2 = db.connect()
            try:
                evaluator.derive_kind(c2, cfg, segment_id, note)
            finally:
                c2.close()
        except Exception:
            log.exception("kind derivation kick failed for segment %d", segment_id)
    threading.Thread(target=_run, daemon=True, name="kind-derive").start()


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
    qsrc = {r[0]: r[1] for r in c.execute("SELECT segment_id, source FROM review_questions WHERE status='open'")}
    from .questions import PRIORITY_CUT
    qprio = {r[0]: r[1] for r in c.execute("SELECT segment_id, priority FROM review_questions WHERE status='open' AND priority IS NOT NULL")}
    urat = {r[0]: (r[1] or "") for r in c.execute("""SELECT e.segment_id, e.rationale FROM seg_evals e
        LEFT JOIN seg_reviews rv ON rv.segment_id = e.segment_id
        WHERE e.uncertain = 1 AND rv.segment_id IS NULL""")}
    def _tier(sid):
        if sid in qprio:  # scored: stakes x uncertainty decides the dot size
            return "high" if qprio[sid] >= PRIORITY_CUT else "routine"
        srcq = qsrc.get(sid, "")
        rat = urat.get(sid, "")
        # Propagation questions stay high-tier even after a re-judgment rewrote the rationale
        # (answering then clearing an answer does exactly that).
        if srcq in ("coherence", "audit", "propagation") or "daily audit" in rat or "your correction" in rat:
            return "high"
        return "routine"
    for s in segs:
        s["needs_review"] = (s["id"] in qsegs or s["id"] in usegs) and s.get("source") != "review"
        if s["needs_review"]:
            s["review_tier"] = _tier(s["id"])
    selected = {s["id"] for s in segs if s["needs_review"]}
    reviewed = c.execute("SELECT COUNT(*) FROM reviews r JOIN switches s ON s.id=r.switch_id WHERE s.ts>=? AND s.ts<?",
                         (t0, t1)).fetchone()[0]
    slim = [{k: s.get(k) for k in ("id", "app", "title", "url", "domain", "clip_start", "clip_end", "duration",
                                   "state", "label", "switch_id", "own_switch_id", "uncertain", "needs_review", "source", "probs", "neutral", "first_screenshot", "activity", "content", "hybrid", "toggl_id", "toggl_desc", "review_tier", "review_chat_mid", "focus_exception")} for s in segs]
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
    # How far the day's headline metrics could still move, if every open question went the
    # other way: the per-metric sum of open questions' stakes (an upper bound, since answers
    # overlap). Zero = the totals are settled as far as the questions can tell.
    import json as _json
    from .questions import STAKE_METRICS
    slack = {k: 0.0 for k in STAKE_METRICS}
    n_prio = 0
    for r in c.execute("""SELECT q.stakes_detail, q.priority FROM review_questions q JOIN segments g ON g.id=q.segment_id
                          WHERE q.status='open' AND q.scored_at IS NOT NULL AND g.start>=? AND g.start<?""", (t0, t1)):
        try:
            det = _json.loads(r["stakes_detail"] or "{}")
        except ValueError:
            det = {}
        for k in slack:
            slack[k] += float(det.get(k) or 0)
        n_prio += int((r["priority"] or 0) >= PRIORITY_CUT)
    return {"day": d.isoformat(), "t0": t0, "t1": t1, "metrics": stats.daily_metrics(c, d), "segments": slim,
            "stamp": stats.day_stamp(c, t0, t1),
            "spend_avg_30d": db.spend_daily_avg(c, 30), "budget": cfg.daily_budget_usd,
            "to_review": len(selected), "reviewed": reviewed,  # open questions + unresolved flags today
            "metric_slack": slack, "priority_open": n_prio,
            "short_breaks": stats.short_breaks(c, t0, t1),
            "toggl": [{"start": max(e["start"], t0), "end": min(e["stop"] or time.time(), t1), "description": e["description"],
                       "project": e["project"], "tags": e["tags"]} for e in toggl.entries_between(c, t0, t1)],
            "count_events": cfg.count_events, "min_focus_span_min": cfg.min_focus_span_min,
            "focus_spans": [{"start": r["start"], "end": r["end"], "duration": r["duration"], "ended_by": r["ended_by"],
                             "review_chat_mid": r.get("review_chat_mid"),
                             "focus_min": r["focus_min"], "detours": len(r["detours"]), "detour_min": r["detour_min"],
                             "subtype": r.get("subtype", "focus"), "fully_absorbed": r.get("fully_absorbed", False),
                             "hybrid": bool(r.get("hybrid")),
                             "mixed": r.get("mixed_read_create", False), "summary": _summary(r),
                             "coherence_issue": _coherence(r), "title": _title(r),
                             "events": [{"start": e["start"], "end": e["end"], "kind": e["state"]} for e in r["detours"]],
                             "apps": _apps_by_time(r["segments"]),
                             "toggl": _toggl_with_coverage(c, r["start"], r["end"])}
                            for r in se["focus_spans"]],
            "distractions": [{"start": r["start"], "end": r["end"], "duration": r["duration"], "is_event": r["is_event"],
                              "where": sorted({s["domain"] or s["app"] for s in r["segments"]})} for r in se["distractions"]],
            "disqualified": [{"start": r["start"], "end": r["end"], "duration": r["duration"],
                              "review_chat_mid": r.get("review_chat_mid"),
                              "reason": r.get("disq_reason", "does not meet the focused-work standard"),
                              "summary": _summary(r)}
                             for r in se.get("disqualified", [])]}


_toggl = None
@app.post("/api/toggl/resync")
def api_toggl_resync(days: float = 7):
    """Manual: pull the last `days` of Toggl entries again, so entries you corrected later are
    reflected. Runs synchronously (one API call) and returns what changed."""
    global _toggl
    if not toggl.token():
        raise HTTPException(400, "no Toggl API token set (menu bar → Set Toggl API Token)")
    if _toggl is None:
        _toggl = toggl.TogglSync(conn())
    r = _toggl.resync(max(1.0, min(days, 92)) * 24)
    if r is None:
        if time.time() < _toggl.quota_until:
            raise HTTPException(429, f"Toggl's hourly API quota is used up - try again in {int(_toggl.quota_until - time.time()) // 60 + 1} min")
        raise HTTPException(502, "Toggl sync failed - see app.log")
    return r


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
            "url", "activity", "content", "hybrid", "switch_id", "source", "first_screenshot", "toggl_id", "toggl_desc")
    in_span_ids = {id(s) for s in sp["segments"]}
    segs = []
    for s in (day_segs or sp["segments"]):
        if s["clip_end"] <= w0 or s["clip_start"] >= w1 or s["duration"] <= 0:
            continue
        d = {k: s.get(k) for k in keys}
        d["in_span"] = id(s) in in_span_ids or (day_segs is None)
        segs.append(d)
    return {"day": day, "start": sp["start"], "end": sp["end"], "duration": sp["duration"],
            "focus_min": sp["focus_min"], "subtype": sp.get("subtype"), "hybrid": bool(sp.get("hybrid")),
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
    try:
        edits.write_span_review(conn(), r.start, r.end, r.verdict, r.subtype, r.note)
    except edits.BadEdit as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.delete("/api/span/review")
def api_span_unreview(start: float, end: float):
    """Withdraw the verdict on a span (the span goes back to the system's own call)."""
    return {"ok": True, "deleted": edits.delete_span_review(conn(), start, end)}


@app.get("/api/efficiency")
def api_efficiency():
    from . import seg_models
    c = conn()
    rep = seg_models.efficiency_report(c)
    rep["antecedent_ranks"] = seg_models.antecedent_rank_report(c)
    return rep


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
    # An answered question stays on the card (greyed, changeable) - a misclick must be easy to fix.
    qrow = c.execute("SELECT * FROM review_questions WHERE segment_id=? AND status IN ('open','answered','unsure')"
                     " ORDER BY (status='open') DESC, created DESC LIMIT 1", (to_id,)).fetchone()
    full["question"] = None
    if qrow:
        from . import questions as _q
        full["question"] = dict(qrow)
        full["question"].pop("undo", None)
        full["question"]["options"] = _json.loads(full["question"]["options"])
        full["question"]["stakes_text"] = _q.stakes_text(qrow["stakes_detail"]) if qrow["scored_at"] else None
        full["question"]["tier"] = (("priority" if (qrow["priority"] or 0) >= _q.PRIORITY_CUT else "routine")
                                    if qrow["scored_at"] else None)
    else:
        # Flagged but no question composed yet: start composing now so a reload has it.
        flagged = c.execute("""SELECT 1 FROM seg_evals e LEFT JOIN seg_reviews r ON r.segment_id=e.segment_id
                               LEFT JOIN review_questions q ON q.segment_id=e.segment_id
                               WHERE e.segment_id=? AND e.uncertain=1 AND r.segment_id IS NULL
                               AND q.id IS NULL""", (to_id,)).fetchone()
        if flagged:
            full["question_pending"] = True
            _kick_question_generation()
    full["to_shots"] = [dict(r) for r in c.execute(
        "SELECT ts, path, display FROM segment_shots WHERE segment_id=? ORDER BY ts", (to_id,))]
    full["to_eval"] = (lambda r: dict(r) if r else None)(
        c.execute("SELECT * FROM seg_evals WHERE segment_id=?", (to_id,)).fetchone())
    full["to_seg_review"] = (lambda r: dict(r) if r else None)(
        c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (to_id,)).fetchone())
    te = next((e for e in toggl.entries_between(c, ts - 1, ts + 1)
               if e["start"] <= ts and (e["stop"] is None or e["stop"] > ts)), None)
    full["toggl"] = {"description": te["description"], "project": te["project"], "tags": te["tags"]} if te else None
    full["toggl_entries"] = [{"start": max(e["start"], ts - 900), "end": min(e["stop"] or time.time(), ts + 900),
                              "description": e["description"], "project": e["project"]}
                             for e in toggl.entries_between(c, ts - 900, ts + 900)]
    # The span this switch lands in (qualifying or disqualified), for audit context.
    from . import summaries as _summ
    full["span"] = None
    full["timeline"] = []
    try:
        ctx_segs = stats.labelled_segments(c, ts - 6 * 3600, ts + 6 * 3600)
        keys = ("id", "clip_start", "clip_end", "duration", "state", "label", "app", "domain", "title",
                "url", "activity", "content", "hybrid", "switch_id", "source", "first_screenshot", "toggl_id", "toggl_desc")
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


_day_chat = None
def get_day_chat():
    global _day_chat
    if _day_chat is None:
        from .chat import DayChat
        _day_chat = DayChat(cfg)
    return _day_chat


@app.get("/api/day/stamp")
def api_day_stamp(day: str | None = None):
    """Cheap change fingerprint of a day; the Day view polls it and redraws only when it moves."""
    d = dt.date.fromisoformat(day) if day else logical_date()
    t0, t1 = stats.day_bounds(d)
    return {"day": d.isoformat(), "stamp": stats.day_stamp(conn(), t0, t1)}


@app.get("/api/chat/message/{mid}")
def api_chat_message(mid: int):
    """Where a chat reply lives, so a flagged judgment can open the conversation it came from."""
    c = conn()
    m = c.execute("SELECT id, kind, ref, role, text, created, question_id FROM chat_messages WHERE id=?", (mid,)).fetchone()
    if not m:
        raise HTTPException(404)
    out = dict(m)
    if m["kind"] == "evaluator":  # the segment chat lives on the capture card, keyed by switch
        sw = c.execute("SELECT id FROM switches WHERE to_segment=?", (int(m["ref"]),)).fetchone()
        out["switch_id"] = sw["id"] if sw else None
    return out


class DayChatReq(BaseModel):
    day: str            # logical day, YYYY-MM-DD
    messages: list[dict]


@app.post("/api/day/chat")
def api_day_chat(req: DayChatReq):
    """Talk with Claude about a whole day. History is kept per day (chat kind 'day', ref = the date)."""
    import datetime as _dt
    try:
        day = _dt.date.fromisoformat(req.day)
    except ValueError:
        raise HTTPException(400, "day must be YYYY-MM-DD")
    try:
        return get_day_chat().reply(conn(), day, req.messages)
    except Exception as e:
        raise HTTPException(500, str(e))


QCTX_S = 900.0  # timeline context on each side of a question's segment (matches ~what the generator saw)


def _shots_for(c, timeline: list[dict]) -> list[dict]:
    """Every capture of the given segments (for the scrubber), oldest first; falls back to each
    segment's first screenshot for segments recorded before per-shot history existed."""
    ids = [s["id"] for s in timeline]
    if not ids:
        return []
    qs = ",".join("?" * len(ids))
    rows = [dict(r) for r in c.execute(
        f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) ORDER BY ts", ids)]
    return rows or [{"segment_id": s["id"], "ts": s["clip_start"], "path": s["first_screenshot"], "display": 0}
                    for s in timeline if s.get("first_screenshot")]


@app.get("/api/questions/queue")
def api_questions_queue(limit: int = 30):
    """Open review questions in review order: the priority tier first (an answer could move
    the day's headline metrics, weighted by the model's uncertainty), then the rest; each tier
    in chronological order."""
    from . import questions as _q
    from .config import logical_date
    c = conn()
    import json as _json
    try:
        _q.score_questions(c, limit=25)  # catch up on a few stale ones; the loop does the bulk
    except Exception:
        log.exception("question scoring failed")
    out = []
    keys = ("id", "clip_start", "clip_end", "duration", "state", "label", "app", "domain", "title",
            "content", "hybrid", "switch_id", "source", "first_screenshot", "last_screenshot", "url", "toggl_id", "toggl_desc")
    rows = [dict(r) for r in c.execute("""SELECT q.*, g.start AS seg_start, COALESCE(g.end, g.start) AS seg_end,
                                 g.app, g.title, g.domain,
                                 (SELECT id FROM switches WHERE to_segment = q.segment_id LIMIT 1) AS switch_id
                          FROM review_questions q JOIN segments g ON g.id = q.segment_id
                          WHERE q.status = 'open'""")]
    for d in rows:
        d["tier"] = ("priority" if (d.get("priority") or 0) >= _q.PRIORITY_CUT else
                     "routine" if d.get("priority") is not None else "unscored")
        d["day"] = logical_date(d["seg_start"]).isoformat()
    rows.sort(key=lambda d: ({"priority": 0, "routine": 1, "unscored": 2}[d["tier"]], d["seg_start"]))
    counts = {"priority": sum(1 for d in rows if d["tier"] == "priority"), "total": len(rows)}
    # Every priority question is offered - as many as it takes to pin down the day's metrics -
    # and the limit only trims the routine tail.
    shown = [d for d in rows if d["tier"] == "priority"] + [d for d in rows if d["tier"] != "priority"][:limit]
    for d in shown:
        d.pop("undo", None)
        d["options"] = _json.loads(d["options"])
        d["stakes_text"] = _q.stakes_text(d.get("stakes_detail"))
        d["queue_counts"] = counts
        segs = stats.labelled_segments(c, d["seg_start"] - QCTX_S, d["seg_end"] + QCTX_S)
        d["timeline"] = [{k: s.get(k) for k in keys} for s in segs if s["duration"] > 0]
        d["timeline_shots"] = _shots_for(c, d["timeline"])
        d["toggl_entries"] = [{"start": max(e["start"], d["seg_start"] - QCTX_S),
                               "end": min(e["stop"] or time.time(), d["seg_end"] + QCTX_S),
                               "description": e["description"], "project": e["project"]}
                              for e in toggl.entries_between(c, d["seg_start"] - QCTX_S, d["seg_end"] + QCTX_S)]
        out.append(d)
    # Flagged segments whose question hasn't been composed yet: show them as pending cards
    # (the person opened the queue - the question must exist NOW) and start generating.
    try:
        from . import questions as _q
        pending = _q._triggers(c)[:8]
        for t in pending:
            g = c.execute("SELECT * FROM segments WHERE id=?", (t["segment_id"],)).fetchone()
            if not g:
                continue
            seg_start, seg_end = g["start"], g["end"] or g["start"]
            segs = stats.labelled_segments(c, seg_start - QCTX_S, seg_end + QCTX_S)
            out.append({
                "id": None, "pending": True, "segment_id": t["segment_id"], "source": t["source"],
                "context": t["reason"], "question": None, "options": [], "tier": None,
                "seg_start": seg_start, "seg_end": seg_end,
                "app": g["app"], "title": g["title"], "domain": g["domain"],
                "switch_id": (lambda r: r["id"] if r else None)(c.execute(
                    "SELECT id FROM switches WHERE to_segment=? LIMIT 1", (t["segment_id"],)).fetchone()),
                "timeline": [{k: s.get(k) for k in keys} for s in segs if s["duration"] > 0],
                "timeline_shots": _shots_for(c, [s for s in segs if s["duration"] > 0]),
                "toggl_entries": [{"start": max(e["start"], seg_start - QCTX_S),
                                   "end": min(e["stop"] or time.time(), seg_end + QCTX_S),
                                   "description": e["description"], "project": e["project"]}
                                  for e in toggl.entries_between(c, seg_start - QCTX_S, seg_end + QCTX_S)]})
        if pending:
            _kick_question_generation()
    except Exception:
        log.exception("pending-question placeholders failed")
    return out


class QAnswer(BaseModel):
    option_index: int
    note: str | None = None


def _restore_rows(c, table: str, key: str, key_val, row: dict | None):
    if row is None:
        c.execute(f"DELETE FROM {table} WHERE {key}=?", (key_val,))
    else:
        cols = list(row)
        c.execute(f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                  [row[k] for k in cols])


def _undo_question_answer(c, undo_json: str | None) -> dict:
    """Put back whatever a previous answer to a review question overwrote (the segment review,
    and the legacy switch review), so changing or clearing an answer leaves no residue."""
    import json as _json
    u = _json.loads(undo_json) if undo_json else {}
    if u.get("seg_id") is not None:
        _restore_rows(c, "seg_reviews", "segment_id", u["seg_id"], u.get("seg_review"))
    if u.get("switch_id") is not None:
        _restore_rows(c, "reviews", "switch_id", u["switch_id"], u.get("review"))
        c.execute("UPDATE switches SET status=? WHERE id=?", (u.get("switch_status"), u["switch_id"]))
    return u


@app.post("/api/questions/{qid}/answer")
def api_question_answer(qid: int, a: QAnswer):
    return _answer_question(conn(), qid, a.option_index, a.note)


def _answer_question(c, qid: int, option_index: int, note: str | None,
                     chat_message_id: int | None = None) -> dict:
    """Record the chosen option of a review question as the person's judgment (segment review +
    legacy switch review), undoing any earlier answer first. Shared by the option buttons and
    by applying a judgment the evaluator proposed in chat."""
    import json as _json
    q = c.execute("SELECT * FROM review_questions WHERE id=?", (qid,)).fetchone()
    if not q:
        raise HTTPException(404)
    opts = _json.loads(q["options"])
    if not (0 <= option_index < len(opts)):
        raise HTTPException(400, "bad option")
    o = opts[option_index]
    a = QAnswer(option_index=option_index, note=note)
    with db.tx(c):
        # Re-answering (fixing a misclick): first put back what the earlier answer overwrote.
        old_u = (_undo_question_answer(c, q["undo"]) if q["status"] == "answered"
                 else (_json.loads(q["undo"]) if q["status"] == "unsure" and q["undo"] else {}))
        ev0 = c.execute("SELECT uncertain FROM seg_evals WHERE segment_id=?", (q["segment_id"],)).fetchone()
        undo = {"uncertain": old_u.get("uncertain", ev0["uncertain"] if ev0 else None)}
        if o.get("switch_label") or o.get("content") or o.get("hybrid") is not None:
            prev = c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (o["segment_id"],)).fetchone()
            undo.update(seg_id=o["segment_id"], seg_review=dict(prev) if prev else None)
            osw = o.get("switch_label")
            ante = (_valid_antecedent(c, o["segment_id"], o.get("antecedent_id"))
                    if osw in ("continuation", "return")
                    else (None if osw else (prev["antecedent_id"] if prev else None)))
            c.execute("INSERT OR REPLACE INTO seg_reviews(segment_id, content, switch_label, interruption_kind, antecedent_id, note, created, hybrid, chat_message_id, focus_exception)"
                      " VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (o["segment_id"],
                       o.get("content") or (prev["content"] if prev else None),
                       osw or (prev["switch_label"] if prev else None),
                       o.get("interruption_kind") if osw == "interruption" else (None if osw else (prev["interruption_kind"] if prev else None)),
                       ante,
                       (a.note.strip() if a.note and a.note.strip() else f"(answered review question: {o['label']})"), time.time(),
                       (int(o["hybrid"]) if o.get("hybrid") is not None else (prev["hybrid"] if prev else None)),
                       chat_message_id if chat_message_id is not None else o.get("proposed_from"),
                       prev["focus_exception"] if prev else None))  # a question answer never touches the ruling
            from . import propagation
            propagation.note_event(c, o["segment_id"])
            old = _NEW2OLD.get((o.get("switch_label"),
                                o.get("interruption_kind") if o.get("switch_label") == "interruption" else None))
            sw = c.execute("SELECT id FROM switches WHERE to_segment=?", (o["segment_id"],)).fetchone()
            if old and sw:
                prev_rev = c.execute("SELECT * FROM reviews WHERE switch_id=?", (sw["id"],)).fetchone()
                undo.update(switch_id=sw["id"], review=dict(prev_rev) if prev_rev else None,
                            switch_status=c.execute("SELECT status FROM switches WHERE id=?", (sw["id"],)).fetchone()[0])
                c.execute("INSERT OR REPLACE INTO reviews(switch_id, label, note, created) VALUES (?,?,?,?)",
                          (sw["id"], old, (a.note.strip() if a.note and a.note.strip() else o["label"]), time.time()))
                c.execute("UPDATE switches SET status='reviewed' WHERE id=?", (sw["id"],))
        c.execute("UPDATE review_questions SET status='answered', answer=?, answer_index=?, note=?, answered_at=?, undo=?"
                  " WHERE id=?",
                  (o["label"], a.option_index, (a.note.strip() or None) if a.note else None, time.time(),
                   _json.dumps(undo), qid))
        # The answer resolves the doubt this question was generated for, even when the chosen
        # option edits a different segment - otherwise the needs-review dot never goes away.
        c.execute("UPDATE seg_evals SET uncertain=0 WHERE segment_id=?", (q["segment_id"],))
    if o.get("switch_label") == "interruption" and not o.get("interruption_kind"):
        _kick_kind_derivation(o["segment_id"], a.note)
    return {"ok": True}


@app.get("/api/questions/{qid}/state")
def api_question_state(qid: int):
    """The current status of one question - the queue panel re-checks this when it shows a
    question, so an answer given elsewhere (the capture page) is reflected."""
    q = conn().execute("SELECT status, answer, answer_index, note FROM review_questions WHERE id=?", (qid,)).fetchone()
    if not q:
        raise HTTPException(404)
    return dict(q)


@app.post("/api/questions/{qid}/reopen")
def api_question_reopen(qid: int):
    """Clear an answer (or an 'I can't tell'): undo its edits and put the question back to open."""
    import json as _json
    c = conn()
    q = c.execute("SELECT * FROM review_questions WHERE id=?", (qid,)).fetchone()
    if not q:
        raise HTTPException(404)
    if q["status"] not in ("answered", "unsure"):
        return {"ok": True}
    with db.tx(c):
        u = _undo_question_answer(c, q["undo"]) if q["status"] == "answered" else (_json.loads(q["undo"]) if q["undo"] else {})
        c.execute("UPDATE review_questions SET status='open', answer=NULL, answer_index=NULL, note=NULL,"
                  " answered_at=NULL, undo=NULL WHERE id=?", (qid,))
        if u.get("uncertain") is not None:
            c.execute("UPDATE seg_evals SET uncertain=? WHERE segment_id=?", (u["uncertain"], q["segment_id"]))
    return {"ok": True}


class UnsureReq(BaseModel):
    note: str | None = None


@app.post("/api/questions/{qid}/unsure")
def api_question_unsure(qid: int, body: UnsureReq):
    """'I can't tell': the question is resolved without a ruling - the model's judgment stands,
    the needs-review flag clears, and the record of an unanswerable doubt is kept so the question
    generator learns which doubts are not worth raising. Reopen puts it back."""
    import json as _json
    c = conn()
    q = c.execute("SELECT * FROM review_questions WHERE id=?", (qid,)).fetchone()
    if not q:
        raise HTTPException(404)
    with db.tx(c):
        if q["status"] == "answered":
            _undo_question_answer(c, q["undo"])
            undo = _json.loads(q["undo"]) if q["undo"] else {}
            undo = {"uncertain": undo.get("uncertain")}
        elif q["status"] == "unsure":
            undo = _json.loads(q["undo"]) if q["undo"] else {}
        else:
            ev0 = c.execute("SELECT uncertain FROM seg_evals WHERE segment_id=?", (q["segment_id"],)).fetchone()
            undo = {"uncertain": ev0["uncertain"] if ev0 else None}
        c.execute("UPDATE review_questions SET status='unsure', answer=?, answer_index=NULL, note=?, answered_at=?, undo=?"
                  " WHERE id=?", ("(can't tell)", (body.note.strip() or None) if body.note else None, time.time(),
                                  _json.dumps(undo), qid))
        c.execute("UPDATE seg_evals SET uncertain=0 WHERE segment_id=?", (q["segment_id"],))
    return {"ok": True}


@app.post("/api/questions/{qid}/dismiss")
def api_question_dismiss(qid: int):
    c = conn()
    with db.tx(c):
        c.execute("UPDATE review_questions SET status='dismissed', answered_at=? WHERE id=?",
                  (time.time(), qid))
        c.execute("""UPDATE seg_evals SET uncertain=0 WHERE segment_id =
                     (SELECT segment_id FROM review_questions WHERE id=?)""", (qid,))
    return {"ok": True}


@app.get("/api/chat/{kind}/{ref}")
def api_chat_history(kind: str, ref: str):
    """Stored conversation for one item (kind: evaluator | switch | span | day), oldest first."""
    from . import chat as _chat
    if kind not in ("evaluator", "switch", "span", "day"):
        raise HTTPException(400, "bad kind")
    return _chat.history(conn(), kind, ref)


class EvalChatReq(BaseModel):
    messages: list[dict]
    question_id: int | None = None  # set when this is a follow-up on a review question


@app.post("/api/segment/{segment_id}/evaluator-chat")
def api_evaluator_chat(segment_id: int, req: EvalChatReq):
    from . import evaluator
    try:
        return evaluator.chat_reply(conn(), cfg, segment_id, req.messages, req.question_id)
    except Exception as e:
        raise HTTPException(500, str(e))


def _chat_message(c, mid: int):
    import json as _json
    m = c.execute("SELECT * FROM chat_messages WHERE id=?", (mid,)).fetchone()
    if not m or m["role"] != "assistant" or not m["proposal"]:
        raise HTTPException(404, "no proposal on that message")
    return m, _json.loads(m["proposal"])


def _save_proposal(c, mid: int, proposal: dict):
    import json as _json
    c.execute("UPDATE chat_messages SET proposal=? WHERE id=?", (_json.dumps(proposal), mid))


class ApplyJudgmentReq(BaseModel):
    question_id: int | None = None  # the question on screen; defaults to the one the chat was about


@app.post("/api/chat/message/{mid}/apply-judgment")
def api_chat_apply_judgment(mid: int, body: ApplyJudgmentReq):
    """Apply a judgment the evaluator proposed in chat. With a review question: the proposal
    becomes a new option (in the person's words) and is answered - so it is greyed, undoable and
    re-answerable like any other. Without one: it is a direct edit of the segment's judgment."""
    import json as _json
    c = conn()
    m, prop = _chat_message(c, mid)
    j = prop.get("judgment")
    if not j or m["kind"] != "evaluator":
        raise HTTPException(400, "no judgment proposed")
    segment_id = int(m["ref"])
    qid = body.question_id if body.question_id is not None else m["question_id"]
    q = c.execute("SELECT * FROM review_questions WHERE id=? AND segment_id=?", (qid, segment_id)).fetchone() if qid else None
    applied = prop.setdefault("applied", {})
    if q is not None:
        opts = _json.loads(q["options"])
        idx = next((i for i, o in enumerate(opts) if o.get("proposed_from") == mid), None)
        if idx is None:
            opts.append({"label": j["label"], "segment_id": segment_id,
                         "switch_label": j.get("switch_label"), "antecedent_id": j.get("antecedent_id"),
                         "interruption_kind": None, "content": j.get("content"), "hybrid": j.get("hybrid"),
                         "proposed_from": mid})
            idx = len(opts) - 1
            with db.tx(c):
                c.execute("UPDATE review_questions SET options=? WHERE id=?", (_json.dumps(opts), q["id"]))
        _answer_question(c, q["id"], idx, j.get("note") or None, chat_message_id=mid)
        applied["judgment"] = {"at": time.time(), "question_id": q["id"], "option_index": idx}
        with db.tx(c):
            _save_proposal(c, mid, prop)
        return {"ok": True, "question_id": q["id"], "option_index": idx, "options": opts, "proposal": prop}
    try:
        derive = edits.write_seg_review(c, segment_id, content=j.get("content"), switch_label=j.get("switch_label"),
                                        antecedent_id=j.get("antecedent_id"), hybrid=j.get("hybrid"),
                                        note=j.get("note") or j["label"], chat_message_id=mid)
    except edits.BadEdit as e:
        raise HTTPException(400, str(e))
    if derive:
        _kick_kind_derivation(segment_id, j.get("note"))
    applied["judgment"] = {"at": time.time(), "segment_id": segment_id}
    with db.tx(c):
        _save_proposal(c, mid, prop)
    return {"ok": True, "segment_id": segment_id, "proposal": prop}


@app.post("/api/chat/message/{mid}/apply-rule")
def api_chat_apply_rule(mid: int):
    """Append a rule the evaluator proposed to the rules document, under a 'Lessons from
    reviews' heading, dated. Every model prompt that embeds the rules picks it up on its next
    build (the evaluator re-reads the file when its mtime changes)."""
    from .classifiers.claude_vision import RULES_PATH
    c = conn()
    m, prop = _chat_message(c, mid)
    rule = (prop.get("rule") or "").strip()
    if not rule:
        raise HTTPException(400, "no rule proposed")
    applied = prop.setdefault("applied", {})
    if not applied.get("rule"):
        heading = "### Lessons from reviews"
        text = RULES_PATH.read_text() if RULES_PATH.exists() else ""
        if heading not in text:
            text = text.rstrip("\n") + "\n\n" + heading + "\n\n"
        line = f"* {rule} (added {dt.date.today().isoformat()} from a review of segment {m['ref']})\n"
        text = text.rstrip("\n") + "\n" + line
        RULES_PATH.write_text(text)
        applied["rule"] = {"at": time.time()}
        with db.tx(c):
            _save_proposal(c, mid, prop)
    return {"ok": True, "proposal": prop}


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
    antecedent_id: int | None = None  # thread pointer, for continuation/return assertions
    note: str | None = None
    hybrid: bool | None = None  # the activity was done inside a live discussion (None = no ruling)
    focus_exception: bool | None = None  # the person's explicit ruling: counts as independent focus anyway


_valid_antecedent = edits.valid_antecedent


_NEW2OLD = edits.NEW2OLD


@app.get("/api/segment/{segment_id}/shots")
def api_segment_shots(segment_id: int):
    """All captures of one segment - the timeline hover scrubs through these by cursor time."""
    c = conn()
    return [dict(r) for r in c.execute(
        "SELECT ts, path, display FROM segment_shots WHERE segment_id=? ORDER BY ts", (segment_id,))]


@app.post("/api/segment/{segment_id}/eval-review")
def api_seg_eval_review(segment_id: int, r: SegEvalReview):
    """The person's direct edit of a segment's judgment (no chat involved: chat_message_id stays NULL)."""
    c = conn()
    try:
        derive = edits.write_seg_review(c, segment_id, content=r.content, switch_label=r.switch_label,
                                        interruption_kind=r.interruption_kind, antecedent_id=r.antecedent_id,
                                        note=r.note, hybrid=r.hybrid, focus_exception=r.focus_exception)
    except edits.BadEdit as e:
        raise HTTPException(400 if "no segment" not in str(e) else 404, str(e))
    if derive:
        _kick_kind_derivation(segment_id, r.note)
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


class RestartBody(BaseModel):
    only_if_stale: bool = False


def _source_files() -> list[Path]:
    here = Path(__file__).parent
    return sorted(here.glob("*.py")) + sorted(here.glob("*.html")) + [CONFIG_PATH]


@app.post("/api/restart")
def api_restart(body: RestartBody):
    """Dev convenience: relaunch the app so it picks up edited source (HTML is read once at import).

    With only_if_stale, a no-op unless some source file is newer than this process. Refuses to
    restart code that will not even compile - the launcher gives up after a burst of crashes,
    and a half-finished edit must not take the monitor down."""
    files = [f for f in _source_files() if f.exists()]
    newest = max((f.stat().st_mtime for f in files), default=0)
    if body.only_if_stale and newest <= STARTED:
        return {"restarted": False, "reason": "up to date"}
    for f in files:
        if f.suffix == ".py":
            try:
                compile(f.read_text(), str(f), "exec")
            except SyntaxError as e:
                return JSONResponse(status_code=409, content={
                    "restarted": False, "reason": f"{Path(e.filename).name}:{e.lineno}: {e.msg}"})
    if restart_hook is None:
        return {"restarted": False, "reason": "not running as the menu bar app"}
    info = restart_hook()
    log.info("restart requested; dashboard reopen=%s", info.get("reopen"))
    return {"restarted": True, **info}


@app.get("/shot/{name}")
def shot(name: str):
    p = screenshot_file(name)
    if p is None:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


def serve():
    import uvicorn
    uvicorn.run(app, host=cfg.web_host, port=cfg.web_port, log_level="warning")

"""Local dashboard: timeline, daily metrics, review queue, model accuracy."""
from __future__ import annotations

import datetime as dt
import logging
import math
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import phone, LABEL_HELP, LABEL_NAMES, LABELS, db, edits, ensemble, stats, toggl
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
        _kind_timers.pop(segment_id, None)
        with _kind_lock:   # one at a time: they share the evaluator object (its last_end / last_answer)
            try:
                from . import evaluator
                c2 = db.connect()
                try:
                    evaluator.derive_kind(c2, cfg, segment_id, note)
                finally:
                    c2.close()
            except Exception:
                log.exception("kind derivation kick failed for segment %d", segment_id)
    # A ruling saved in two or three clicks used to start a re-judgment per click, at once, on one
    # shared evaluator object: three runs of the same stretch within 24 s, the later ones cut
    # short by the first one's state (2026-11-21). The run starts 5 s after the LAST save.
    old = _kind_timers.pop(segment_id, None)
    if old is not None:
        old.cancel()
    t = threading.Timer(5.0, _run)
    t.daemon = True
    _kind_timers[segment_id] = t
    t.start()


_score_busy = threading.Lock()
_kind_timers: dict = {}
_kind_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index():
    _warm_phone_cache()
    return HTML


_phone_warm = threading.Lock()


def _warm_phone_cache() -> None:
    """Keep phone.autolock_s worked out for the last 8 days, in the background.

    Every timeline payload asks phone.window for the Auto-Lock interval, which phone.py works
    out per six hours of the day (0.12 s each) and keeps for ten minutes. /api/span/queue covers
    seven days, so opening the Review tab with a cold cache took 2.2 to 2.9 s against 0.33 s
    warm (2026-11-21). One thread, started when the server starts or the page is first loaded,
    fills the cache every 8 minutes; a six-hour piece whose two days either side have all
    arrived from the phone cannot change, so its answer is kept for six hours."""
    if not _phone_warm.acquire(blocking=False):
        return   # already running
    def _run():
        while True:
            try:
                pc = phone._connect(cfg)
                if pc is not None:
                    try:
                        now = time.time()
                        through = phone.through(cfg) or 0.0
                        for part in range(int((now - 8 * 86400) // 21600), int(now // 21600) + 1):
                            mid = part * 21600 + 10800.0
                            val = phone.autolock_s(pc, mid)
                            if mid + phone.AUTOLOCK_HALF_WINDOW_S < through - 3600:
                                phone._AUTOLOCK_CACHE[part] = (time.time() + 6 * 3600, val)
                    finally:
                        pc.close()
            except Exception:
                log.exception("phone cache warm-up failed")
            time.sleep(480)
    threading.Thread(target=_run, daemon=True, name="phone-autolock-warm").start()


app.router.on_startup.append(_warm_phone_cache)


def _day_arg(day: str | None):
    """(date, t0, t1) for a ?day= value, or a 400. 2026-11-21: `day=garbage` on /api/day/stamp and
    `day=0001-01-01` / `9999-12-31` on /api/day were 500s (an unparsed date; a date the time
    zone arithmetic cannot represent)."""
    try:
        d = dt.date.fromisoformat(day) if day else logical_date()
        if not 2000 <= d.year <= 2100:
            raise ValueError("out of range")
        t0, t1 = stats.day_bounds(d)
    except (ValueError, OverflowError, OSError):
        raise HTTPException(400, "day must be a date YYYY-MM-DD between 2000 and 2100")
    return d, t0, t1


def _ts_arg(*ts: float) -> None:
    """400 for a time that is not a real moment (nan, inf, 1e18 were 500s on /api/span/at, 2026-11-21)."""
    for t in ts:
        if not math.isfinite(t) or not 0 < t < 4_102_444_800:   # up to the year 2100
            raise HTTPException(400, "not a valid time")


def _capped(n: int, lo: int, hi: int) -> int:
    return max(lo, min(int(n), hi))


@app.get("/api/day")
def api_day(day: str | None = None):
    d, t0, t1 = _day_arg(day)
    c = conn()
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
                                   "state", "label", "focused", "switch", "self_distraction", "next_task", "sub_of", "sub_absorbed", "diversion", "switch_id", "own_switch_id", "uncertain", "needs_review", "source", "probs", "neutral", "first_screenshot", "activity", "content", "hybrid", "toggl_id", "toggl_desc", "review_tier", "review_chat_mid", "focus_exception",
                                   # 2026-11-21: `phone` was left out, so the Day tooltip never said "low confidence" on a phone block
                                   "phone")} for s in segs]
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
            # open questions + unresolved flags today, plus each span waiting on the
            # "was part of this span unfocused?" question (it is a review question like the others)
            "to_review": len(selected) + sum(1 for r in se["focus_spans"] if r.get("unfocused_ask")),
            "reviewed": reviewed,
            "metric_slack": slack, "priority_open": n_prio,
            "short_breaks": stats.short_breaks(c, t0, t1),
            "phone": phone.window(cfg, t0, t1),
            "toggl": [{"start": max(e["start"], t0), "end": min(e["stop"] or time.time(), t1), "description": e["description"],
                       "project": e["project"], "tags": e["tags"]} for e in toggl.entries_between(c, t0, t1)],
            "count_events": cfg.count_events, "min_focus_span_min": cfg.min_focus_span_min,
            "focus_spans": [{"start": r["start"], "end": r["end"], "duration": r["duration"], "ended_by": r["ended_by"],
                             "review_chat_mid": r.get("review_chat_mid"),
                             "focus_min": r["focus_min"], "detours": len(r["detours"]), "detour_min": r["detour_min"],
                             "subtype": r.get("subtype", "focus"), "fully_absorbed": r.get("fully_absorbed", False),
                             "hybrid": bool(r.get("hybrid")),
                             "unfocused_min": r.get("unfocused_min", 0.0), "unfocused_max_min": r.get("unfocused_max_min", 0.0),
                             "unfocused_ask": bool(r.get("unfocused_ask")), "unfocused_seg_ids": r.get("unfocused_seg_ids", []),
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
        raise HTTPException(502, "Toggl sync failed - see ~/Library/Logs/focus-and-distraction-monitor.log")
    return r


def _toggl_row(r: dict, now: float) -> dict:
    """One toggl_entries row as the executive assistant expects it (same shape as Toggl's own API gave it)."""
    running = r["stop"] is None
    secs = (now if running else r["stop"]) - r["start"]
    iso = lambda t: dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds") if t else None
    return {"id": r["id"], "description": r["description"], "project": r["project"] or None,
            "tags": [t for t in r["tags"].split(",") if t], "start": iso(r["start"]), "stop": iso(r["stop"]),
            "minutes": round(max(0.0, secs) / 60), "running": running, "synced": iso(r["synced"])}


@app.get("/api/toggl/current")
def api_toggl_current():
    """The entry running at the last sync (NULL stop), or null. The local table is refreshed every 5 minutes,
    so this can lag Toggl by that much; `synced` says when. For the executive assistant (2026-11-11), so that
    it stops polling Toggl itself and sharing this app's hourly API quota."""
    c = conn()
    r = c.execute("SELECT * FROM toggl_entries WHERE stop IS NULL ORDER BY start DESC LIMIT 1").fetchone()
    return _toggl_row(dict(r), time.time()) if r else None


@app.get("/api/toggl/entries")
def api_toggl_entries(days: float = 7):
    """Entries that overlap the last `days` days (max 92), oldest first, from the local table."""
    now = time.time()
    t0 = now - max(1.0, min(days, 92)) * 86400
    return [_toggl_row(r, now) for r in toggl.entries_between(conn(), t0, now + 86400)]


@app.get("/api/metrics")
def api_metrics(days: int = 30):
    # capped: ?days=100000 ran for minutes (2026-11-21); the page asks for 90 at most
    return {"count_events": cfg.count_events, "days": _metrics_days(conn(), _capped(days, 1, 366))}


_METRICS: dict = {}   # date -> (fingerprint, that day's metrics)


def _metrics_days(c, days: int) -> list[dict]:
    """stats.metrics_range with each past day's numbers kept until the day changes. A day costs
    about 40 ms to work out and Trends asks for up to 90 on every visit (0.75 s for 30 days, 2 to
    3.6 s for 90, 2026-11-21), while a past day almost never changes. The key is stats.day_stamp
    of the day and of the days either side, because a span that crosses 4am is assembled from
    both days. Today is always worked out afresh (its spend is not in the stamp)."""
    today = logical_date()
    stamps: dict = {}
    def stamp(d):
        if d not in stamps:
            stamps[d] = stats.day_stamp(c, *stats.day_bounds(d))
        return stamps[d]
    out = []
    for i in range(days - 1, -1, -1):
        d = today - dt.timedelta(days=i)
        key = (stamp(d - dt.timedelta(days=1)), stamp(d), stamp(d + dt.timedelta(days=1)))
        hit = _METRICS.get(d)
        if i == 0 or hit is None or hit[0] != key:
            hit = _METRICS[d] = (key, stats.daily_metrics(c, d))
        out.append(hit[1])
    return out


@app.get("/api/agreement")
def api_agreement(days: int = 60):
    """The app's judgments vs the person's: agreements and corrections, overall and by day."""
    return stats.agreement_report(conn(), _capped(days, 1, 366))


@app.get("/api/accuracy")
def api_accuracy(days: int = 60):
    c = conn()
    return {"daily": stats.accuracy_over_time(c, _capped(days, 1, 366)), "scores": ensemble.model_scores(c),
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
    keys = ("clip_start", "clip_end", "duration", "state", "label", "focused", "switch", "self_distraction", "next_task", "sub_of", "sub_absorbed", "diversion", "app", "domain", "title",
            "url", "activity", "content", "hybrid", "switch_id", "own_switch_id", "source", "first_screenshot", "toggl_id", "toggl_desc")
    in_span_ids = {id(s) for s in sp["segments"]}
    segs = []
    for s in (day_segs or sp["segments"]):
        if s["clip_end"] <= w0 or s["clip_start"] >= w1 or s["duration"] <= 0:
            continue
        d = {k: s.get(k) for k in keys}
        d["in_span"] = id(s) in in_span_ids or (day_segs is None)
        segs.append(d)
    from . import exceptions
    ex = exceptions._open_span_row(c, sp["start"], sp["end"])
    return {"exception": {"id": ex["id"], "reason": ex["reason"], "app_call": ex["app_call"], "rule": ex["rule"]} if ex else None,
            "day": day, "start": sp["start"], "end": sp["end"], "duration": sp["duration"],
            "focus_min": sp["focus_min"], "subtype": sp.get("subtype"), "hybrid": bool(sp.get("hybrid")),
            "disq_reason": sp.get("disq_reason"), "detours": len(sp.get("detours", [])),
            "detour_min": sp.get("detour_min", 0), "ended_by": sp.get("ended_by"),
            "sub_threads": sp.get("sub_threads", []),
            "summary": row["summary"] if row else None,
            "title": row["title"] if row else None,
            "coherence_issue": coherence_issue,
            "unfocused_min": sp.get("unfocused_min", 0.0), "unfocused_max_min": sp.get("unfocused_max_min", 0.0),
            "unfocused_ask": bool(sp.get("unfocused_ask")), "unfocused_seg_ids": sp.get("unfocused_seg_ids", []),
            "apps": _apps_by_time(sp["segments"]),
            "toggl": _toggl_with_coverage(c, sp["start"], sp["end"]),
            "phone": phone.window(cfg, w0, w1),
            "toggl_entries": [{"start": max(e["start"], w0), "end": min(e["stop"] or time.time(), w1),
                               "description": e["description"], "project": e["project"]}
                              for e in toggl.entries_between(c, w0, w1)],
            "segments": segs}


@app.get("/api/day-start")
def api_day_start():
    """The hour the logical day starts at (config day_start_hour), for the dashboard's "today"."""
    from .config import day_start_hour
    return {"hour": day_start_hour()}


@app.get("/api/span/queue")
def api_span_queue(days: int = 7):
    """Every finished, unconfirmed hypothesized span of the last N days: today's first
    (chronological within the day), then yesterday's, and so on back."""
    c = conn()
    out = []
    today = logical_date()
    for i in range(_capped(days, 1, 31)):
        day = today - dt.timedelta(days=i)
        t0, t1 = stats.day_bounds(day)
        day_segs = stats.labelled_segments(c, t0, t1)
        se = stats.spans_and_events(day_segs, conn=c)
        for sp in sorted(se["focus_spans"] + se["disqualified"], key=lambda s: s["start"]):
            if sp.get("ended_by") in ("ongoing", "awaiting evaluation"):
                continue
            if sp.get("confirmed") and not sp.get("unfocused_ask"):
                continue   # a confirmed span comes back only while its unfocused-time question is open
            out.append(_span_payload(c, sp, day.isoformat(), day_segs))
    return out


@app.get("/api/span/checks")
def api_span_checks(days: int = 3):
    """The spans the person is asked to confirm, shown ahead of the review questions
    (Eli, 2026-11-22: "spot check every focus span, and then fewer over time as the confidence
    increases", and the same for spans the app leaves out that might count or be an exception).
    Finished spans without a verdict, oldest first; which are asked is stats.span_check_due at
    the rate for the span's own kind of call (stats.span_check_rate). Only the last few days:
    older ones are not remembered well enough (they stay in the full list, /api/span/queue)."""
    c = conn()
    rates: dict = {}
    def rate_for(sp):
        own = sp.get("own_call") or {}
        key = None if own.get("focus") else (own.get("reason") or "")
        if key not in rates:
            rates[key] = stats.span_check_rate(c, key)
        return rates[key]
    out = []
    today = logical_date()
    for i in reversed(range(_capped(days, 1, 7))):
        day = today - dt.timedelta(days=i)
        t0, t1 = stats.day_bounds(day)
        day_segs = stats.labelled_segments(c, t0, t1)
        se = stats.spans_and_events(day_segs, conn=c)
        for sp in sorted(se["focus_spans"] + se["disqualified"], key=lambda s: s["start"]):
            if sp.get("ended_by") in ("ongoing", "awaiting evaluation") or sp.get("confirmed"):
                continue
            rate = rate_for(sp)
            if stats.span_check_due(sp, rate["rate"]):
                out.append(dict(_span_payload(c, sp, day.isoformat(), day_segs), check=rate))
    return {"spans": out}


@app.get("/api/span/at")
def api_span_at(ts: float):
    """The hypothesized span (qualifying or disqualified) containing ts, as a review payload."""
    _ts_arg(ts)
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
    exception: bool = False       # the person ticked "this goes against my written rules"


@app.post("/api/span/review")
def api_span_review(r: SpanReview):
    _ts_arg(r.start, r.end)
    from . import exceptions
    c = conn()
    sp = exceptions.find_span(c, r.start, r.end)   # the app's own call, read before the verdict changes the span
    try:
        edits.write_span_review(c, r.start, r.end, r.verdict, r.subtype, r.note, own=sp.get("own_call") if sp else None)
    except edits.BadEdit as e:
        raise HTTPException(400, str(e))
    # Marked as an exception to the rules: kept on record, with the note as the reason (exceptions.py).
    eid = exceptions.record_span(c, r.start, r.end, r.verdict, r.subtype, sp.get("own_call") if sp else None,
                                 r.exception, reason=r.note if r.exception else None)
    return {"ok": True, "exception_id": eid}


class SpanUnfocused(BaseModel):
    segment_ids: list[int]
    unfocused: bool            # True = "yes, that time was unfocused"; False = "no, I was still on it"
    note: str | None = None


@app.post("/api/span/unfocused")
def api_span_unfocused(r: SpanUnfocused):
    """The person's answer to "the evaluator judged part of this span unfocused - was it?".
    Stored as their focused/unfocused ruling on each of those blocks. Ruled unfocused, the time
    becomes time away from the span (5 minutes of it ends the span there); ruled focused, it is
    ordinary focus time. Either way the question is settled and stops being asked."""
    c = conn()
    note = (r.note or "").strip() or ("(answered: this time in the span was unfocused)" if r.unfocused
                                      else "(answered: this time in the span was focus, not unfocused)")
    try:
        for sid in r.segment_ids[:500]:
            edits.write_seg_review(c, int(sid), focused=not r.unfocused, note=note, merge_existing=True)
    except edits.BadEdit as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "n": len(r.segment_ids)}


@app.delete("/api/span/review")
def api_span_unreview(start: float, end: float):
    """Withdraw the verdict on a span (the span goes back to the system's own call)."""
    _ts_arg(start, end)
    from . import exceptions
    n = edits.delete_span_review(conn(), start, end)
    exceptions.span_verdict_removed(conn(), start, end)
    return {"ok": True, "deleted": n}


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


@app.get("/api/efforts/accuracy")
def api_efforts_accuracy():
    from . import efforts
    return efforts.accuracy(conn())


@app.get("/api/efforts/trials")
def api_efforts_trials(role: str | None = None, effort: str | None = None, limit: int = 50):
    from . import efforts
    return efforts.trials(conn(), role, effort, _capped(limit, 1, 500))   # limit=-1 returned everything (0.66 s)


@app.get("/api/efforts/trial/{trial_id}")
def api_efforts_trial(trial_id: int):
    from . import efforts
    d = efforts.trial_detail(conn(), trial_id)
    if not d:
        raise HTTPException(404)
    return d


@app.get("/api/review/queue")
def api_review_queue(limit: int = 30, all: bool = False):
    limit = _capped(limit, 1, 500)
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


@app.get("/api/segment/{segment_id}/judgment")
def api_segment_judgment(segment_id: int):
    """The evaluator's judgment of one block and the person's ruling on it: what the block editor
    on a question card needs (the capture page gets the same from /api/switch)."""
    c = conn()
    ev = c.execute("SELECT * FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    rv = c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
    return {"eval": dict(ev) if ev else None, "review": dict(rv) if rv else None}


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
    qrow = c.execute("SELECT * FROM review_questions WHERE segment_id=? AND status IN ('open','answered','unsure','none')"
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
                                    AND q.status NOT IN ('declined', 'dismissed', 'archived')  -- as in questions._triggers
                               WHERE e.segment_id=? AND e.uncertain=1 AND r.segment_id IS NULL
                               AND q.id IS NULL""", (to_id,)).fetchone()
        if flagged:
            full["question_pending"] = True
            _kick_question_generation()
    full["to_shots"] = [dict(r) for r in c.execute(
        "SELECT ts, path, display FROM segment_shots WHERE segment_id=? ORDER BY ts", (to_id,))]
    from . import signals as _signals   # what the Mac recorded beside the capture
    full["to_signals"] = _signals.describe(c, full["to"], full["from"]["app"] if full["from"] else None)
    full["to_eval"] = (lambda r: dict(r) if r else None)(
        c.execute("SELECT * FROM seg_evals WHERE segment_id=?", (to_id,)).fetchone())
    full["to_seg_review"] = (lambda r: dict(r) if r else None)(
        c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (to_id,)).fetchone())
    te = next((e for e in toggl.entries_between(c, ts - 1, ts + 1)
               if e["start"] <= ts and (e["stop"] is None or e["stop"] > ts)), None)
    full["toggl"] = {"description": te["description"], "project": te["project"], "tags": te["tags"]} if te else None
    full["phone"] = phone.window(cfg, ts - 900, ts + 900)
    full["toggl_entries"] = [{"start": max(e["start"], ts - 900), "end": min(e["stop"] or time.time(), ts + 900),
                              "description": e["description"], "project": e["project"]}
                             for e in toggl.entries_between(c, ts - 900, ts + 900)]
    # The span this switch lands in (qualifying or disqualified), for audit context.
    from . import summaries as _summ
    full["span"] = None
    full["timeline"] = []
    try:
        ctx_segs = stats.labelled_segments(c, ts - 6 * 3600, ts + 6 * 3600)
        keys = ("id", "clip_start", "clip_end", "duration", "state", "label", "focused", "switch", "self_distraction", "next_task", "sub_of", "sub_absorbed", "diversion", "app", "domain", "title",
                "url", "activity", "content", "hybrid", "switch_id", "own_switch_id", "source", "first_screenshot", "toggl_id", "toggl_desc")
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
    d, t0, t1 = _day_arg(day)
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
    # only those inside the window the segments cover (a long block reaching into it brought all
    # of its captures, hours outside it: 2026-11-21)
    lo = min(s.get("clip_start", s.get("start", 0)) for s in timeline)
    hi = max(s.get("clip_end", s.get("end") or 0) or 0 for s in timeline)
    rows = [dict(r) for r in c.execute(
        f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) AND ts >= ? AND ts <= ? ORDER BY ts",
        ids + [lo - 1, hi + 1])]
    return rows or [{"segment_id": s["id"], "ts": s["clip_start"], "path": s["first_screenshot"], "display": 0}
                    for s in timeline if s.get("first_screenshot")]


def _question_segments(c, segment_id: int, t0: float, t1: float) -> list[dict]:
    """The blocks drawn on a question card: the usual timeline, with the question's own block
    always in it. A phone app segment judged time away is carved out of the Mac record
    (stats._carve_phone), and where it covered the whole of the question's block that block was
    gone from the card: nothing highlighted, no "moment in question" captures (9 of 224 open
    questions, 2026-11-21: 3153, 3165, 3185, 3192, 3277, 3194, 3196, 3216, 3229). Here the
    block is put back whole and whatever overlapped it is cut back to what lies outside it.
    The phone lane under the timeline still shows the phone use."""
    segs = [s for s in stats.labelled_segments(c, t0, t1) if s["duration"] > 0]
    if any(s["id"] == segment_id for s in segs):
        return segs
    own = next((s for s in stats.labelled_segments(c, t0, t1, phone=False)
                if s["id"] == segment_id and s["duration"] > 0), None)
    if own is None:
        return segs
    a, b = own["clip_start"], own["clip_end"]
    out = [own]
    for s in segs:
        if s["clip_end"] <= a or s["clip_start"] >= b:
            out.append(s)
            continue
        if s["clip_start"] < a:
            out.append(dict(s, clip_end=a, duration=a - s["clip_start"]))
        if s["clip_end"] > b:
            out.append(dict(s, clip_start=b, duration=s["clip_end"] - b))
    out.sort(key=lambda s: s["clip_start"])
    return out


@app.get("/api/questions/queue")
def api_questions_queue(limit: int = 1000):
    """Open review questions in chronological order, oldest first (user ruling 2026-11-16: "next"
    must be the next question on that day; the earlier priority-tier-first order jumped between
    days). The tier still rides on each question for the card to show. Nothing older than
    question_archive_days is open, so the whole list is short enough to return."""
    from . import questions as _q
    from .config import logical_date
    c = conn()
    import json as _json
    # Scoring runs beside the request, not inside it. Every answer makes scores stale, so the
    # first load after one waited 3 to 5 s here (the delay Eli saw after clicking a review dot,
    # 2026-11-21). The tiers shown can be one load behind.
    def _score():
        if not _score_busy.acquire(blocking=False):
            return
        try:
            c2 = db.connect()
            try:
                _q.score_questions(c2, limit=25)
            finally:
                c2.close()
        except Exception:
            log.exception("question scoring failed")
        finally:
            _score_busy.release()
    threading.Thread(target=_score, daemon=True, name="score-questions").start()
    out = []
    keys = ("id", "clip_start", "clip_end", "duration", "state", "label", "focused", "switch", "self_distraction", "next_task", "sub_of", "sub_absorbed", "diversion", "app", "domain", "title",
            "content", "hybrid", "switch_id", "own_switch_id", "source", "first_screenshot", "last_screenshot", "url", "toggl_id", "toggl_desc")
    rows = [dict(r) for r in c.execute("""SELECT q.*, g.start AS seg_start, COALESCE(g.end, g.start) AS seg_end,
                                 g.app, g.title, g.domain,
                                 (SELECT id FROM switches WHERE to_segment = q.segment_id LIMIT 1) AS switch_id
                          FROM review_questions q JOIN segments g ON g.id = q.segment_id
                          WHERE q.status = 'open'""")]
    for d in rows:
        d["tier"] = ("priority" if (d.get("priority") or 0) >= _q.PRIORITY_CUT else
                     "routine" if d.get("priority") is not None else "unscored")
        d["day"] = logical_date(d["seg_start"]).isoformat()
    rows.sort(key=lambda d: d["seg_start"])
    counts = {"priority": sum(1 for d in rows if d["tier"] == "priority"), "total": len(rows)}
    shown = rows[:limit]
    for d in shown:
        d.pop("undo", None)
        d["options"] = _json.loads(d["options"])
        d["stakes_text"] = _q.stakes_text(d.get("stakes_detail"))
        d["queue_counts"] = counts
        # No timeline here: building one per question made this endpoint take 25 s and 10 MB
        # once ~460 priority questions were open (2026-11-14). The card fetches its own from
        # /api/questions/{id}/context when it is shown.
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
            segs = _question_segments(c, t["segment_id"], seg_start - QCTX_S, seg_end + QCTX_S)
            out.append({
                "id": None, "pending": True, "segment_id": t["segment_id"], "source": t["source"],
                "day": logical_date(seg_start).isoformat(),
                "context": t["reason"], "question": None, "options": [], "tier": None,
                "seg_start": seg_start, "seg_end": seg_end,
                "app": g["app"], "title": g["title"], "domain": g["domain"],
                "switch_id": (lambda r: r["id"] if r else None)(c.execute(
                    "SELECT id FROM switches WHERE to_segment=? LIMIT 1", (t["segment_id"],)).fetchone()),
                "timeline": [{k: s.get(k) for k in keys} for s in segs if s["duration"] > 0],
                "timeline_shots": _shots_for(c, [s for s in segs if s["duration"] > 0]),
                "phone": phone.window(cfg, seg_start - QCTX_S, seg_end + QCTX_S),
                "toggl_entries": [{"start": max(e["start"], seg_start - QCTX_S),
                                   "end": min(e["stop"] or time.time(), seg_end + QCTX_S),
                                   "description": e["description"], "project": e["project"]}
                                  for e in toggl.entries_between(c, seg_start - QCTX_S, seg_end + QCTX_S)]})
        if pending:
            _kick_question_generation()
    except Exception:
        log.exception("pending-question placeholders failed")
    out.sort(key=lambda d: d["seg_start"])  # placeholders for questions still being written take their place in time
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


def _spot_unmark(c, segment_id: int) -> None:
    """Take the "spot check: the person says ... is wrong" mark back off the app's reasoning."""
    import re
    from . import questions as _q
    r = c.execute("SELECT rationale FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    if r and r["rationale"] and _q.SPOT_MARK in r["rationale"]:
        c.execute("UPDATE seg_evals SET rationale=? WHERE segment_id=?",
                  (re.sub(re.escape(_q.SPOT_MARK) + r"[^|]*?(?= \| |$)", "", r["rationale"]), segment_id))


def _spot_unanswer(c, segment_id: int) -> None:
    """A spot check's answer was cleared (or became "I can't tell"): no verdict, so it does not
    count in the agreement figure, and no mark left on the reasoning."""
    c.execute("UPDATE agreement_samples SET verdict=NULL, answered_at=NULL WHERE segment_id=?", (segment_id,))
    _spot_unmark(c, segment_id)


_SEG_COLS = ("segment_id", "content", "switch_label", "interruption_kind", "antecedent_id", "note", "created", "hybrid",
             "chat_message_id", "focus_exception", "focused", "self_distraction", "next_task", "sub_of")


def _ruling_row(c, o: dict, prev: dict | None, note: str | None, chat_message_id: int | None, created: float) -> dict | None:
    """The seg_reviews row that choosing option `o` makes of the ruling `prev` (None = no ruling
    yet): the option's fields laid over the earlier ones. None when the option says nothing
    about the block. One function for the answer itself and for replaying it (see
    _undo_question_answer)."""
    # self_distraction on an option (since 2026-11-20): true = I drifted there without deciding
    # to, false = I went there on purpose. Before, an answer could keep or clear the
    # evaluator's flag but never set it, so "I was after something specific: an errand"
    # (q2942: content task, focused) left the visit counted as a self-distraction event.
    # Options stored before that have no such field: None = leave it as it is. Like
    # edits.write_seg_review, it is not kept on a continuation, where it means nothing.
    o_sd = o.get("self_distraction") if o.get("switch_label") != "continuation" else None
    # next_task and focus_exception (2026-11-20) come only on an option made from a judgment the
    # evaluator proposed in chat (or, for next_task, one the day chat wrote): "that was the next
    # task" and "count this as focus anyway" could be said in the chat and not recorded.
    o_nt = o.get("next_task") if o.get("switch_label") in (None, "interruption") else None
    o_fx = True if o.get("focus_exception") is True and o.get("proposed_from") else None
    if not (o.get("switch_label") or o.get("content") or o.get("hybrid") is not None or o.get("focused") is not None
            or o_sd is not None or o_nt is not None or o_fx):
        return None
    osw = o.get("switch_label")
    _keep = lambda k: prev.get(k) if prev is not None else None
    ante = (_valid_antecedent(c, o["segment_id"], o.get("antecedent_id")) if osw in ("continuation", "return")
            else (None if osw else _keep("antecedent_id")))
    kept_kind = _keep("interruption_kind")
    if o_sd is False and kept_kind == "self_distraction":
        kept_kind = None   # an older ruling's kind must not contradict "on purpose"
    return {
        "segment_id": o["segment_id"],
        "content": o.get("content") or _keep("content"),
        "switch_label": osw or _keep("switch_label"),
        "interruption_kind": o.get("interruption_kind") if osw == "interruption" else (None if osw else kept_kind),
        "antecedent_id": ante,
        "note": note.strip() if note and note.strip() else f"(answered review question: {o['label']})",
        "created": created,
        "hybrid": int(o["hybrid"]) if o.get("hybrid") is not None else _keep("hybrid"),
        "chat_message_id": chat_message_id if chat_message_id is not None else o.get("proposed_from"),
        "focus_exception": 1 if o_fx else _keep("focus_exception"),   # only a chat proposal the person applies sets it
        # the person's focused/unfocused ruling: set by the option, else kept
        "focused": int(bool(o["focused"])) if o.get("focused") is not None else _keep("focused"),
        "self_distraction": int(bool(o_sd)) if o_sd is not None else (None if osw else _keep("self_distraction")),
        "next_task": (int(bool(o_nt)) if o_nt is not None and (osw or _keep("switch_label")) == "interruption"
                      else (None if osw else _keep("next_task"))),
        # a sub-thread option names what it grew out of; a plain "separate thread" option on a
        # block the app had as a sub-thread says it is not one (0)
        "sub_of": ((_valid_antecedent(c, o["segment_id"], o.get("sub_of")) if o.get("sub_of") else 0)
                   if osw == "interruption" else (None if osw else _keep("sub_of"))),
    }


def _standing_answers(c, seg_id: int, not_qid: int | None) -> list[tuple[dict, dict]]:
    """(question row, its undo record) for every answered question whose answer wrote a ruling
    on this block, oldest answer first."""
    import json as _json
    out = []
    for q in c.execute("SELECT * FROM review_questions WHERE status='answered' AND undo LIKE ?",
                       (f'%"seg_id": {int(seg_id)}%',)).fetchall():
        u = _json.loads(q["undo"] or "{}")
        if q["id"] != not_qid and u.get("seg_id") == seg_id and u.get("seg_written") is not None:
            out.append((dict(q), u))
    out.sort(key=lambda t: t[1]["seg_written"])
    return out


def _undo_question_answer(c, undo_json: str | None, qid: int | None = None) -> dict:
    """Take one answer to a review question back out of the record, so that changing or
    clearing an answer leaves no residue.

    Several questions can sit on one block (one about its switch, one about its content), and
    their answers are laid over each other in one seg_reviews row. Until 2026-11-21 each answer
    simply restored the row it had found: block 36902, answer q3171, answer q3172, clear q3171,
    clear q3172 left q3171's ruling behind with both questions open (clearing q3171 saw a row
    that was no longer its own and left it; clearing q3172 then put back the row from before
    it, which was q3171's). Now, when later answers stand on top of this one, the row is built
    again from the ruling this answer found plus those later answers in order, and each of them
    is handed the row it now rests on. Whatever order the answers are given and cleared in, the
    block ends with exactly the answers still standing, and with nothing when none are.
    A ruling edited by hand after the answer is still not the answer's to take back."""
    import json as _json
    u = _json.loads(undo_json) if undo_json else {}
    seg = u.get("seg_id")
    if seg is not None:
        cur = c.execute("SELECT created FROM seg_reviews WHERE segment_id=?", (seg,)).fetchone()
        mine = u.get("seg_written")
        if mine is None or cur is None or abs(cur["created"] - mine) <= 1e-3:
            _restore_rows(c, "seg_reviews", "segment_id", seg, u.get("seg_review"))
        else:
            later = [t for t in _standing_answers(c, seg, qid) if t[1]["seg_written"] > mine]
            if later and abs(later[-1][1]["seg_written"] - cur["created"]) <= 1e-3:
                row = u.get("seg_review")
                for q2, u2 in later:
                    o2 = _json.loads(q2["options"])[q2["answer_index"]]
                    u2["seg_review"] = row
                    row = _ruling_row(c, o2, row, q2["note"], u2.get("seg_cmid"), u2["seg_written"]) or row
                    c.execute("UPDATE review_questions SET undo=? WHERE id=?", (_json.dumps(u2), q2["id"]))
                _restore_rows(c, "seg_reviews", "segment_id", seg, row)
            elif later:
                # a hand edit on top of everything: the row stays, and the next answer up now
                # rests on what this one had found
                later[0][1]["seg_review"] = u.get("seg_review")
                c.execute("UPDATE review_questions SET undo=? WHERE id=?", (_json.dumps(later[0][1]), later[0][0]["id"]))
    if u.get("phone_key") is not None:
        _restore_rows(c, "phone_reviews", "seg_key", u["phone_key"], u.get("phone_review"))
    if u.get("switch_id") is not None:
        # an answer recorded before 2026-11-21 kept the legacy switch review it replaced
        _restore_rows(c, "reviews", "switch_id", u["switch_id"], u.get("review"))
        c.execute("UPDATE switches SET status=? WHERE id=?", (u.get("switch_status"), u["switch_id"]))
    if seg is not None:
        edits.sync_legacy(c, seg)   # the legacy switch review follows whatever ruling is left
    return u


def _flag_before(c, q, old_u: dict) -> int | None:
    """seg_evals.uncertain as it was before any question on this block was settled. A second
    question answered on the same block used to record the flag as the first answer had left
    it (0), so clearing both in the order they were given ended with the doubt flag off."""
    import json as _json
    if old_u.get("uncertain") is not None:
        return old_u["uncertain"]
    for r in c.execute("SELECT undo FROM review_questions WHERE segment_id=? AND id!=? AND status IN ('answered','unsure','none')"
                       " AND undo IS NOT NULL", (q["segment_id"], q["id"])).fetchall():
        v = _json.loads(r["undo"] or "{}").get("uncertain")
        if v is not None:
            return v
    ev0 = c.execute("SELECT uncertain FROM seg_evals WHERE segment_id=?", (q["segment_id"],)).fetchone()
    return ev0["uncertain"] if ev0 else None


def _other_settled(c, q) -> bool:
    """Is another question on this block still settled in a way that cleared its doubt flag?"""
    import json as _json
    for r in c.execute("SELECT status, source, options, answer_index FROM review_questions WHERE segment_id=? AND id!=?"
                       " AND status IN ('answered','unsure','none')", (q["segment_id"], q["id"])).fetchall():
        if r["status"] != "answered":
            if r["source"] != "phone":
                return True
        elif r["answer_index"] is not None and not _json.loads(r["options"])[r["answer_index"]].get("phone"):
            return True
    return False


def _unapply_proposals(c, q, keep_index: int | None = None) -> None:
    """A chat proposal applied as the answer to this question is no longer the answer: take
    its "applied" mark off, so its card can be clicked again. (2026-11-21: the card stayed
    "applied as your answer (option 4)", disabled, after that answer was cleared.)"""
    import json as _json
    for m in c.execute("SELECT id, proposal FROM chat_messages WHERE kind='evaluator' AND ref=? AND proposal LIKE '%\"applied\"%'",
                       (str(q["segment_id"]),)).fetchall():
        prop = _json.loads(m["proposal"])
        aj = (prop.get("applied") or {}).get("judgment")
        if aj and aj.get("question_id") == q["id"] and (keep_index is None or aj.get("option_index") != keep_index):
            prop["applied"].pop("judgment")
            _save_proposal(c, m["id"], prop)


@app.get("/api/questions/{qid}/context")
def api_question_context(qid: int):
    """The timeline around one question's segment, fetched when its card is shown."""
    c = conn()
    g = c.execute("""SELECT q.segment_id, g.start AS seg_start, COALESCE(g.end, g.start) AS seg_end
                     FROM review_questions q JOIN segments g ON g.id = q.segment_id WHERE q.id=?""", (qid,)).fetchone()
    if not g:
        raise HTTPException(404)
    t0, t1 = g["seg_start"] - QCTX_S, g["seg_end"] + QCTX_S
    keys = ("id", "clip_start", "clip_end", "duration", "state", "label", "focused", "switch", "self_distraction", "next_task", "sub_of", "sub_absorbed", "diversion", "app", "domain", "title",
            "content", "hybrid", "switch_id", "own_switch_id", "source", "first_screenshot", "last_screenshot", "url", "toggl_id", "toggl_desc")
    timeline = [{k: s.get(k) for k in keys} for s in _question_segments(c, g["segment_id"], t0, t1)]
    return {"timeline": timeline, "timeline_shots": _shots_for(c, timeline),
            "phone": phone.window(cfg, t0, t1),
            "toggl_entries": [{"start": max(e["start"], t0), "end": min(e["stop"] or time.time(), t1),
                               "description": e["description"], "project": e["project"]}
                              for e in toggl.entries_between(c, t0, t1)]}


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
        old_u = (_undo_question_answer(c, q["undo"], qid) if q["status"] == "answered"
                 else (_json.loads(q["undo"]) if q["status"] in ("unsure", "none") and q["undo"] else {}))
        undo = {"uncertain": _flag_before(c, q, old_u)}
        if o.get("phone"):   # a ruling on a phone app segment: its own table, read by stats._phone_rows
            prev_p = c.execute("SELECT * FROM phone_reviews WHERE seg_key=?", (o["phone_key"],)).fetchone()
            undo.update(phone_key=o["phone_key"], phone_review=dict(prev_p) if prev_p else None)
            e = dict(o["phone"])
            if prev_p is not None:   # what it was used for and whether it was time away are asked separately: keep the other
                for k in ("switch_label", "content", "focused", "self_distraction", "away"):
                    if e.get(k) is None:
                        e[k] = prev_p[k]
            c.execute("INSERT OR REPLACE INTO phone_reviews(seg_key, switch_label, content, focused, self_distraction, away, note, created)"
                      " VALUES (?,?,?,?,?,?,?,?)",
                      (o["phone_key"], e.get("switch_label"), e.get("content"), e.get("focused"), e.get("self_distraction"),
                       e.get("away"), (a.note.strip() if a.note and a.note.strip() else f"(answered review question: {o['label']})"),
                       time.time()))
        prev = c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (o["segment_id"],)).fetchone() if o.get("segment_id") else None
        prev = dict(prev) if prev is not None else None
        row = _ruling_row(c, o, prev, a.note, chat_message_id, time.time()) if o.get("segment_id") else None
        if row is not None:
            _restore_rows(c, "seg_reviews", "segment_id", o["segment_id"], row)
            # seg_written: which row this answer wrote, so clearing the answer later can tell whether
            # the ruling is still that one, has another answer on top, or was edited by hand since;
            # seg_cmid: the chat message it came from, for replaying it (see _undo_question_answer)
            undo.update(seg_id=o["segment_id"], seg_review=prev, seg_written=row["created"], seg_cmid=row["chat_message_id"])
            from . import propagation
            propagation.note_event(c, o["segment_id"])
            # the legacy switch review follows the ruling (edits.sync_legacy); until 2026-11-21 it
            # was written here and its earlier state kept in this undo record
            edits.sync_legacy(c, o["segment_id"])
        _unapply_proposals(c, q, keep_index=option_index)
        c.execute("UPDATE review_questions SET status='answered', answer=?, answer_index=?, note=?, answered_at=?, undo=?"
                  " WHERE id=?",
                  (o["label"], a.option_index, (a.note.strip() or None) if a.note else None, time.time(),
                   _json.dumps(undo), qid))
        # The answer resolves the doubt this question was generated for, even when the chosen
        # option edits a different segment - otherwise the needs-review dot never goes away.
        if q["source"] == "sample":   # a spot check (questions.spot_checks): record the verdict
            c.execute("UPDATE agreement_samples SET verdict=?, answered_at=? WHERE segment_id=?",
                      (o.get("sample_verdict"), time.time(), q["segment_id"]))
        if q["source"] == "sample":   # no stale mark from an earlier 'No' on this spot check
            _spot_unmark(c, q["segment_id"])
        if o.get("sample_verdict") == "disagree" and o.get("focused") is None:
            # he says the confident judgment is wrong on this axis: raise the doubt, so the usual
            # question about what it really was gets written ('No' on focused is itself the ruling)
            from .questions import SPOT_MARK
            wrong = {"content": "the kind of activity", "switch": "the switch (continuation / shift / return)"}.get(o.get("axis"), "this judgment")
            c.execute("UPDATE seg_evals SET uncertain=1, rationale=COALESCE(rationale,'') || ? WHERE segment_id=?",
                      (f"{SPOT_MARK}{wrong} is wrong" + (f" ({a.note.strip()})" if a.note and a.note.strip() else ""),
                       q["segment_id"]))
        elif not o.get("phone"):   # a phone question only borrows its computer segment as an anchor
            c.execute("UPDATE seg_evals SET uncertain=0 WHERE segment_id=?", (q["segment_id"],))
    if o.get("switch_label") == "interruption" and not o.get("interruption_kind") and not o.get("sample_verdict"):
        _kick_kind_derivation(o["segment_id"], a.note)   # (a spot-check 'Yes' only confirms: nothing to derive)
    return {"ok": True}


@app.get("/api/llm/status")
def api_llm_status():
    """Which account the judgment calls are drawing on right now, and today's calls per backend."""
    from . import llm
    return llm.status()


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
    if q["status"] not in ("answered", "unsure", "none"):
        return {"ok": True}
    with db.tx(c):
        u = _undo_question_answer(c, q["undo"], qid) if q["status"] == "answered" else (_json.loads(q["undo"]) if q["undo"] else {})
        c.execute("UPDATE review_questions SET status='open', answer=NULL, answer_index=NULL, note=NULL,"
                  " answered_at=NULL, undo=NULL WHERE id=?", (qid,))
        # the doubt flag comes back only when no other question on the block still settles it
        # (two questions on one block, 2026-11-21: see _flag_before)
        if u.get("uncertain") is not None and not _other_settled(c, q):
            c.execute("UPDATE seg_evals SET uncertain=? WHERE segment_id=?", (u["uncertain"], q["segment_id"]))
        _unapply_proposals(c, q)
        if q["source"] == "sample":
            _spot_unanswer(c, q["segment_id"])
    return {"ok": True}


class UnsureReq(BaseModel):
    note: str | None = None


def _settle_without_ruling(c, qid: int, status: str, answer: str, note: str | None) -> dict:
    """Resolve a question without choosing an option. Any earlier answer is undone; the
    needs-review flag clears (the person has looked); the record is kept for the generator."""
    import json as _json
    q = c.execute("SELECT * FROM review_questions WHERE id=?", (qid,)).fetchone()
    if not q:
        raise HTTPException(404)
    with db.tx(c):
        if q["status"] == "answered":
            _undo_question_answer(c, q["undo"], qid)
            undo = _json.loads(q["undo"]) if q["undo"] else {}
            undo = {"uncertain": undo.get("uncertain")}
            _unapply_proposals(c, q)
        elif q["status"] in ("unsure", "none"):
            undo = _json.loads(q["undo"]) if q["undo"] else {}
        else:
            undo = {"uncertain": _flag_before(c, q, {})}
        c.execute("UPDATE review_questions SET status=?, answer=?, answer_index=NULL, note=?, answered_at=?, undo=?"
                  " WHERE id=?", (status, answer, (note.strip() or None) if note else None, time.time(),
                                  _json.dumps(undo), qid))
        if q["source"] != "phone":
            c.execute("UPDATE seg_evals SET uncertain=0 WHERE segment_id=?", (q["segment_id"],))
        if q["source"] == "sample":
            _spot_unanswer(c, q["segment_id"])
    return {"ok": True}


@app.post("/api/questions/{qid}/unsure")
def api_question_unsure(qid: int, body: UnsureReq):
    """'I can't tell': the question is resolved without a ruling - the model's judgment stands,
    the needs-review flag clears, and the record of an unanswerable doubt is kept so the question
    generator learns which doubts are not worth raising. Reopen puts it back."""
    return _settle_without_ruling(conn(), qid, "unsure", "(can't tell)", body.note)


@app.post("/api/questions/{qid}/none")
def api_question_none(qid: int, body: UnsureReq):
    """'None of the above': the person knows what happened but no option says it. Nothing is
    written to the segment's judgment (there is no option to write); the note is the record in
    their words, and the evaluator chat under the question is where it becomes a judgment - a
    proposal applied there is appended as a new option and answers the question. The generator
    is shown these so its options cover the real readings next time."""
    return _settle_without_ruling(conn(), qid, "none", "(none of the options)", body.note)


@app.post("/api/questions/{qid}/dismiss")
def api_question_dismiss(qid: int):
    c = conn()
    if not c.execute("SELECT 1 FROM review_questions WHERE id=?", (qid,)).fetchone():
        raise HTTPException(404, "no such question")   # it used to answer ok for an id that does not exist
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


@app.post("/api/questions/{qid}/followup")
def api_question_followup(qid: int):
    """Called by the review card right after an answer: the evaluator may ask one follow-up
    question when the answer went against its judgment (evaluator.followup_question)."""
    from . import evaluator
    try:
        return evaluator.followup_question(conn(), cfg, qid)
    except Exception as e:
        log.exception("follow-up question failed")
        raise HTTPException(500, str(e))


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
                         "focused": j.get("focused"), "self_distraction": j.get("self_distraction"),
                         # next_task / focus_exception: only a judgment proposed in chat carries them
                         "next_task": j.get("next_task"), "focus_exception": j.get("focus_exception"),
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
                                        focused=j.get("focused"), self_distraction=j.get("self_distraction"),
                                        next_task=j.get("next_task"), focus_exception=j.get("focus_exception"),
                                        note=j.get("note") or j["label"], chat_message_id=mid)
    except edits.BadEdit as e:
        raise HTTPException(400, str(e))
    if derive:
        _kick_kind_derivation(segment_id, j.get("note"))
    applied["judgment"] = {"at": time.time(), "segment_id": segment_id}
    with db.tx(c):
        _save_proposal(c, mid, prop)
    return {"ok": True, "segment_id": segment_id, "proposal": prop}


def _append_rule(rule: str, origin: str) -> None:
    """Add one line to the rules document under 'Lessons from reviews', dated."""
    from .classifiers.claude_vision import RULES_PATH
    heading = "### Lessons from reviews"
    text = RULES_PATH.read_text() if RULES_PATH.exists() else ""
    if heading not in text:
        text = text.rstrip("\n") + "\n\n" + heading + "\n\n"
    RULES_PATH.write_text(text.rstrip("\n") + "\n" + f"* {rule} (added {dt.date.today().isoformat()} {origin})\n")
    _commit_rules(rule)


def _commit_rules(rule: str) -> None:
    """Commit the rules document after a rule is accepted (Eli, 2026-11-21), so its history is in
    git. Only that file, whatever else is modified or staged; any edits of Eli's own that were
    waiting in it go into the same commit. Never from a test run on a database copy, and a
    failure only logs."""
    import os
    import subprocess
    from .classifiers.claude_vision import RULES_PATH
    if os.environ.get("FOCUS_MONITOR_DATA"):
        return
    try:
        msg = "Rules: accepted from the dashboard: " + (rule if len(rule) <= 160 else rule[:157] + "...")
        r = subprocess.run(["git", "commit", "-q", "-m", msg, "--", RULES_PATH.name], cwd=RULES_PATH.parent,
                           capture_output=True, text=True, timeout=20)
        if r.returncode:
            log.warning("could not commit the rules document: %s", (r.stderr or r.stdout).strip()[:300])
        else:
            log.info("committed the rules document")
    except Exception:
        log.exception("could not commit the rules document")


@app.get("/api/rule-drafts")
def api_rule_drafts():
    """Rules the app drafted from repeated rulings: the open ones with the rulings each rests on,
    and the recently decided ones (rule_drafts.py)."""
    from . import rule_drafts
    return rule_drafts.drafts(conn())


@app.get("/api/rule-exceptions")
def api_rule_exceptions():
    """The rulings made against the written rules that still stand (exceptions.py)."""
    from . import exceptions
    return exceptions.listing(conn())


class ExceptionMark(BaseModel):
    segment_id: int
    reason: str | None = None


@app.post("/api/rule-exceptions/mark-segment")
def api_rule_exception_mark(body: ExceptionMark):
    """Mark the ruling on one block as an exception to the rules, by hand."""
    from . import exceptions
    eid = exceptions.mark_segment(conn(), body.segment_id, body.reason)
    if eid is None:
        raise HTTPException(400, "rule on this block first: an exception is a ruling of yours")
    return {"ok": True, "exception_id": eid}


@app.delete("/api/rule-exceptions/{eid}")
def api_rule_exception_unmark(eid: int):
    from . import exceptions
    if not exceptions.withdraw(conn(), eid):
        raise HTTPException(404)
    return {"ok": True}


class ExceptionReason(BaseModel):
    reason: str


@app.post("/api/rule-exceptions/{eid}/reason")
def api_rule_exception_reason(eid: int, body: ExceptionReason):
    from . import exceptions
    if not exceptions.set_reason(conn(), eid, body.reason):
        raise HTTPException(404)
    return {"ok": True}


@app.get("/api/rule-drafts/count")
def api_rule_drafts_count():
    return {"open": conn().execute("SELECT COUNT(*) FROM rule_drafts WHERE status='open'").fetchone()[0]}


class RuleDecision(BaseModel):
    action: str               # add | reject
    text: str | None = None   # the wording to add, if the person edited it


@app.post("/api/rule-drafts/{did}/decide")
def api_rule_draft_decide(did: int, body: RuleDecision):
    c = conn()
    d = c.execute("SELECT * FROM rule_drafts WHERE id=?", (did,)).fetchone()
    if not d or body.action not in ("add", "reject"):
        raise HTTPException(404 if not d else 400)
    if d["status"] != "open":
        return {"ok": True, "status": d["status"]}
    rule = " ".join((d["rule"] if body.text is None else body.text).split())   # emptied on purpose stays empty
    if body.action == "add":
        if not rule:
            raise HTTPException(400, "empty rule")
        import json as _json
        n_ans, n_exc = len(_json.loads(d['support'] or '[]')), len(_json.loads(d['exceptions'] or '[]'))
        _append_rule(rule, "drafted from " + " and ".join(
            x for x in (f"{n_ans} of your review answers" if n_ans else "",
                        f"{n_exc} exception{'s' if n_exc != 1 else ''} you made" if n_exc else "") if x))
    with db.tx(c):
        c.execute("UPDATE rule_drafts SET status=?, decided_at=?, final_text=? WHERE id=?",
                  ("added" if body.action == "add" else "rejected", time.time(), rule if body.action == "add" else None, did))
    return {"ok": True, "status": "added" if body.action == "add" else "rejected"}


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
        _commit_rules(rule)
        applied["rule"] = {"at": time.time()}
        with db.tx(c):
            _save_proposal(c, mid, prop)
    return {"ok": True, "proposal": prop}


@app.post("/api/span/chat")
def api_span_chat(req: SpanChatReq):
    try:
        return get_span_chat().reply(conn(), req.start, req.end, req.messages)   # reply, message_id, applied, proposal
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
    focused: bool | None = None           # the person's ruling: attention was focused (True) / unfocused (False)
    self_distraction: bool | None = None  # on a shift: sought out stimulation without a clear intent
    next_task: bool | None = None         # on a shift: the next task under the same task-processing intention
    sub_of: int | None = None             # on a shift: a sub-thread of this block's thread; -1 = the block before; 0 = not one


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
                                        note=r.note, hybrid=r.hybrid, focus_exception=r.focus_exception,
                                        focused=r.focused, self_distraction=r.self_distraction,
                                        next_task=r.next_task, sub_of=r.sub_of)
    except edits.BadEdit as e:
        raise HTTPException(400 if "no segment" not in str(e) else 404, str(e))
    if derive:
        _kick_kind_derivation(segment_id, r.note)
    return {"ok": True}


@app.delete("/api/segment/{segment_id}/eval-review")
def api_seg_eval_unreview(segment_id: int):
    c = conn()
    # If this ruling is the one a question's answer wrote, "undo edits" is clearing that answer:
    # the question goes back to open and the doubt flag comes back with it. Deleting the row alone
    # left the question marked answered with no ruling behind it, so the doubt just vanished.
    import json as _json
    cur = c.execute("SELECT created FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
    if cur is not None:
        for q in c.execute("SELECT id, undo FROM review_questions WHERE status='answered' AND undo LIKE ?"
                           " ORDER BY answered_at DESC", (f'%"seg_id": {segment_id},%',)).fetchall():
            u = _json.loads(q["undo"] or "{}")
            if u.get("seg_id") == segment_id and u.get("seg_written") is not None and abs(u["seg_written"] - cur["created"]) <= 1e-3:
                return api_question_reopen(q["id"])
    return {"ok": True, "restored": edits.undo_seg_review(c, segment_id)}


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
    # rglob: classifiers/ is a subpackage, and an edit there used to read as "up to date"
    return sorted(here.rglob("*.py")) + sorted(here.glob("*.html")) + [CONFIG_PATH]


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

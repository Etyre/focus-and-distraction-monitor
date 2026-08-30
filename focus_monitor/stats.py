"""Derive states, spans, events and daily metrics from segments + labels.

State machine: each segment gets a state from the label of the switch that led into it.
  continuation -> inherits the previous segment's state (focus stays focus, distracted stays distracted)
  task_change  -> 'focus' ("focus start"): begins or resumes a focus span
  interruption -> 'interrupted'
  distraction  -> 'distracted'
  idle segment -> 'idle' (breaks spans)
  neutral segment (presence check etc.) -> inherits the previous segment's state; never an event
A human review overrides the ensemble label. Unclassified switches count as continuation.
"""
from __future__ import annotations

import datetime as dt
import time

from . import db

STATE_OF = {"task_change": "focus", "interruption": "interrupted", "distraction": "distracted"}


def _has_toggl(conn) -> bool:
    return bool(conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='toggl_entries'").fetchone())


from .config import day_bounds, logical_date  # day boundary lives in config (default 4am-4am)


def logical_today() -> dt.date:
    return logical_date()


def labelled_segments(conn, t0: float, t1: float) -> list[dict]:
    from .config import load_config
    cfg = load_config()
    now = time.time()
    segs = [dict(r) for r in db.segments_between(conn, t0, t1)]
    if not segs:
        return []
    ids = [s["id"] for s in segs]
    q = ",".join("?" * len(ids))
    sw = conn.execute(f"""
        SELECT s.id, s.to_segment, s.status, s.group_id, e.label AS e_label, e.uncertain, r.label AS r_label,
               e.p_continuation, e.p_interruption, e.p_distraction, e.p_task_change, e.activity AS e_activity
        FROM switches s
        LEFT JOIN ensemble e ON e.switch_id = COALESCE(s.group_id, s.id)
        LEFT JOIN reviews r ON r.switch_id = COALESCE(s.group_id, s.id)
        WHERE s.to_segment IN ({q})""", ids).fetchall()
    tg = conn.execute("SELECT id, start, stop, tags, description FROM toggl_entries WHERE start < ? AND COALESCE(stop, ?) > ? ORDER BY start",
                      (t1, now, t0)).fetchall() if _has_toggl(conn) else []
    def toggl_at(ts):
        for e in tg:
            if e["start"] <= ts and (e["stop"] is None or e["stop"] > ts):
                return e
        return None
    by_to = {r["to_segment"]: r for r in sw}
    prev_state = "focus"
    last_focus_ctx = None  # (app, domain) of the most recent focused segment
    for s in segs:
        s["end_eff"] = s["end"] or now
        s["clip_start"], s["clip_end"] = max(s["start"], t0), min(s["end_eff"], t1)
        s["duration"] = max(0.0, s["clip_end"] - s["clip_start"])
        _te = toggl_at(s["start"])
        s["toggl_id"] = _te["id"] if _te else None
        s["toggl_tags"] = (_te["tags"] or "") if _te else ""
        s["toggl_desc"] = (_te["description"] or "") if _te else ""
        r = by_to.get(s["id"])
        if s["idle"]:
            s.update(state="idle", label=None, switch_id=None, uncertain=0, source=None)
            prev_state = "idle"
            continue
        if s.get("neutral"):
            s.update(state=prev_state if prev_state != "idle" else "focus", label=None, switch_id=None,
                     uncertain=0, source="neutral")
            if s["state"] == "focus":
                last_focus_ctx = (s["app"], s["domain"])
            prev_state = s["state"]
            continue
        if r is None:  # first segment, or a return to the same window after a neutral one
            st = prev_state if prev_state != "idle" else "focus"
            s.update(state=st, label="continuation", switch_id=None, uncertain=0, source="start")
        else:
            label = r["r_label"] or r["e_label"] or "continuation"
            transit = r["status"] == "transit"
            if label == "continuation":
                if prev_state == "idle":
                    st = "focus"
                elif prev_state in ("interrupted", "distracted"):
                    same = last_focus_ctx == (s["app"], s["domain"])
                    st = "focus" if (same or cfg.is_focus_context(s["app"], s["title"], s["url"])) else prev_state
                else:
                    st = "focus"
            else:
                st = STATE_OF[label]
            s["activity"] = r["e_activity"]
            s.update(state=st, label=label, switch_id=r["group_id"] or r["id"],
                     uncertain=0 if transit else int(r["uncertain"] or 0),
                     source="transit" if transit else ("review" if r["r_label"] else ("model" if r["e_label"] else "pending")),
                     probs=None if r["e_label"] is None else {
                         "continuation": r["p_continuation"], "interruption": r["p_interruption"],
                         "distraction": r["p_distraction"], "task_change": r["p_task_change"]})
        if s["state"] == "focus":
            last_focus_ctx = (s["app"], s["domain"])
        prev_state = s["state"]
    return segs


MAX_GAP = 60.0  # seconds of unrecorded time that breaks a run (collector down, sleep, ...)


def _runs(segs: list[dict], state: str, new_run_on_label: str | None = None) -> list[dict]:
    runs, cur = [], None
    for s in segs:
        if s["state"] == state and s["duration"] > 0:
            contiguous = cur is not None and s["clip_start"] - cur["end"] <= MAX_GAP
            if contiguous and not (new_run_on_label and s["label"] == new_run_on_label):
                cur["end"] = s["clip_end"]
                cur["segments"].append(s)
                continue
            cur = {"start": s["clip_start"], "end": s["clip_end"], "segments": [s]}
            runs.append(cur)
        else:
            cur = None
    for r in runs:
        r["duration"] = r["end"] - r["start"]
    return runs


def span_subtype(sp: dict, cfg) -> str:
    """Classify a focus span into one of five kinds (Rules doc + user categories):
    creative_work, focused_task, reading, planning, other_focused. Prefers Claude's per-switch
    `activity` tag; falls back to app/Toggl heuristics."""
    from collections import Counter
    fsegs = [s for r in sp["runs"] if r["cat"] == "focus" for s in r["segments"]]
    tot = sum(s["duration"] for s in fsegs) or 1.0

    # 1) Content signal: dominant Claude activity, mapped to a subtype.
    ACT2SUB = {"writing": "creative_work", "coding": "creative_work", "task_execution": "focused_task",
               "reading": "reading", "planning": "planning", "browsing": "other_focused", "other": "other_focused"}
    grp = Counter()
    for s in fsegs:
        a = s.get("activity")
        if a:
            grp[ACT2SUB.get(a, "other_focused")] += s["duration"]
    if sum(grp.values()) / tot >= 0.4:
        return grp.most_common(1)[0][0]

    # 2) Toggl tags/description hint.
    tagtext = " ".join(f"{s.get('toggl_tags','')} {s.get('toggl_desc','')}" for s in fsegs).lower()
    if any(k in tagtext for k in ("writ", "essay", "draft", "coding", "software", "creativ", "video")):
        return "creative_work"
    if any(k in tagtext for k in ("focused work", "task", "admin", "process", "email", "errand", "logistic")):
        return "focused_task"
    if any(k in tagtext for k in ("plan", "reflect", "goal", "journal", "metacog", "prioriti", "choosing")):
        return "planning"
    if any(k in tagtext for k in ("read", "study")):
        return "reading"

    # 3) App heuristic: authoring surface => creative; else other focused work.
    creative = sum(s["duration"] for s in fsegs if cfg.is_authoring(s["app"], s["title"], s["url"]))
    if creative / tot >= cfg.creative_frac:
        return "creative_work"
    return "other_focused"


def _detour_label(run: dict) -> str:
    """A merged detour's dominant kind, by time (interrupted vs distracted)."""
    from collections import Counter
    c = Counter()
    for s in run["segments"]:
        c[s["state"]] += s["duration"]
    return c.most_common(1)[0][0] if c else "interrupted"


def _runs3(segs: list[dict]) -> list[dict]:
    """Runs of three categories: 'focus', 'detour' (interrupted+distracted MERGED - stacked
    interruptions/distractions are one event), and 'idle'. Split on recording gaps and on a
    deliberate focus start."""
    runs, cur = [], None
    for s in segs:
        if s["duration"] <= 0:
            continue
        cat = "focus" if s["state"] == "focus" else ("idle" if s["state"] == "idle" else "detour")
        gap = cur is not None and s["clip_start"] - cur["end"] > MAX_GAP
        new = (cur is None or cur["cat"] != cat or gap or (cat == "focus" and s["label"] == "task_change"))
        if new:
            cur = {"cat": cat, "start": s["clip_start"], "end": s["clip_end"], "segments": [s],
                   "focus_start": cat == "focus" and s["label"] == "task_change", "gap_before": gap}
            runs.append(cur)
        else:
            cur["end"] = s["clip_end"]
            cur["segments"].append(s)
    for r in runs:
        r["duration"] = r["end"] - r["start"]
        r["toggl_id"] = next((sg.get("toggl_id") for sg in r["segments"] if sg.get("toggl_id")), None)
        if r["cat"] == "detour":
            r["state"] = _detour_label(r)
    return runs


def spans_and_events(segs: list[dict], min_focus_min: float | None = None, break_min: float | None = None) -> dict:
    """Unified model (Rules and heuristics doc). A focus span holds attention on one task/project.
    ONE threshold governs every diversion: any contiguous time AWAY from the span's task - whether a
    distraction, an interruption, or a hop to different work - ends the span only if it reaches
    `break_min` (retroactively, at the moment you left the task). A diversion shorter than that is a
    contained attention-interruption event and the span continues. Going idle (>= idle threshold) or a
    recording gap also ends the span."""
    if min_focus_min is None or break_min is None:
        from .config import load_config
        cfg = load_config()
        min_focus_min = min_focus_min if min_focus_min is not None else cfg.min_focus_span_min
        break_min = break_min if break_min is not None else cfg.break_min
    runs = _runs3(segs)
    break_s = break_min * 60

    def _same_task(sp, r):
        """Is focus run r the same task as span sp? (returning to the exact window/site, a
        continuation-type focus, or a task-change the model is not confident about -> same task.)"""
        focus_runs = [x for x in sp["runs"] if x["cat"] == "focus"]
        if not focus_runs:
            return False
        first = r["segments"][0]
        last = focus_runs[-1]["segments"][-1]
        if (first["app"], first["domain"]) == (last["app"], last["domain"]):
            return True
        if not r.get("focus_start"):
            return True  # continuation-type focus = same task
        if first.get("source") == "review":
            return False  # a human-confirmed task change is authoritative
        return (first.get("probs") or {}).get("task_change", 0.0) < 0.7

    def _finalize(sp, ended_by):
        focus_runs = [r for r in sp["runs"] if r["cat"] == "focus"]
        last_focus = focus_runs[-1]
        sp["runs"] = sp["runs"][: sp["runs"].index(last_focus) + 1]
        sp["end"] = last_focus["end"]
        sp["detours"] = [r for r in sp["runs"] if r["cat"] == "detour"]
        sp["focus_min"] = sum(r["duration"] for r in sp["runs"] if r["cat"] == "focus") / 60
        sp["detour_min"] = sum(r["duration"] for r in sp["detours"]) / 60
        sp["duration"] = sp["end"] - sp["start"]
        sp["ended_by"] = ended_by
        sp["segments"] = [seg for r in sp["runs"] for seg in r["segments"]]
        return sp

    def _assemble(rs):
        out, n, i = [], len(rs), 0
        while i < n and rs[i]["cat"] != "focus":
            i += 1
        if i >= n:
            return out
        cur = {"start": rs[i]["start"], "runs": [rs[i]]}
        i += 1
        away = []  # runs since we last left cur's task (detours and/or different-task focus)

        def away_s():
            return sum(r["duration"] for r in away)

        def reason():
            if any(r["cat"] == "focus" for r in away):
                return "task change"
            dd = sum(r["duration"] for r in away if r.get("state") == "distracted")
            ii = sum(r["duration"] for r in away if r.get("state") == "interrupted")
            return "distracted" if dd > ii else "interrupted"

        while i < n:
            r = rs[i]
            if r.get("gap_before"):
                out.append(_finalize(cur, reason() if away else "gap (not recorded)"))
                return out + _assemble((away + rs[i:]) if away else rs[i:])
            if r["cat"] == "idle":
                out.append(_finalize(cur, reason() if away else "away"))
                return out + _assemble((away + rs[i + 1:]) if away else rs[i + 1:])
            if r["cat"] == "focus" and _same_task(cur, r):
                if away:
                    if away_s() < break_s:  # brief diversion -> contained interruption(s), span continues
                        for ar in away:
                            cur["runs"].append(dict(ar, cat="detour", state="interrupted") if ar["cat"] == "focus" else ar)
                        away = []
                    else:                    # away >= 5 min -> span ended when we left the task
                        out.append(_finalize(cur, reason()))
                        return out + _assemble(rs[i - len(away):])
                cur["runs"].append(r)
            else:                            # away from the task: a detour or a different-task focus run
                away.append(r)
                if away_s() >= break_s:
                    out.append(_finalize(cur, reason()))
                    return out + _assemble(rs[i - len(away) + 1:])
            i += 1
        if away:
            out.append(_finalize(cur, reason()))
            return out + _assemble(away)
        last = cur["runs"][-1]
        ongoing = last["end"] >= time.time() - MAX_GAP and last["cat"] == "focus"
        out.append(_finalize(cur, "ongoing" if ongoing else "gap (not recorded)"))
        return out

    spans = _assemble(runs)

    qualifying = [sp for sp in spans if sp["focus_min"] >= min_focus_min]
    short = [sp for sp in spans if sp["focus_min"] < min_focus_min]
    from .config import load_config as _lc
    _cfg = _lc()
    for sp in qualifying:
        sp["fully_absorbed"] = len(sp["detours"]) == 0
        sp["subtype"] = span_subtype(sp, _cfg)
    interruptions, distractions = [], []
    for sp in qualifying:
        for d in sp["detours"]:
            d["is_event"] = True; d["inside_span"] = True
            (interruptions if d["state"] == "interrupted" else distractions).append(d)
    span_ends = {round(sp["end"]) for sp in qualifying}
    for r in runs:
        if r["cat"] == "detour" and not r.get("inside_span"):
            r["is_event"] = any(abs(r["start"] - e) <= MAX_GAP for e in span_ends)
            r["inside_span"] = False
            (interruptions if r["state"] == "interrupted" else distractions).append(r)
    interruptions.sort(key=lambda r: r["start"]); distractions.sort(key=lambda r: r["start"])
    return {"focus_spans": qualifying, "short_focus": short,
            "interruptions": interruptions, "distractions": distractions}


def select_for_review(conn, t0: float, t1: float, n: int) -> list[int]:
    """The top-N switches worth reviewing in [t0,t1): ambiguous, focus-leaving, still unreviewed,
    ranked by stakes x uncertainty (importance x (1 - top probability)). Recomputed each call, so
    a 2 pm uncertain switch is never crowded out by an earlier marginal one."""
    rows = conn.execute("""
        SELECT s.id,
               e.importance * (1.0 - max(e.p_continuation, e.p_interruption, e.p_distraction, e.p_task_change)) AS score
        FROM switches s JOIN ensemble e ON e.switch_id = s.id
        LEFT JOIN reviews r ON r.switch_id = s.id
        WHERE s.ts >= ? AND s.ts < ? AND e.uncertain = 1 AND r.switch_id IS NULL AND s.status != 'transit'
        ORDER BY score DESC, s.ts DESC LIMIT ?""", (t0, t1, n)).fetchall()
    return [r[0] for r in rows]


def short_breaks(conn, t0: float, t1: float) -> list[dict]:
    return [{"start": max(r["start"], t0), "end": min(r["end"], t1)} for r in
            conn.execute("SELECT start, end FROM inactivity WHERE start < ? AND end > ? ORDER BY start", (t1, t0))]


def daily_metrics(conn, day: dt.date) -> dict:
    t0, t1 = day_bounds(day)
    segs = labelled_segments(conn, t0, t1)
    se = spans_and_events(segs)
    focus = se["focus_spans"]
    breaks = short_breaks(conn, t0, t1)
    return {
        "short_breaks": len(breaks),
        "short_break_min": sum(b["end"] - b["start"] for b in breaks) / 60,
        "day": day.isoformat(),
        "longest_focus_min": max((s["duration"] for s in focus), default=0) / 60,
        "longest_absorbed_min": max((s["duration"] for s in focus if s.get("fully_absorbed")), default=0) / 60,
        "fully_absorbed_spans": sum(1 for s in focus if s.get("fully_absorbed")),
        "total_focus_min": sum(s["focus_min"] for s in focus),
        "detours_in_spans": sum(len(s["detours"]) for s in focus),
        "detour_min_in_spans": sum(s["detour_min"] for s in focus),
        # An attention-interruption event = any diversion inside a focus span (self-distraction included).
        "interruption_count": (sum(1 for r in se["interruptions"] if r.get("inside_span"))
                               + sum(1 for r in se["distractions"] if r.get("inside_span"))),
        "distraction_count": sum(1 for r in se["distractions"] if r.get("inside_span")),
        "distracted_min": sum(d["duration"] for d in se["distractions"]) / 60,
        "interrupted_min": sum(d["duration"] for d in se["interruptions"]) / 60,
        "active_min": sum(s["duration"] for s in segs if s["state"] != "idle") / 60,
        "switches": sum(1 for s in segs if s["switch_id"]),
        "uncertain": sum(1 for s in segs if s["uncertain"]),
        "reviewed": sum(1 for s in segs if s["source"] == "review"),
        "spend_usd": conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM predictions WHERE created>=? AND created<?",
                                  (t0, t1)).fetchone()[0],
    }


def metrics_range(conn, days: int) -> list[dict]:
    today = logical_today()
    return [daily_metrics(conn, today - dt.timedelta(days=i)) for i in range(days - 1, -1, -1)]


def accuracy_over_time(conn, days: int) -> list[dict]:
    """Per day (by review date): accuracy of each model + ensemble vs. human labels."""
    t0, _ = day_bounds(logical_today() - dt.timedelta(days=days - 1))
    rows = conn.execute("""
        SELECT r.created, r.label, p.model,
               p.p_continuation, p.p_interruption, p.p_distraction, p.p_task_change
        FROM reviews r JOIN predictions p ON p.switch_id=r.switch_id
        WHERE r.created >= ? AND p.created < r.created""", (t0,)).fetchall()
    rows += conn.execute("""
        SELECT r.created, r.label, 'ensemble' AS model,
               e.p_continuation, e.p_interruption, e.p_distraction, e.p_task_change
        FROM reviews r JOIN ensemble e ON e.switch_id=r.switch_id
        WHERE r.created >= ? AND e.created < r.created""", (t0,)).fetchall()
    per: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        day = logical_date(r["created"]).isoformat()
        probs = {"continuation": r["p_continuation"], "interruption": r["p_interruption"],
                 "distraction": r["p_distraction"], "task_change": r["p_task_change"]}
        per.setdefault((day, r["model"]), []).append(int(max(probs, key=probs.get) == r["label"]))
    out = []
    for (day, model), hits in sorted(per.items()):
        out.append({"day": day, "model": model, "n": len(hits), "accuracy": sum(hits) / len(hits)})
    return out


def focus_context(conn, from_seg: dict | None) -> tuple[str, float]:
    """State the person was in just before a switch and, if focused, how long that run had lasted (min).

    The switch matters most when it leaves an ongoing focus span - these are the cases the whole
    system prioritises (Claude budget, review queue, model scoring).
    """
    if not from_seg:
        return "focus", 0.0
    end = from_seg["end"] or time.time()
    segs = labelled_segments(conn, end - 4 * 3600, end)
    segs = [s for s in segs if s["clip_start"] < end]
    if not segs:
        return "focus", 0.0
    state = segs[-1]["state"]
    if state != "focus":
        return state, 0.0
    se = spans_and_events(segs, min_focus_min=0)
    for sp in reversed(se["focus_spans"]):
        if sp["end"] >= end - 1 and sp["start"] <= end:
            return "focus", sp["focus_min"]
    return "focus", 0.0

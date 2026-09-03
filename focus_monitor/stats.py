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
               e.p_continuation, e.p_interruption, e.p_distraction, e.p_task_change, e.activity AS e_activity,
               tc.thread AS thread
        FROM switches s
        LEFT JOIN ensemble e ON e.switch_id = COALESCE(s.group_id, s.id)
        LEFT JOIN reviews r ON r.switch_id = COALESCE(s.group_id, s.id)
        LEFT JOIN thread_calls tc ON tc.switch_id = COALESCE(s.group_id, s.id)
        WHERE s.to_segment IN ({q})""", ids).fetchall()
    ev = {r["segment_id"]: r for r in conn.execute(
        f"SELECT * FROM seg_evals WHERE segment_id IN ({q})", ids)}
    rv = {r["segment_id"]: r for r in conn.execute(
        f"SELECT * FROM seg_reviews WHERE segment_id IN ({q})", ids)}

    def verdict(seg_id):
        """(content, switch, kind, user_edited, uncertain) - the user's edit overrides the
        evaluator field-by-field; None when neither exists."""
        e, u = ev.get(seg_id), rv.get(seg_id)
        if e is None and u is None:
            return None
        g = lambda k: (u[k] if u is not None and u[k] else (e[k] if e is not None else None))
        unc = 0 if u is not None else int((e["uncertain"] if e is not None else 0) or 0)
        import json as _json
        pr = None
        if e is not None and e["probs"]:
            try:
                pr = _json.loads(e["probs"])
            except ValueError:
                pr = None
        u_sw = u is not None and bool(u["switch_label"])
        u_kind = u is not None and bool(u["interruption_kind"])
        ante = g("antecedent_id")
        u_ante = u is not None and bool(u["antecedent_id"])
        return g("content"), g("switch_label"), g("interruption_kind"), u is not None, unc, pr, u_sw, u_kind, ante, u_ante
    tg = conn.execute("SELECT id, start, stop, tags, description FROM toggl_entries WHERE start < ? AND COALESCE(stop, ?) > ? ORDER BY start",
                      (t1, now, t0)).fetchall() if _has_toggl(conn) else []
    inact = conn.execute("SELECT start, end FROM inactivity WHERE start < ? AND end > ?", (t1, t0)).fetchall()
    def toggl_at(ts):
        for e in tg:
            if e["start"] <= ts and (e["stop"] is None or e["stop"] > ts):
                return e
        return None
    def toggl_near(ts, slack=120.0):
        """The entry running at ts, tolerating brief gaps in a series of entries."""
        for e in tg:
            if e["start"] - slack <= ts and (e["stop"] is None or e["stop"] + slack > ts):
                return e
        return None
    by_to = {r["to_segment"]: r for r in sw}
    prev_state = "focus"
    last_focus_ctx = None  # (app, domain) of the most recent focused segment
    last_detour = None  # ((app, domain), state) of the most recent detour segment: an ongoing
    # distraction/interruption context. A mere "continuation" back into it RESUMES the detour -
    # a brief flip to notes must not launder a video session into focus. Cleared by a deliberate
    # focus start (task_change) in that same context.
    for s in segs:
        s["end_eff"] = s["end"] or now
        s["clip_start"], s["clip_end"] = max(s["start"], t0), min(s["end_eff"], t1)
        s["duration"] = max(0.0, s["clip_end"] - s["clip_start"])
        _te = toggl_at(s["start"])
        s["toggl_id"] = _te["id"] if _te else None
        s["toggl_tags"] = (_te["tags"] or "") if _te else ""
        s["toggl_desc"] = (_te["description"] or "") if _te else ""
        s["inact_s"] = sum(max(0.0, min(i["end"], s["clip_end"]) - max(i["start"], s["clip_start"])) for i in inact)
        _ta, _tb = toggl_near(s["start"]), toggl_near(s["end_eff"])
        s["tg_near_a"] = _ta["id"] if _ta else None
        s["tg_near_b"] = _tb["id"] if _tb else None
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
        v = verdict(s["id"])
        if v is not None:
            # The evaluator's (or the user's edited) per-segment judgment drives the state.
            content, vsw, vkind, edited, v_unc, v_probs, u_sw, u_kind, v_ante, u_ante = v
            # Thread pointer: meaningful only for continuation/return (an interruption IS
            # the birth of a new thread). Link confidence: 1.0 for a user-pinned edge, else
            # the evaluator's conditional p for its top candidate (0.85 for pre-prob rows).
            if vsw in (None, "continuation", "return") and v_ante is not None:
                s["antecedent_id"] = v_ante
                s["antecedent_p"] = (1.0 if u_ante else
                                     (v_probs or {}).get("antecedent", {}).get(str(v_ante), 0.85))
            else:
                s["antecedent_id"] = None
            vsw = vsw or "continuation"
            if vsw == "continuation":
                st = prev_state if prev_state != "idle" else "focus"
            elif vsw == "return":
                st = "focus"
            else:  # interruption: what kind of shift?
                st = {"self_distraction": "distracted", "focus_start": "focus",
                      "detour": "interrupted"}.get(vkind or "detour", "interrupted")
            if content == "passive":
                st = "distracted"  # Rules doc: passive consumption never counts toward focus
            label = ("continuation" if vsw in ("continuation", "return") else
                     {"self_distraction": "distraction", "focus_start": "task_change",
                      "detour": "interruption"}.get(vkind or "detour", "interruption"))
            s["content"] = content
            # Split confidence: 1.0/0.0 only for the fields the USER actually set. A
            # content-only edit must not harden the model's probabilistic split, and a
            # switch-only "separate thread" ruling leaves sustainment (the kind) to the
            # app's derived half - p(interruption)=1 by ruling, so split = p(kind=focus_start).
            if u_sw:
                s["split_user"] = True
                if vsw != "interruption":
                    s["eval_split_p"] = 0.0
                elif u_kind:
                    s["eval_split_p"] = 1.0 if vkind == "focus_start" else 0.0
                else:
                    kp = (v_probs or {}).get("kind", {})
                    s["eval_split_p"] = (kp.get("focus_start", 0.0) if kp else
                                         (1.0 if vkind == "focus_start" else
                                          0.0 if vkind else 0.5))
            elif v_probs:
                s["eval_split_p"] = (v_probs.get("switch", {}).get("interruption", 0.0)
                                     * v_probs.get("kind", {}).get("focus_start", 0.0))
            if r is not None:
                s["activity"] = r["e_activity"]
            s.update(state=st, label=label, switch_id=(r["group_id"] or r["id"]) if r is not None else None,
                     uncertain=v_unc, source="review" if edited else "eval")
        elif r is None:  # first segment, or a return to the same window after a neutral one
            st = prev_state if prev_state != "idle" else "focus"
            if (st == "focus" and last_detour and last_detour[0] == (s["app"], s["domain"])
                    and last_focus_ctx != (s["app"], s["domain"])
                    and not cfg.is_focus_context(s["app"], s["title"], s["url"])):
                st = last_detour[1]  # returning (even after idle) to an ongoing detour context
            s.update(state=st, label="continuation", switch_id=None, uncertain=0, source="start")
        else:
            label = r["r_label"] or r["e_label"] or "continuation"
            transit = r["status"] == "transit"
            if label == "continuation":
                thread = None if r["r_label"] else r["thread"]  # a human review outranks the resolver
                if thread == "work":
                    st = "focus"  # the resolver's judgment: this continues the work thread
                elif thread in ("interruption", "distraction"):
                    st = "interrupted" if thread == "interruption" else "distracted"
                elif prev_state in ("interrupted", "distracted"):
                    same = last_focus_ctx == (s["app"], s["domain"])
                    st = "focus" if (same or cfg.is_focus_context(s["app"], s["title"], s["url"])) else prev_state
                else:
                    st = "focus"
                if (thread is None and st == "focus" and last_detour and last_detour[0] == (s["app"], s["domain"])
                        and last_focus_ctx != (s["app"], s["domain"])
                        and not cfg.is_focus_context(s["app"], s["title"], s["url"])):
                    st = last_detour[1]  # fallback only: back into the recent detour context
            else:
                st = STATE_OF[label]
            s["activity"] = r["e_activity"]
            s.update(state=st, label=label, switch_id=r["group_id"] or r["id"],
                     uncertain=0 if transit else int(r["uncertain"] or 0),
                     source="transit" if transit else ("review" if r["r_label"] else ("model" if r["e_label"] else "pending")),
                     probs=None if r["e_label"] is None else {
                         "continuation": r["p_continuation"], "interruption": r["p_interruption"],
                         "distraction": r["p_distraction"], "task_change": r["p_task_change"]})
        # Categorical rule (Rules doc + user ruling): passive-consumption domains (video
        # watching) are never focused work, however the switch was labelled - continuation
        # inheritance must not launder a video session into focus. Only a human review of
        # the inbound switch can override.
        if (s["state"] == "focus" and s.get("source") != "review"
                and any((s.get("domain") or "") == d or (s.get("domain") or "").endswith("." + d)
                        for d in cfg.passive_domains)):
            s["state"] = "distracted"
        if s["state"] == "focus":
            last_focus_ctx = (s["app"], s["domain"])
            if s.get("label") == "task_change" and last_detour and last_detour[0] == (s["app"], s["domain"]):
                last_detour = None  # a deliberate focus start here ends the detour context
        elif s["state"] in ("interrupted", "distracted"):
            last_detour = ((s["app"], s["domain"]), s["state"])
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


def span_subtype(sp: dict, cfg) -> str | None:
    """Classify a focus span into one of five kinds (Rules doc + user categories):
    creative_work, focused_task, reading, planning, other_focused. Prefers Claude's per-switch
    `activity` tag; falls back to app/Toggl heuristics.

    Hard rules (Rules doc) - failing either gate means the span is NOT focused work of any
    kind: returns None (setting sp["disq_reason"]) and the caller drops it from the focus
    spans, so its time counts as unfocused.
    - focused_task requires a running Toggl entry (>= task_toggl_frac of focus time).
    - reading requires interleaved note-taking (notes surfaces >= reading_notes_frac of
      focus time); passive essay/post-hopping without notes is unfocused."""
    from collections import Counter
    fsegs = [s for r in sp["runs"] if r["cat"] == "focus" for s in r["segments"]]
    tot = sum(s["duration"] for s in fsegs) or 1.0

    toggl_s = sum(s["duration"] for s in fsegs if s.get("toggl_id"))
    task_ok = toggl_s / tot >= cfg.task_toggl_frac

    def _is_notes(s):
        dom = s.get("domain") or ""
        return (s["app"] in cfg.notes_apps or cfg.is_authoring(s["app"], s.get("title") or "", s.get("url") or "")
                or any(dom == d or dom.endswith("." + d) for d in cfg.focus_domains))
    notes_s = sum(s["duration"] for s in fsegs if _is_notes(s))
    reading_ok = notes_s / tot >= cfg.reading_notes_frac

    # 1) Content signal: the evaluator's per-segment content label (preferred), falling back
    #    to the legacy per-switch activity tag.
    ACT2SUB = {"writing": "creative_work", "coding": "creative_work", "task_execution": "focused_task",
               "reading": "reading", "planning": "planning", "browsing": "other_focused", "other": "other_focused"}
    CONTENT2SUB = {"creative": "creative_work", "task": "focused_task", "reading": "reading",
                   "planning": "planning", "other": "other_focused", "meeting": "meeting"}
    grp = Counter()
    for s in fsegs:
        c = s.get("content")
        if c in CONTENT2SUB:
            grp[CONTENT2SUB[c]] += s["duration"]
            continue
        if c in ("passive", "idle", "transition"):
            continue
        a = s.get("activity")
        if a:
            grp[ACT2SUB.get(a, "other_focused")] += s["duration"]
    if sum(grp.values()) / tot >= 0.4:
        top = grp.most_common(1)[0][0]
        if top == "focused_task" and not task_ok:
            sp["disq_reason"] = "task activity without a Toggl entry"
            return None  # task churn without Toggl: not focused work at all
        if top == "reading" and not reading_ok:
            sp["disq_reason"] = "reading without note-taking"
            return None  # passive reading, no notes back-and-forth: not focused work
        return top

    # 2) Toggl tags/description hint.
    tagtext = " ".join(f"{s.get('toggl_tags','')} {s.get('toggl_desc','')}" for s in fsegs).lower()
    if any(k in tagtext for k in ("writ", "essay", "draft", "coding", "software", "creativ", "video")):
        return "creative_work"
    if task_ok and any(k in tagtext for k in ("focused work", "task", "admin", "process", "email", "errand", "logistic")):
        return "focused_task"
    if any(k in tagtext for k in ("plan", "reflect", "goal", "journal", "metacog", "prioriti", "choosing")):
        return "planning"
    if any(k in tagtext for k in ("read", "study")):
        if not reading_ok:
            sp["disq_reason"] = "reading without note-taking"
            return None
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


SUBTYPES = ("creative_work", "focused_task", "reading", "planning", "other_focused", "meeting")


def _span_review_for(rows, sp):
    """The user's verdict on essentially THIS span. The overlap must cover >= 80% of BOTH the
    span and the reviewed range: a verdict is about the span as it was shown, and must not
    chase a substantially re-bounded span - e.g. "not a focus span" on a badly-assembled 41
    minutes must not demote the genuine 30-minute span later corrected out of its middle."""
    best, best_ov = None, 0.0
    for r in rows:
        ov = min(sp["end"], r["end"]) - max(sp["start"], r["start"])
        if ov > best_ov:
            best, best_ov = r, ov
    if best is None:
        return None
    ok = (best_ov >= 0.8 * (sp["end"] - sp["start"])
          and best_ov >= 0.8 * (best["end"] - best["start"]))
    return best if ok else None


def _apply_span_reviews(conn, qualifying: list[dict], disqualified: list[dict]) -> tuple[list, list]:
    """User verdicts are authoritative: 'not_focus' demotes a span, 'focus' confirms it (and can
    resurrect a gate-disqualified one, with the user's subtype). Unreviewed spans are marked
    confirmed=False - they stand provisionally until reviewed."""
    rows = conn.execute("SELECT * FROM span_reviews").fetchall()
    keep_q, keep_d = [], []
    for sp in qualifying:
        r = _span_review_for(rows, sp)
        if r is None:
            sp["confirmed"] = False
            keep_q.append(sp)
        elif r["verdict"] == "not_focus":
            sp["confirmed"], sp["subtype"] = True, None
            sp["disq_reason"] = "you reviewed this: not a focus span"
            keep_d.append(sp)
        else:
            sp["confirmed"] = True
            if r["subtype"] in SUBTYPES:
                sp["subtype"] = r["subtype"]
            sp["fully_absorbed"] = len(sp["detours"]) == 0 and sp["subtype"] not in ("focused_task", "meeting")
            keep_q.append(sp)
    for sp in disqualified:
        r = _span_review_for(rows, sp)
        if r is not None and r["verdict"] == "focus":
            sp["confirmed"] = True
            sp["subtype"] = r["subtype"] if r["subtype"] in SUBTYPES else "other_focused"
            sp.pop("disq_reason", None)
            sp["fully_absorbed"] = len(sp["detours"]) == 0 and sp["subtype"] not in ("focused_task", "meeting")
            keep_q.append(sp)
        else:
            sp["confirmed"] = r is not None
            keep_d.append(sp)
    keep_q.sort(key=lambda s: s["start"])
    return keep_q, keep_d


def spans_and_events(segs: list[dict], min_focus_min: float | None = None, break_min: float | None = None,
                     conn=None) -> dict:
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
    # Spans are LAZILY evaluated, like segments (user rule 2026-09-01): never assemble a span
    # over not-yet-evaluated segments - wait for their verdicts instead of guessing and later
    # shrinking the span. Truncate assembly at the evaluation frontier: the last segment with a
    # verdict (eval or user review). Windows with no verdicts at all (legacy history) keep the
    # legacy behavior.
    frontier_ts = None
    last_v = None
    for i, s in enumerate(segs):
        if s.get("source") in ("eval", "review"):
            last_v = i
    if last_v is not None and last_v < len(segs) - 1:
        tail = segs[last_v + 1:]
        if any(s.get("source") not in ("eval", "review") and s["state"] != "idle"
               and not s.get("neutral") and s["duration"] > 0 for s in tail):
            frontier_ts = segs[last_v]["clip_end"]
            segs = segs[:last_v + 1]
    runs = _runs3(segs)
    break_s = break_min * 60
    ante_of = {s["id"]: (s.get("antecedent_id"), s.get("antecedent_p", 0.85)) for s in segs}

    def _same_task(sp, r, away=()):
        """Is focus run r the same task as span sp? (returning to the exact window/site, a
        continuation-type focus, or a task-change the model is not confident about -> same task.)"""
        focus_runs = [x for x in sp["runs"] if x["cat"] == "focus"]
        if not focus_runs:
            return False
        first = r["segments"][0]
        last_run = focus_runs[-1]
        last = last_run["segments"][-1]
        if (first["app"], first["domain"]) == (last["app"], last["domain"]):
            return True
        # Stated thread pointer first: where the evaluator (or the user) named an antecedent,
        # following the chain answers the membership question directly - no guessing. Link
        # confidences MULTIPLY along the path (the chain only holds if every link does); a
        # chain that lands while still confident decides, one that drops below 0.5 falls
        # through to the heuristics below rather than overriding them with a shaky pointer.
        sp_ids = {sg["id"] for run in sp["runs"] if run["cat"] == "focus" for sg in run["segments"]}
        away_ids = {sg["id"] for run in away for sg in run["segments"]}
        a, hops = first.get("antecedent_id"), 0
        conf = first.get("antecedent_p", 0.85) if a is not None else 0.0
        while a is not None and hops < 20 and conf >= 0.5:
            if a in sp_ids:
                return True
            if a in away_ids:
                return False
            a, p = ante_of.get(a, (None, 0.0))
            conf *= p
            hops += 1
        if not r.get("focus_start"):
            # Continuation/return judgments are CHAINED - they continue whatever came
            # immediately before, which is not the span once `away` holds a focus run already
            # refused as a different task (postmortem 2026-09-02: a 19-second detour split a
            # new task in two, and the second half - labeled a mere continuation - was glued
            # back onto the old span, burying a correctly judged boundary and making the
            # user's own split ruling ineffective). With a rival thread in away, a
            # non-focus_start run rejoins the span only by returning to its surfaces.
            away_focus = [x for x in away if x["cat"] == "focus"]
            if not away_focus:
                return True  # continuation-type focus with no rival thread = the span continuing
            span_ctxs = {(sg["app"], sg["domain"]) for sg in last_run["segments"]}
            if (first["app"], first["domain"]) in span_ctxs:
                return True   # back on the span's own surfaces
            return False      # continues the away thread (chain semantics)
        if first.get("eval_split_p") is not None:
            # The judgment belongs to the model (and above it, the user). Whether task-to-task
            # switching under a running Toggl entry is one focused-task block is the
            # EVALUATOR's call (its prompt carries the rule: Toggl is necessary, never
            # sufficient). The stricter threshold applies only when the USER set the switch -
            # a content-only edit leaves the split a model judgment at the model threshold.
            thr = 0.5 if first.get("split_user") else 0.25
            return first["eval_split_p"] < thr
        # ---- legacy fallback only (segments never judged by the evaluator) ----
        # Toggl glue is TASK-SCOPED: the Rules doc's stitching rule applies to focused-task
        # blocks only, and a running entry is a necessary condition, not sufficient.
        last_tg = last.get("tg_near_b") or last.get("toggl_id")
        new_tg_running = first.get("toggl_id")  # entry must be RUNNING at the new segment's
        new_tg = first.get("tg_near_a") or first.get("toggl_id")  # start; a stop ends the intent
        task_ctx = ("task_execution" in (last.get("activity"), first.get("activity"))
                    or "task" in (last.get("content"), first.get("content")))
        if last_tg and new_tg_running and last_tg == new_tg_running and task_ctx:
            return True  # task block under one running entry
        if first.get("source") == "review":
            return False  # a human-confirmed task change is authoritative
        if last_tg and new_tg and task_ctx:
            return True  # a series of Toggl entries over task churn stays one block
        if last.get("activity") == "reading" and first.get("activity") in ("writing", "coding"):
            return True  # reading flowing into creative work stays one span (Rules doc)
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
            if r["cat"] == "focus" and _same_task(cur, r, away):
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
    if frontier_ts is not None:
        for sp in spans:
            if abs(sp["end"] - frontier_ts) <= MAX_GAP and sp["ended_by"] in ("ongoing", "gap (not recorded)"):
                sp["ended_by"] = "awaiting evaluation"

    qualifying = [sp for sp in spans if sp["focus_min"] >= min_focus_min]
    short = [sp for sp in spans if sp["focus_min"] < min_focus_min]
    from .config import load_config as _lc
    _cfg = _lc()
    for sp in qualifying:
        sp["subtype"] = span_subtype(sp, _cfg)
        # Fully-absorbed = zero interruptions AND a single object of attention - a focused-task
        # span is inherently task-switching, so it never counts (Rules doc).
        sp["fully_absorbed"] = len(sp["detours"]) == 0 and sp["subtype"] not in ("focused_task", "meeting")
        # The doc defines fully-absorbed as "a LENGTH of focus span with 0 interruption
        # events" - a stretch, not only a whole span. Longest unbroken focus stretch:
        best = cur = 0.0
        for r_ in sp["runs"]:
            if r_.get("gap_before"):
                cur = 0.0
            if r_["cat"] == "focus":
                cur += r_["duration"]
                best = max(best, cur)
            elif r_["cat"] == "detour":
                cur = 0.0
        sp["absorbed_max_min"] = best / 60
        # Reading flowing into creative work is one span (Rules doc) but the timeline colors
        # each part by its own activity when both sides are substantial.
        fs = [s for r in sp["runs"] if r["cat"] == "focus" for s in r["segments"]]
        ft = sum(s["duration"] for s in fs) or 1.0
        rd = sum(s["duration"] for s in fs if s.get("activity") == "reading") / ft
        cr = sum(s["duration"] for s in fs if s.get("activity") in ("writing", "coding")) / ft
        sp["mixed_read_create"] = rd >= 0.2 and cr >= 0.2
    # Spans disqualified by the Toggl gate (task churn, no entry) are unfocused time, not spans.
    disqualified = [sp for sp in qualifying if sp["subtype"] is None]
    qualifying = [sp for sp in qualifying if sp["subtype"] is not None]
    if conn is not None:  # the user's span reviews are authoritative over the gates
        qualifying, disqualified = _apply_span_reviews(conn, qualifying, disqualified)
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
    return {"focus_spans": qualifying, "short_focus": short, "disqualified": disqualified,
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
    ids = [r[0] for r in rows]
    # The chunk evaluator's own uncertainty flags rank ahead of ensemble uncertainty - its
    # judgments are what actually drive the states now.
    ev_rows = conn.execute("""
        SELECT s.id FROM seg_evals v
        JOIN switches s ON s.to_segment = v.segment_id
        LEFT JOIN seg_reviews u ON u.segment_id = v.segment_id
        LEFT JOIN reviews r ON r.switch_id = s.id
        WHERE s.ts >= ? AND s.ts < ? AND v.uncertain = 1 AND u.segment_id IS NULL
              AND r.switch_id IS NULL AND s.status != 'transit'
        ORDER BY s.ts DESC""", (t0, t1)).fetchall()
    out = [r[0] for r in ev_rows] + [i for i in ids if i not in {r[0] for r in ev_rows}]
    return out[:n]


def short_breaks(conn, t0: float, t1: float) -> list[dict]:
    return [{"start": max(r["start"], t0), "end": min(r["end"], t1)} for r in
            conn.execute("SELECT start, end FROM inactivity WHERE start < ? AND end > ? ORDER BY start", (t1, t0))]


def daily_metrics(conn, day: dt.date) -> dict:
    t0, t1 = day_bounds(day)
    segs = labelled_segments(conn, t0, t1)
    se = spans_and_events(segs, conn=conn)
    focus = se["focus_spans"]
    # Meetings are focus, but their own kind: counted separately, never in "total focus".
    work = [s for s in focus if s.get("subtype") != "meeting"]
    meetings = [s for s in focus if s.get("subtype") == "meeting"]
    breaks = short_breaks(conn, t0, t1)
    return {
        "short_breaks": len(breaks),
        "short_break_min": sum(b["end"] - b["start"] for b in breaks) / 60,
        "day": day.isoformat(),
        "longest_focus_min": max((s["duration"] for s in work), default=0) / 60,
        "longest_absorbed_min": max((s.get("absorbed_max_min", 0.0) for s in work
                                     if s.get("subtype") != "focused_task"), default=0.0),
        "fully_absorbed_spans": sum(1 for s in work if s.get("fully_absorbed")),
        "total_focus_min": sum(s["focus_min"] for s in work),
        "meeting_focus_min": sum(s["focus_min"] for s in meetings),
        "detours_in_spans": sum(len(s["detours"]) for s in focus),
        "detour_min_in_spans": sum(s["detour_min"] for s in focus),
        # An attention-interruption event = any diversion inside a focus span (self-distraction included).
        "interruption_count": (sum(1 for r in se["interruptions"] if r.get("inside_span"))
                               + sum(1 for r in se["distractions"] if r.get("inside_span"))),
        "distraction_count": sum(1 for r in se["distractions"] if r.get("inside_span")),
        "distracted_min": sum(d["duration"] for d in se["distractions"]) / 60,
        "interrupted_min": sum(d["duration"] for d in se["interruptions"]) / 60,
        "active_min": sum(s["duration"] for s in segs if s["state"] != "idle") / 60,
        # Rules doc: passive consumption is tracked in its own right (never counts toward focus).
        "passive_min": sum(s["duration"] for s in segs if s.get("content") == "passive") / 60,
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
    se = spans_and_events(segs, min_focus_min=0, conn=conn)
    for sp in reversed(se["focus_spans"]):
        if sp["end"] >= end - 1 and sp["start"] <= end:
            return "focus", sp["focus_min"]
    return "focus", 0.0

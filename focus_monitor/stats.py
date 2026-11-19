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
import json
import time

from . import db

STATE_OF = {"task_change": "focus", "interruption": "interrupted", "distraction": "distracted"}


def _has_toggl(conn) -> bool:
    return bool(conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='toggl_entries'").fetchone())


from .config import day_bounds, logical_date  # day boundary lives in config (default 4am-4am)


def logical_today() -> dt.date:
    return logical_date()


def labelled_segments(conn, t0: float, t1: float, overrides: dict | None = None) -> list[dict]:
    """Segments in [t0,t1) with their resolved judgment and state. `overrides` ({segment_id:
    {content, switch_label, interruption_kind, antecedent_id, hybrid}}) simulates a user review
    of those segments in memory - what the day would look like if an answer were given - without
    touching the database."""
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
    for s in segs:  # provenance: which chat reply (if any) the person's edit came out of
        r0 = rv.get(s["id"])
        s["review_chat_mid"] = r0["chat_message_id"] if r0 is not None and "chat_message_id" in r0.keys() else None
    if overrides:
        blank = {"segment_id": None, "content": None, "switch_label": None, "interruption_kind": None,
                 "antecedent_id": None, "note": None, "created": now, "hybrid": None, "focus_exception": None,
                 "focused": None, "self_distraction": None, "next_task": None}
        for sid, o in overrides.items():
            if sid not in ids:
                continue
            base = dict(rv[sid]) if sid in rv else dict(blank, segment_id=sid)
            for k in ("content", "switch_label", "interruption_kind", "antecedent_id", "hybrid",
                      "focused", "self_distraction", "next_task"):
                if o.get(k) is not None:
                    base[k] = o[k]
            if o.get("switch_label") == "interruption":
                base["antecedent_id"] = None
                if o.get("interruption_kind") is None:
                    base["interruption_kind"] = None
            rv[sid] = base

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
        hyb = (u["hybrid"] if u is not None and u["hybrid"] is not None else (e["hybrid"] if e is not None else 0))
        # The person's explicit "counts as focus anyway" ruling - a review-only field.
        fex = int((u["focus_exception"] if u is not None and "focus_exception" in u.keys() else 0) or 0)

        def g2(k):
            """A 0/1 field where 0 is a real answer: the person's value when they gave one."""
            if u is not None and k in u.keys() and u[k] is not None:
                return int(u[k]), True
            if e is not None and k in e.keys() and e[k] is not None:
                return int(e[k]), False
            return None, False
        foc, u_foc = g2("focused")
        sd, _ = g2("self_distraction")
        nxt, _ = g2("next_task")
        sw_, kind_ = g("switch_label"), g("interruption_kind")
        if sw_ == "interruption" and foc is not None and not u_kind:
            # Under the redesign the older per-shift kind is not a judgment of its own: it
            # follows from the focus field and the self-distraction flag (whoever set them).
            kind_ = "self_distraction" if sd else ("focus_start" if foc else "detour")
        return (g("content"), sw_, kind_, u is not None, unc, pr, u_sw, u_kind, ante, u_ante,
                int(hyb or 0), fex, foc, u_foc, sd, nxt)
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
    sd_live = False  # the thread being followed began with a self-distraction shift
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
            (content, vsw, vkind, edited, v_unc, v_probs, u_sw, u_kind, v_ante, u_ante, v_hyb, v_fex,
             v_foc, u_foc, v_sd, v_nxt) = v
            s["hybrid"] = v_hyb
            s["focus_exception"] = v_fex
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
            # Redesign 2026-11-19 (Rules doc): a segment judged with a focused/unfocused field
            # takes its state from that field. A segment judged before it (or by the local
            # models, or a pass-through/idle one) has no such field and keeps the older rule:
            # the state follows the switch and is inherited through continuations.
            by_field = v_foc is not None and content not in ("transition", "idle")
            if vsw == "interruption":
                sd_live = bool(v_sd) if v_sd is not None else vkind == "self_distraction"
            elif vsw == "return":
                # going back to a distraction is judged afresh (Eli, 2026-11-19); a return to
                # work, and every return judged before that date, carries no flag
                sd_live = bool(v_sd)
            s["switch"] = vsw
            s["redesign"] = v_foc is not None   # judged with the 2026-11-19 fields
            s["focused"] = bool(v_foc) if by_field else None
            s["user_unfocused"] = bool(by_field and u_foc and not v_foc)
            s["self_distraction"] = bool(vsw in ("interruption", "return") and sd_live)
            s["next_task"] = bool(vsw == "interruption" and v_nxt)
            if by_field:
                # "Distracted" is not a judgment about the segment: it marks unfocused time that
                # follows a self-distraction shift, so the event can be counted and coloured.
                st = "focus" if v_foc else ("distracted" if sd_live else "interrupted")
            elif vsw == "continuation":
                st = prev_state if prev_state != "idle" else "focus"
            elif vsw == "return":
                st = "focus"
            else:  # interruption: what kind of shift?
                st = {"self_distraction": "distracted", "focus_start": "focus",
                      "detour": "interrupted"}.get(vkind or "detour", "interrupted")
            if content == "passive":
                # Rules doc: passive consumption never counts toward focus. Whether it was a
                # self-distraction is about intent (a planned video is not one), so under the
                # redesign the colour follows the shift, not the content.
                st = ("distracted" if sd_live else "interrupted") if by_field else "distracted"
            elif content == "correspondence" and st == "focus" and not s.get("toggl_id") and not v_fex:
                # Rules doc: focus spans are independent work, never interactive - chatting or
                # answering mail is a diversion, not focus. The one exception is correspondence
                # processed inside a Toggl-covered task block, which rides along with that span -
                # unless the person explicitly ruled this block an exception (focus_exception).
                st = "interrupted"
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
                     own_switch_id=r["id"] if r is not None else None,  # ungrouped: the switch INTO this segment
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
            s.update(state=st, label=label, switch_id=r["group_id"] or r["id"], own_switch_id=r["id"],
                     uncertain=0 if transit else int(r["uncertain"] or 0),
                     source="transit" if transit else ("review" if r["r_label"] else ("model" if r["e_label"] else "pending")),
                     probs=None if r["e_label"] is None else {
                         "continuation": r["p_continuation"], "interruption": r["p_interruption"],
                         "distraction": r["p_distraction"], "task_change": r["p_task_change"]})
        # Categorical rule (Rules doc + user ruling): passive-consumption domains (video
        # watching) are never focused work, however the switch was labelled - continuation
        # inheritance must not launder a video session into focus. Only a human review of
        # the inbound switch can override. Exception (Eli, 2026-11-19): a video watched in
        # service of the current task is part of the span. A segment judged under the redesign
        # gets its state from its own focused field, not by inheritance, so when the evaluator
        # looked at it and called it focused, non-passive work, that judgment stands.
        if (s["state"] == "focus" and s.get("source") != "review"
                and not (s.get("redesign") and s.get("focused") and s.get("content") != "passive")
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
UNFOCUSED_ASK_FRAC = 0.20          # ask when more than this share of a span was judged unfocused
UNFOCUSED_ASK_STRETCH_S = 600.0    # ... or when one unfocused stretch lasted this long


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


def _opens_creative(sp: dict, cfg) -> bool:
    """True when the span's opening unbroken focused block (before any diversion or unfocused
    stretch) has at least min_unbroken_min of creative content before any other kind of work."""
    need, got = cfg.min_unbroken_min * 60.0, 0.0
    for r in sp["runs"]:
        if r["cat"] != "focus" or r.get("soft"):
            break
        for s in r["segments"]:
            c = s.get("content")
            if c == "creative":
                got += s["duration"]
            elif c not in ("transition", "idle"):
                return got >= need
    return got >= need


def span_subtype(sp: dict, cfg) -> str | None:
    """Classify a focus span into one of its kinds (Rules doc + user categories):
    creative_work, focused_task, reading, planning, other_focused, meeting. Prefers Claude's per-switch
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
                   "planning": "planning", "other": "other_focused", "meeting": "meeting",
                   # focus-state correspondence only exists under a running Toggl entry (see
                   # labelled_segments), where it is part of a task block
                   "correspondence": "focused_task"}
    grp = Counter()
    for s in fsegs:
        c = s.get("content")
        if c in CONTENT2SUB:
            # Correspondence the person ruled an exception (drafting on their own schedule) is
            # creative work for the span's kind - there is no Toggl task block to attach it to.
            grp["creative_work" if (c == "correspondence" and s.get("focus_exception")) else CONTENT2SUB[c]] += s["duration"]
            continue
        if c in ("passive", "idle", "transition"):
            continue
        a = s.get("activity")
        if a:
            grp[ACT2SUB.get(a, "other_focused")] += s["duration"]
    if sum(grp.values()) / tot >= 0.4:
        top = grp.most_common(1)[0][0]
        if top == "focused_task" and not task_ok and _opens_creative(sp, cfg):
            # A span that began as creative work keeps that kind when the same thread turns
            # into quick task-mode work (Eli, 2026-11-19): carefully designing with an agent,
            # then rapid back-and-forth on the same project, needs no Toggl entry. Only the
            # same thread gets here - a shift to another task is a new thread (_same_task),
            # so interleaved other work is time away and ends the span by the usual rules.
            return "creative_work"
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
    """Whether a stretch of time away counts as a self-distraction ('distracted') or a plain
    diversion ('interrupted'). Under the redesign a self-distraction is an EVENT - a shift made
    without clear intent - so the stretch is one when it contains such a shift. Stretches judged
    before the redesign keep the older rule: the dominant kind by time."""
    if any(s.get("focused") is not None for s in run["segments"]):
        return "distracted" if any(s.get("self_distraction") for s in run["segments"]) else "interrupted"
    from collections import Counter
    c = Counter()
    for s in run["segments"]:
        c[s["state"]] += s["duration"]
    return c.most_common(1)[0][0] if c else "interrupted"


def _runs3(segs: list[dict]) -> list[dict]:
    """Runs of three categories: 'focus', 'detour' (interrupted+distracted MERGED - stacked
    interruptions/distractions are one event), and 'idle'. Split on recording gaps and on a
    deliberate focus start.

    Redesign 2026-11-19: a segment judged with a focused/unfocused field can be unfocused and
    still belong to a thread of work ("half-hearted fiddling in the same doc"). Such time forms
    a SOFT run: cat 'focus' with soft=True. Inside a running span on its own thread it counts
    as focus time (Rules doc); it can never start a span, never counts toward a span's opening
    10 minutes, and off the span's thread it is time away like any other. Unfocused time that
    began with a shift (a new, unfocused thread), passive consumption, and time the person
    themselves ruled unfocused are ordinary detours."""
    runs, cur = [], None
    in_detour = False   # the thread being followed began with a shift into unfocused time
    last_soft = False
    for s in segs:
        if s["duration"] <= 0:
            continue
        by_field = s.get("focused") is not None
        sw = s.get("switch") if by_field else None
        soft = False
        if s["state"] == "focus":
            cat, in_detour = "focus", False
        elif s["state"] == "idle":
            cat = "idle"
        else:
            cat = "detour"
            if by_field:
                if sw == "interruption":
                    in_detour = True
                elif sw == "return":
                    in_detour = bool(s.get("self_distraction"))   # back to a distraction, not to work
                soft = (s["focused"] is False and not in_detour and not s.get("user_unfocused")
                        and s.get("content") != "passive")
            elif s.get("content") in ("transition", "idle") and s.get("switch") == "continuation":
                soft = last_soft          # a pass-through or a pause inside soft time stays in it
            else:
                in_detour = True          # judged the older way: a detour is a detour
            if soft:
                cat = "focus"
        last_soft = soft
        gap = cur is not None and s["clip_start"] - cur["end"] > MAX_GAP
        new = (cur is None or cur["cat"] != cat or gap or cur.get("soft", False) != soft
               or (cat == "focus" and not soft and s["label"] == "task_change")
               # a shift or a return is a thread boundary: the next run may or may not belong
               # to the span, and _same_task has to be asked about it on its own
               or (cat == "focus" and sw in ("interruption", "return")))
        if new:
            cur = {"cat": cat, "start": s["clip_start"], "end": s["clip_end"], "segments": [s],
                   "focus_start": cat == "focus" and not soft and s["label"] == "task_change",
                   "gap_before": gap, "soft": soft}
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
        sp["review_chat_mid"] = r["chat_message_id"] if r is not None and "chat_message_id" in r.keys() else None
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
            sp["fully_absorbed"] = (len(sp["detours"]) == 0 and sp["subtype"] not in ("focused_task", "meeting")
                                    and not sp.get("hybrid_min"))
            keep_q.append(sp)
    for sp in disqualified:
        r = _span_review_for(rows, sp)
        sp["review_chat_mid"] = r["chat_message_id"] if r is not None and "chat_message_id" in r.keys() else None
        if r is not None and r["verdict"] == "focus":
            sp["confirmed"] = True
            sp["subtype"] = r["subtype"] if r["subtype"] in SUBTYPES else "other_focused"
            sp.pop("disq_reason", None)
            sp["fully_absorbed"] = (len(sp["detours"]) == 0 and sp["subtype"] not in ("focused_task", "meeting")
                                    and not sp.get("hybrid_min"))
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
    from .config import load_config as _lc0
    _cfg0 = _lc0()
    title_apps = set(_cfg0.title_apps)
    # min_focus_min=0 is a caller asking for every hypothesized span (focus_context): no gate.
    unbroken_s = _cfg0.min_unbroken_min * 60 if min_focus_min > 0 else 0
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
        by_field = first.get("focused") is not None   # judged under the redesign
        if by_field and first.get("switch") == "interruption":
            # A shift is a new thread, whatever window it is in. It stays in the span only as
            # the NEXT TASK of a focused task block (the evaluator's judgment), and only under
            # a running Toggl entry (Rules doc: the app applies that rule, as a veto).
            return bool(first.get("next_task") and (first.get("toggl_id") or first.get("tg_near_a")))
        has_ptr = by_field and first.get("antecedent_id") is not None
        # Rules doc: hand-coded rules may veto, never affirm. "Same window" is such a rule, so
        # where the evaluator has stated which thread this continues, that is read first.
        if not has_ptr and (first["app"], first["domain"]) == (last["app"], last["domain"]):
            # Same window/site = same task - except in an app whose window title names the
            # project (config title_apps, e.g. Agent Coding): there two projects share one app
            # and no site, and the Rules doc counts each project separately. A different title
            # falls through to the stated thread pointer and the evaluator's own judgment.
            ta, tb = (first.get("title") or ""), (last.get("title") or "")
            if not (first["app"] in title_apps and ta and tb and ta != tb):
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
        if has_ptr and a is None and conf >= 0.5:
            # The stated chain was followed, with confidence, back to where its thread began,
            # and that thread is neither this span nor the time away from it: a different thread.
            return False
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
        while i < n and (rs[i]["cat"] != "focus" or rs[i].get("soft")):
            i += 1   # a span starts on focused time; unfocused time on a thread does not start one
        if i >= n:
            return out
        cur = {"start": rs[i]["start"], "runs": [rs[i]]}
        i += 1
        away = []  # runs since we last left cur's task (detours and/or different-task focus)

        def away_s():
            return sum(r["duration"] for r in away)

        def reason():
            if any(r["cat"] == "focus" and not r.get("soft") for r in away):
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
                        # One attention-interruption event per time the person leaves the span,
                        # however many things they pass through while away (a focused errand,
                        # then a feed, then back = one event).
                        segs_away = [sg for ar in away for sg in ar["segments"]]
                        merged = {"cat": "detour", "soft": False, "focus_start": False, "gap_before": False,
                                  "start": away[0]["start"], "end": away[-1]["end"],
                                  "duration": sum(ar["duration"] for ar in away), "segments": segs_away,
                                  "toggl_id": next((ar.get("toggl_id") for ar in away if ar.get("toggl_id")), None)}
                        lab = _detour_label(merged)
                        merged["state"] = lab if lab in ("distracted", "interrupted") else "interrupted"
                        merged["parts"] = list(away)
                        cur["runs"].append(merged)
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

    from .config import load_config as _lc
    _cfg = _lc()
    # A focus span STARTS with its continuous block (user rule 2026-11-19: "every focus span
    # has to start with at least 10 consecutive focused minutes on the focused object"). A
    # choppy opening - focus broken by diversions before the first clean `min_unbroken_min` -
    # is cut off the front and kept as short focus, not span time. Not applied to a meeting
    # (a meeting is a meeting) or to a span the user has confirmed as shown (their review is
    # about those bounds, and is authoritative over the gates).
    if unbroken_s > 0:
        _rev = conn.execute("SELECT * FROM span_reviews").fetchall() if conn is not None else []
        cut = []
        for sp in spans:
            if sp["focus_min"] < min_focus_min or span_subtype(dict(sp), _cfg) == "meeting":
                continue
            rv = _span_review_for(_rev, sp)
            if rv is not None and rv["verdict"] == "focus":
                continue
            acc, k0, at = 0.0, None, None
            for k, r_ in enumerate(sp["runs"]):
                if r_.get("gap_before") or r_["cat"] == "detour" or r_.get("soft"):
                    acc, k0 = 0.0, None   # the opening 10 minutes have to be FOCUSED time
                if r_["cat"] == "focus" and not r_.get("soft"):
                    k0 = k if k0 is None else k0
                    acc += r_["duration"]
                    if acc >= unbroken_s:
                        at = k0
                        break
            if not at:  # no block at all (disqualified below), or the span already starts with it
                continue
            head_runs = sp["runs"][:at]
            if any(r_["cat"] == "focus" for r_ in head_runs):
                cut.append(_finalize({"start": sp["start"], "runs": head_runs}, "interrupted"))
            sp["runs"] = sp["runs"][at:]
            sp["start"] = sp["runs"][0]["start"]
            sp["trimmed_start_min"] = (sp["start"] - head_runs[0]["start"]) / 60
            _finalize(sp, sp["ended_by"])
        spans = sorted(spans + cut, key=lambda x: x["start"])
    qualifying = [sp for sp in spans if sp["focus_min"] >= min_focus_min]
    short = [sp for sp in spans if sp["focus_min"] < min_focus_min]
    for sp in qualifying:
        sp["subtype"] = span_subtype(sp, _cfg)
        # Hybrid span: the activity (reading, task, ...) happened inside a live discussion for
        # most of it. It keeps its activity subtype but is not INDEPENDENT focus - it is counted
        # with meetings, and can never be fully absorbed.
        fs = [s for r in sp["runs"] if r["cat"] == "focus" for s in r["segments"]]
        ftot = sum(s["duration"] for s in fs) or 1.0
        hyb_s = sum(s["duration"] for s in fs if s.get("hybrid"))
        sp["hybrid_min"] = hyb_s / 60   # minutes of this span that were inside a live discussion
        sp["hybrid"] = hyb_s / ftot >= 0.5
        # Fully-absorbed = zero interruptions AND a single object of attention - a focused-task
        # span is inherently task-switching, so it never counts (Rules doc). Nor does any span
        # that was partly carried on with someone else.
        sp["fully_absorbed"] = (len(sp["detours"]) == 0 and sp["subtype"] not in ("focused_task", "meeting")
                                and hyb_s == 0)
        # The doc defines fully-absorbed as "a LENGTH of focus span with 0 interruption
        # events" - a stretch, not only a whole span. Longest unbroken focus stretch:
        best = cur = 0.0
        best_f = cur_f = 0.0      # the same, counting only time judged focused (for the gate)
        soft_s = soft_run = soft_max = 0.0
        soft_ids: list[int] = []
        for r_ in sp["runs"]:
            if r_.get("gap_before"):
                cur = cur_f = 0.0
            if r_["cat"] == "focus":
                cur += r_["duration"]
                best = max(best, cur)
                if r_.get("soft"):
                    cur_f = 0.0
                    soft_s += r_["duration"]
                    soft_run += r_["duration"]
                    soft_max = max(soft_max, soft_run)
                    soft_ids += [sg["id"] for sg in r_["segments"]]
                else:
                    cur_f += r_["duration"]
                    best_f = max(best_f, cur_f)
                    soft_run = 0.0
            elif r_["cat"] == "detour":
                cur = cur_f = soft_run = 0.0
        sp["absorbed_max_min"] = best / 60
        # Time on the span's own thread that the evaluator judged UNFOCUSED. It counts as focus
        # (Rules doc: "I may still be thinking"), but when it is more than 20% of the span, or
        # one stretch of 10 minutes or more, the person is asked whether it really was
        # unfocused and should break the span. Their answer is stored per segment (focused 1/0).
        sp["unfocused_min"] = soft_s / 60
        sp["unfocused_max_min"] = soft_max / 60
        sp["unfocused_seg_ids"] = soft_ids
        sp["unfocused_ask"] = bool(soft_ids) and (soft_s > UNFOCUSED_ASK_FRAC * sp["duration"]
                                                   or soft_max >= UNFOCUSED_ASK_STRETCH_S)
        best = best_f
        # A span needs one unbroken block of `min_unbroken_min` (10) on its own object (user
        # rule 2026-11-17). The break threshold alone lets A 4 min / B 3 min / A 4 min ... run
        # on as one long span on A with B as contained interruptions, since no single absence
        # reaches 5 minutes; a stretch with no 10-minute continuous block is not a focus span.
        # Decided here, not by the evaluator: the honest switch labels (left A, returned to A)
        # are exactly what assembles that span.
        if best < unbroken_s and sp["subtype"] not in (None, "meeting"):  # a meeting is a meeting
            sp["subtype"] = None
            sp["disq_reason"] = f"no continuous block of {_cfg0.min_unbroken_min:g} minutes"
    # Spans disqualified by the Toggl gate (task churn, no entry) are unfocused time, not spans.
    disqualified = [sp for sp in qualifying if sp["subtype"] is None]
    qualifying = [sp for sp in qualifying if sp["subtype"] is not None]
    if conn is not None:  # the user's span reviews are authoritative over the gates
        qualifying, disqualified = _apply_span_reviews(conn, qualifying, disqualified)
    interruptions, distractions = [], []
    for sp in qualifying:
        for d in sp["detours"]:
            d["is_event"] = True; d["inside_span"] = True
            for part in d.get("parts", ()):
                part["inside_span"] = True   # not counted again below as time outside spans
            for sg in d["segments"]:
                sg["diversion"] = True   # time off the span's thread, focused or not (for the timeline)
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


def _is_sd_shift(s: dict) -> bool:
    return bool(s.get("redesign") and s.get("self_distraction") and s["duration"] > 0)


def self_distraction_counts(segs: list[dict], se: dict) -> tuple[int, int]:
    """(self-distraction events in the day, bounce-aways).

    A self-distraction event is a SHIFT made without clear intent, at any time of day; each one
    counts once. A bounce-away is the one shift by which the person LEAVES a focus span, when
    that shift is a self-distraction (Eli, 2026-11-19: writing -> email -> Bluesky -> writing
    has one bounce-away at most, "the moment you go from writing to email"; a self-distraction
    later in the same time away is a self-distraction event and not a bounce-away; leaving for
    a deliberate errand and then drifting is no bounce-away). So bounce-aways <= events, and
    at most one per leaving. Going back to a distraction visited a few minutes earlier is a
    'return' carrying its own self-distraction judgment, and counts like a shift. The leaving shift is the first shift of a diversion inside a span,
    or the first shift after a span's end. Stretches judged before the redesign have no
    per-shift flag and keep the older count: one per stretch of mostly-distracted time."""
    events = sum(1 for s in segs if _is_sd_shift(s))
    bounces = 0
    for sp in se["focus_spans"]:
        for d in sp["detours"]:
            first = next((sg for sg in d["segments"] if sg.get("switch") in ("interruption", "return")), None)
            bounces += int(first is not None and _is_sd_shift(first))
        prev_end = sp["end"]
        for s in segs:                     # the shift that ended the span, if a shift ended it
            if s["clip_start"] < sp["end"] - 1 or s["duration"] <= 0:
                continue
            if s["clip_start"] - prev_end > MAX_GAP or s["state"] == "idle":
                break
            if s.get("switch") in ("interruption", "return"):
                bounces += int(_is_sd_shift(s))
                break
            prev_end = s["clip_end"]       # windows passed through on the way out
    for d in se["distractions"]:
        if not any(sg.get("redesign") for sg in d["segments"]):
            events += 1
            bounces += int(bool(d.get("inside_span")))
    return events, bounces


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


def daily_metrics(conn, day: dt.date, overrides: dict | None = None) -> dict:
    t0, t1 = day_bounds(day)
    segs = labelled_segments(conn, t0, t1, overrides)
    se = spans_and_events(segs, conn=conn)
    focus = se["focus_spans"]
    # Meetings are focus, but their own kind: counted separately, never in "total focus".
    # Hybrid time (an activity carried on inside a live discussion) goes with them, minute by
    # minute: a mostly-hybrid span moves whole, a solo span with a hybrid stretch moves that stretch.
    work = [s for s in focus if s.get("subtype") != "meeting" and not s.get("hybrid")]
    meetings = [s for s in focus if s.get("subtype") == "meeting" or s.get("hybrid")]
    hybrid_in_work = sum(min(s.get("hybrid_min", 0.0), s["focus_min"]) for s in work)
    breaks = short_breaks(conn, t0, t1)
    sd_events, bounces = self_distraction_counts(segs, se)
    return {
        "short_breaks": len(breaks),
        "short_break_min": sum(b["end"] - b["start"] for b in breaks) / 60,
        "day": day.isoformat(),
        "longest_focus_min": max((s["duration"] for s in work), default=0) / 60,
        "longest_absorbed_min": max((s.get("absorbed_max_min", 0.0) for s in work
                                     if s.get("subtype") != "focused_task"), default=0.0),
        "fully_absorbed_spans": sum(1 for s in work if s.get("fully_absorbed")),
        "total_focus_min": sum(s["focus_min"] for s in work) - hybrid_in_work,
        "meeting_focus_min": sum(s["focus_min"] for s in meetings) + hybrid_in_work,
        "detours_in_spans": sum(len(s["detours"]) for s in focus),
        "detour_min_in_spans": sum(s["detour_min"] for s in focus),
        # An attention-interruption event = any diversion inside a focus span (self-distraction included).
        "interruption_count": (sum(1 for r in se["interruptions"] if r.get("inside_span"))
                               + sum(1 for r in se["distractions"] if r.get("inside_span"))),
        "distraction_count": bounces,   # older name for bounce_aways, kept for callers
        # Rules doc: a self-distraction event can happen at any time of day; a BOUNCE-AWAY is
        # one that happens during a focus span. (Until 2026-11-19 the app had these two names
        # the other way round.)
        "self_distraction_events": sd_events,
        "bounce_aways": bounces,
        "distracted_min": sum(d["duration"] for d in se["distractions"]) / 60,
        "interrupted_min": sum(d["duration"] for d in se["interruptions"]) / 60,
        "active_min": sum(s["duration"] for s in segs if s["state"] != "idle") / 60,
        # Rules doc: passive consumption is tracked in its own right (never counts toward focus).
        "passive_min": sum(s["duration"] for s in segs if s.get("content") == "passive") / 60,
        # Correspondence (chat, email) is work but interactive: tracked in its own right, never a focus span.
        "correspondence_min": sum(s["duration"] for s in segs if s.get("content") == "correspondence") / 60,
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


def day_stamp(conn, t0: float, t1: float) -> str:
    """Fingerprint of everything the Day view draws from. The dashboard polls it and reloads the
    day only when it changes - so an edit made anywhere (a chat, the review queue, a Toggl
    correction, the evaluator catching up) shows on the timeline within seconds. The live
    segment's end is rounded to the minute so an open day refreshes about once a minute."""
    import hashlib
    qs = (
        ("SELECT COUNT(*), MAX(id), CAST(MAX(COALESCE(end, start)) / 60 AS INTEGER) FROM segments"
         " WHERE start >= ? AND start < ?", (t0, t1)),
        ("SELECT COUNT(*), MAX(e.created) FROM seg_evals e JOIN segments g ON g.id = e.segment_id"
         " WHERE g.start >= ? AND g.start < ?", (t0, t1)),
        ("SELECT COUNT(*), MAX(r.created) FROM seg_reviews r JOIN segments g ON g.id = r.segment_id"
         " WHERE g.start >= ? AND g.start < ?", (t0, t1)),
        ("SELECT COUNT(*), MAX(r.created) FROM reviews r JOIN switches s ON s.id = r.switch_id"
         " WHERE s.ts >= ? AND s.ts < ?", (t0, t1)),
        ("SELECT COUNT(*), MAX(e.created) FROM ensemble e JOIN switches s ON s.id = e.switch_id"
         " WHERE s.ts >= ? AND s.ts < ?", (t0, t1)),
        ("SELECT COUNT(*), MAX(created) FROM span_reviews WHERE start < ? AND end > ?", (t1, t0)),
        ("SELECT COUNT(*), MAX(created) FROM span_summaries WHERE start < ? AND end > ?", (t1, t0)),
        ("SELECT COUNT(*), MAX(q.created), MAX(q.answered_at), SUM(q.status = 'open') FROM review_questions q"
         " JOIN segments g ON g.id = q.segment_id WHERE g.start >= ? AND g.start < ?", (t0, t1)),
        ("SELECT COUNT(*), MAX(start), MAX(COALESCE(stop, 0)), SUM(LENGTH(COALESCE(description, '')))"
         " FROM toggl_entries WHERE start < ? AND COALESCE(stop, ?) > ?", (t1, t1, t0)),
        ("SELECT COUNT(*), MAX(end) FROM inactivity WHERE start >= ? AND start < ?", (t0, t1)),
    )
    raw = "|".join(",".join(str(v) for v in conn.execute(sql, args).fetchone()) for sql, args in qs)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _same_thread(conn, a: int | None, b: int | None, max_hops: int = 60) -> bool:
    """Two thread pointers agree when they name the same segment or one is an ancestor of the
    other along the antecedent chain (the person's links outrank the evaluator's)."""
    if a is None or b is None:
        return a == b
    if a == b:
        return True

    def chain(sid):
        seen = []
        cur = sid
        while cur is not None and cur not in seen and len(seen) < max_hops:
            seen.append(cur)
            r = conn.execute("""SELECT COALESCE(rv.antecedent_id, e.antecedent_id) AS a
                                FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
                                LEFT JOIN seg_reviews rv ON rv.segment_id = g.id WHERE g.id = ?""", (cur,)).fetchone()
            cur = r["a"] if r else None
        return seen
    return b in chain(a) or a in chain(b)


def agreement_report(conn, days: int = 60) -> dict:
    """How often the app's judgment matched the person's. Each of the person's segment
    verdicts (seg_reviews) is compared with the judgment the app had made BEFORE they ruled
    (seg_evals created earlier); a verdict agrees when every field it set matches: content,
    switch relation, and the thread pointer (same thread, not necessarily the same link).
    Span verdicts count a confirmed span as agreement and a rejected one as a correction."""
    t0, _ = day_bounds(logical_today() - dt.timedelta(days=days - 1))
    raw = conn.execute("""
        SELECT r.segment_id, r.created, r.content AS rc, r.switch_label AS rsw, r.antecedent_id AS ra,
               e.content AS ec, e.switch_label AS esw, e.antecedent_id AS ea, e.created AS ec_at, g.start
        FROM seg_reviews r JOIN seg_evals e ON e.segment_id = r.segment_id
        JOIN segments g ON g.id = r.segment_id ORDER BY r.created""").fetchall()
    rows = []
    unscored = 0
    for r in raw:
        if r["ec_at"] < r["created"]:
            rows.append(dict(r))
            continue
        # The app re-judged this segment after the person ruled (re-evaluation, propagation),
        # so the live row is not what they corrected. The judgment they saw is in the
        # transcript of the earliest evaluation run that covered the segment.
        found = None
        for run in conn.execute("""SELECT response FROM eval_runs WHERE t0 <= ? AND t1 >= ? AND created < ?
                                   AND response IS NOT NULL ORDER BY created LIMIT 4""",
                                (r["start"], r["start"], r["created"])):
            try:
                js = json.loads(run["response"])
                segs = js.get("segments", js) if isinstance(js, dict) else js
                found = next((x for x in segs if x.get("segment_id") == r["segment_id"]), None)
            except (ValueError, AttributeError, TypeError):
                found = None
            if found:
                break
        if not found:
            unscored += 1
            continue
        ante = (found.get("antecedents") or [{}])[0].get("segment_id") if found.get("switch") in ("continuation", "return") else None
        rows.append(dict(r, ec=found.get("content"), esw=found.get("switch"), ea=ante))
    per_day: dict[str, dict] = {}
    tot = {"n": 0, "agree": 0, "by_head": {"content": 0, "switch": 0, "thread": 0}}
    recent = {"n": 0, "agree": 0}
    for r in rows:
        heads = []
        if r["rc"]:
            heads.append(("content", r["rc"] == r["ec"]))
        if r["rsw"]:
            heads.append(("switch", r["rsw"] == r["esw"]))
            if r["rsw"] in ("continuation", "return") and r["rsw"] == r["esw"] and r["ra"] is not None:
                heads.append(("thread", _same_thread(conn, r["ra"], r["ea"])))
        if not heads:
            continue
        agree = all(ok for _, ok in heads)
        tot["n"] += 1
        tot["agree"] += int(agree)
        for h, ok in heads:
            if not ok:
                tot["by_head"][h] += 1
        if r["created"] >= t0:
            day = logical_date(r["created"]).isoformat()
            d = per_day.setdefault(day, {"day": day, "n": 0, "agree": 0, "content": 0, "switch": 0, "thread": 0})
            d["n"] += 1
            d["agree"] += int(agree)
            for h, ok in heads:
                if not ok:
                    d[h] += 1
        if r["created"] >= time.time() - 30 * 86400:
            recent["n"] += 1
            recent["agree"] += int(agree)
    spans = conn.execute("SELECT created, verdict FROM span_reviews ORDER BY created").fetchall()
    sp_tot = {"n": 0, "agree": 0}
    sp_recent = {"n": 0, "agree": 0}
    sp_day: dict[str, dict] = {}
    for s in spans:
        ok = int(s["verdict"] == "focus")
        sp_tot["n"] += 1
        sp_tot["agree"] += ok
        if s["created"] >= time.time() - 30 * 86400:
            sp_recent["n"] += 1
            sp_recent["agree"] += ok
        if s["created"] >= t0:
            day = logical_date(s["created"]).isoformat()
            d = sp_day.setdefault(day, {"day": day, "n": 0, "agree": 0})
            d["n"] += 1
            d["agree"] += ok
    ratio = lambda x: (x["agree"] / x["n"]) if x["n"] else None
    return {"segments": {"total": tot, "ratio": ratio(tot), "recent": recent, "ratio_recent": ratio(recent),
                         "daily": [dict(d, ratio=d["agree"] / d["n"]) for d in sorted(per_day.values(), key=lambda d: d["day"])],
                         "judged_after_ruling": unscored},
            "spans": {"total": sp_tot, "ratio": ratio(sp_tot), "recent": sp_recent, "ratio_recent": ratio(sp_recent),
                      "daily": [dict(d, ratio=d["agree"] / d["n"]) for d in sorted(sp_day.values(), key=lambda d: d["day"])]}}

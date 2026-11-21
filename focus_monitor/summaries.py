"""Span summaries: once a hypothesized focus span has ended, Claude writes a short factual
summary of what the person was actually doing in it. The summary is shown in every capture
view (timeline hover) inside that span - together with the running Toggl entry - so each
span judgment can be audited at a glance."""
from __future__ import annotations

import datetime as _dt
import datetime as dt
import logging
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import PRICE_INPUT, PRICE_OUTPUT, Config, load_config


class SpanSummaryJudgement(BaseModel):
    title: str = Field(description="3-8 words naming the OBJECT OF ATTENTION, like a document title - e.g. 'Reading METR's incident report', 'Editing elityre.com date page'. Use an epistemic marker ('likely ...') only if genuinely unsure.")
    summary: str = Field(description="1-2 factual sentences: what the person was actually doing.")
    coherent: bool = Field(description="THE CHECK: a focus span means ONE project/task/intention held throughout, with contained interruptions under 5 minutes each. Is the stretch you just described plausibly one object of attention? False whenever your own description contradicts that - e.g. the first stretch is a different activity than the rest, phrases like 'switched to' or 'from X on', or a sequence of unrelated things. (The reading->creative merge only excuses a shift when the writing grows out of the SAME material being read; moving to a different project is a different object.)")
    issue: str | None = Field(default=None, description="When coherent is false: one sentence naming the contradiction and, if apparent, where the real boundary lies.")

log = logging.getLogger("summaries")

_client = None
_no_creds_logged = False


def _get_client():
    global _client, _no_creds_logged
    if _client is None:
        import anthropic
        from .config import load_api_key
        load_api_key()  # Finder-launched processes have no shell env; pull from Keychain
        c = anthropic.Anthropic()
        if not (getattr(c, "api_key", None) or getattr(c, "auth_token", None)):
            if not _no_creds_logged:
                log.warning("no Anthropic credentials; span summaries disabled")
                _no_creds_logged = True
            return None
        _client = c
    return _client


def lookup(conn, start: float, end: float):
    """The stored summary matching [start, end). The overlap must cover >= 85% of BOTH the
    span and the stored range - a summary written for materially different boundaries (e.g.
    before a span was recalculated) is stale and must not be served; returning None here is
    what triggers regeneration."""
    rows = conn.execute("SELECT * FROM span_summaries WHERE start < ? AND end > ?",
                        (end, start)).fetchall()
    best, best_ov = None, 0.0
    for r in rows:
        ov = min(end, r["end"]) - max(start, r["start"])
        if ov > best_ov:
            best, best_ov = r, ov
    if best is None:
        return None
    ok = best_ov >= 0.85 * (end - start) and best_ov >= 0.85 * (best["end"] - best["start"])
    return best if ok else None


def dur_text(seconds: float) -> str:
    """A duration for a model to read: whole seconds under two minutes (the Rules doc's
    pass-through limit is 15 seconds, and "0.2m" covered 9 to 15), minutes and seconds above."""
    sec = max(0.0, float(seconds or 0.0))
    if sec < 120:
        return f"{sec:.0f}s"
    m, r = divmod(int(round(sec)), 60)
    return f"{m // 60}h{m % 60:02d}m{r:02d}s" if m >= 60 else f"{m}m{r:02d}s"


def verdict_text(r) -> str:
    """The fields a ruling (a seg_reviews row, or anything with those keys) sets, in the words
    models are given. Empty when it sets none."""
    from .seg_models import shown_switch
    g = lambda k: (r[k] if k in r.keys() else None)
    bits = [f"content={g('content')}" if g("content") else None,
            f"switch={shown_switch(g('switch_label'))}" if g("switch_label") else None,
            f"continues id={g('antecedent_id')}" if g("antecedent_id") else None,
            None if g("focused") is None else ("focused" if g("focused") else "unfocused"),
            None if g("self_distraction") is None else ("self-distraction" if g("self_distraction") else "not a self-distraction"),
            None if g("next_task") is None else ("next task in the block" if g("next_task") else "not the next task"),
            None if g("hybrid") is None else ("hybrid (inside a live discussion)" if g("hybrid") else "solo, not hybrid"),
            "counts as independent focus by their explicit exception" if g("focus_exception") else None,
            # the older single label for a shift, on rulings made before 2026-11-19
            f"kind of shift (older label)={g('interruption_kind')}" if g("interruption_kind") and g("focused") is None else None]
    return ", ".join(b for b in bits if b)


def _tags(s) -> list[str]:
    bits = []
    if s.get("content"):
        bits.append(f"<{s['content']}>" + (" <FOCUS EXCEPTION by ruling>" if s.get("focus_exception") else ""))
    if s.get("focused") is not None:
        # judged under the 2026-11-19 redesign: a focus judgment per block, and the switch as
        # continuation / shift / return with two yes-no flags on a shift
        bits.append("<focused>" if s["focused"] else
                    ("<unfocused, ruled by the person>" if s.get("user_unfocused") else "<unfocused>"))
        if s.get("switch") == "interruption":
            bits.append("<switch: shift" + (", self-distraction" if s.get("self_distraction") else "")
                        + (", next task in the block" if s.get("next_task") else "") + ">")
        elif s.get("switch") == "return":
            bits.append("<switch: return>")
    elif s.get("label") and s["label"] != "continuation":
        # judged before the redesign: the stored four-way label, in the words models are given
        bits.append("<switch: " + {"interruption": "shift (detour)", "distraction": "shift (self_distraction)",
                                   "task_change": "shift (focus_start)"}.get(s["label"], s["label"]) + ">")
    return bits


def seg_line(s, full: bool = False, ruling=None, quick_pass: bool = False) -> str:
    """One activity-log line for a segment (a stats.labelled_segments row). Times are in the
    zone the Mac was in then (config.local_dt): re-judgments used to show activity from before
    the move to Eastern time three hours late (segment 24715: 13:39 when first judged, 16:39
    when re-judged).

    The default is the short form the question writer, the chats and the propagation check
    use. `full` (the evaluator and the daily audit, 2026-11-20) leaves nothing out and adds
    nothing that is not a judgment:
      * start to the second, duration in seconds under two minutes, the whole title, the URL;
      * the segment's real start and length even where the window shown cuts it;
      * judgment tags ONLY for a segment the evaluator (or the person) has judged. The short
        form also prints the app-derived state word, the old per-switch classifier's label and
        its [activity] guess, on judged and unjudged lines alike; on unjudged lines the
        evaluator read them as existing judgments (it said shift on 44% of lines carrying the
        old tag against 24% without);
      * `ruling` (the person's seg_reviews row) marks the line RULED BY THE PERSON with the
        fields they set and their note in full;
      * `quick_pass` marks the tags as an earlier low-effort judgment (history re-judgment)."""
    from .config import local_dt
    if not full:
        t = local_dt(s["clip_start"]).strftime("%H:%M")
        what = (s.get("app") or "") + ((" " + s["domain"]) if s.get("domain") else "")
        title = (s.get("title") or "")[:70]
        bits = [t, "%4.1fm" % (s["duration"] / 60), s.get("state", ""), what, title] + _tags(s)
        if s.get("activity"):
            bits.append(f"[{s['activity']}]")
        if s.get("toggl_desc"):
            bits.append(f"(toggl: {s['toggl_desc']})")
        ia, dur2 = s.get("inact_s") or 0.0, s.get("duration") or 1.0
        if ia >= 60 and ia >= 0.3 * dur2:
            bits.append(f"(input idle {ia / 60:.0f}m of {dur2 / 60:.0f}m)")
        return "  ".join(str(b) for b in bits)
    start = s.get("start", s["clip_start"])
    end = s.get("end_eff") or s.get("clip_end") or start
    one = lambda v: " ".join(str(v or "").split())
    bits = [local_dt(start).strftime("%H:%M:%S"), dur_text(end - start)]
    if s.get("idle"):
        bits.append("AWAY (ten minutes or more with no keyboard or mouse input)")
    else:
        # A window labelled "idle" for ten minutes or more has state idle too (stats), but it is a
        # real window with a title and input counts: printed as AWAY it lost all of that, and as a
        # segment to judge its line confirmed its own label (segment 28260, 2026-11-21).
        if s.get("state") == "idle":
            bits.append("[counted as time away: labelled idle for ten minutes or more]")
        bits.append(one(s.get("app")))
        if s.get("domain"):
            bits.append("site: " + s["domain"])
        if one(s.get("title")):
            bits.append("title: “" + one(s["title"]) + "”")
        if s.get("url"):
            bits.append("url: " + s["url"])
        if s.get("neutral"):
            bits.append("(neutral window, e.g. the presence-check form: recorded, never judged)")
    if "end" in s and s.get("end") is None:
        bits.append("(still in front when this log was written)")
    judged = s.get("source") in ("eval", "review") and not s.get("idle")
    if judged:
        tags = _tags(s)
        if s.get("focused") is not None and s.get("switch") in (None, "continuation"):
            tags.append("<switch: continuation>")
        elif s.get("focused") is None and s.get("redesign"):
            # a pass-through or idle segment judged since 2026-11-19: no focus tag, but its switch is a judgment too
            tags.append("<switch: " + {"interruption": "shift", "return": "return"}.get(s.get("switch"), "continuation") + ">")
        if s.get("antecedent_id") and s.get("switch") in ("continuation", "return"):
            tags.append(f"<continues id={s['antecedent_id']}>")
        if s.get("hybrid"):
            tags.append("<hybrid>")
        if s.get("uncertain"):
            tags.append("<flagged uncertain>")
        if tags:
            bits.append(("stands as: " if ruling is not None else
                         "earlier quick-pass judgment: " if quick_pass else "judged: ") + " ".join(tags))
    if ruling is not None:
        note = one(ruling["note"])
        bits.append("RULED BY THE PERSON: " + (verdict_text(ruling) or "(no field set)")
                    + (f"; their note: “{note}”" if note else ""))
    if not s.get("idle"):
        if s.get("toggl_desc") or s.get("toggl_id"):
            bits.append("(toggl entry covering most of it: “" + one(s.get("toggl_desc")) + "”"
                        + (f" | project: {s['toggl_project']}" if s.get("toggl_project") else "")
                        + (f" | tags: {s['toggl_tags']}" if s.get("toggl_tags") else "") + ")")
        else:
            bits.append("(no toggl entry during it)")
        ia = s.get("inact_s") or 0.0
        if ia >= 1:
            bits.append(f"(no input for {dur_text(ia)} of it, in pauses of a minute or more)")
    return "  ".join(str(b) for b in bits)


def _span_log(sp) -> str:
    lines = [seg_line(s) for r in sp["runs"] for s in r["segments"]]
    if len(lines) > 150:  # keep the prompt bounded for very long spans
        lines = lines[:75] + [f"  ... {len(lines) - 150} segments elided ..."] + lines[-75:]
    return "\n".join(lines)


PROMPT = """Below is the activity log of one hypothesized focus span ({t0}-{t1}, {mins:.0f} min), with
about ten minutes of what came before and after it. Lines marked "(in the span ...)" are the span;
lines marked "(OUTSIDE the span ...)" are context only: do not describe them as part of it, but
use them to see whether the span's start and end are in the right place.
Write a 1-2 sentence factual summary of what the person was actually DOING - the undertaking,
not the window inventory: "editing the date page of your personal site (VS Code + Terminal +
checking the rendered pages)" rather than "cycling between Code, Terminal, and Chrome". Infer
the activity from titles, content labels, the notes after "// note:", the screenshots and patterns,
with epistemic markers ("apparently", "likely") when inferring. Name the concrete sites, apps, or
documents as evidence; be honest about drift, channel-surfing, or mixed activity; note whether a
Toggl entry was running.

Then CHECK your own description against the definition: a focus span holds ONE object of
attention throughout (window-switching within one piece of work is fine; interruptions under
5 minutes are contained). If what you just described is not plausibly one object of attention,
say so (coherent=false) and name the contradiction - this triggers review of the span's
boundaries. A segment marked RULED BY THE PERSON is settled: what they ruled is what happened.

WHAT THE APP WORKED OUT ABOUT THIS SPAN (from the judgments on the lines below; facts about the
app's bookkeeping, not evidence of what the person was doing):
{facts}

{shots}

{log}"""

SUMMARY_IMAGES = 60      # images per call, as the evaluator's budget (evaluator.IMAGE_BUDGET)
CONTEXT_S = 10 * 60      # log shown before and after the span, marked as outside it
MAX_TOKENS = 4000        # 500 before: thinking counts against it, and a cut-off answer is no summary


def span_facts(sp) -> list[str]:
    """What the app computed about a span (stats.spans_and_events), in sentences: its kind or
    why it was disqualified, how it ended, the diversions it contains, an opening that was cut
    off, time judged unfocused, the person's verdict. First entry: its bounds."""
    from .config import local_dt
    t = lambda ts: local_dt(ts).strftime("%H:%M:%S")
    out = [f"{t(sp['start'])} to {t(sp['end'])}, {dur_text(sp['duration'])} long, {sp.get('focus_min', 0):.0f} min of it counted as focus."]
    if sp.get("disq_reason"):
        out.append(f"The app does NOT count it as a focus span: {sp['disq_reason']}.")
    elif sp.get("subtype"):
        out.append(f"The app counts it as a focus span of kind '{sp['subtype']}'"
                   + (", carried on inside a live discussion for most of it (hybrid)" if sp.get("hybrid") else "")
                   + (", fully absorbed (no diversion)" if sp.get("fully_absorbed") else "") + ".")
    if sp.get("confirmed") is not None and sp.get("review_chat_mid") is not None or sp.get("confirmed"):
        out.append("The person has reviewed this span themselves" + (" (through a chat)" if sp.get("review_chat_mid") else "") + ".")
    if sp.get("ended_by"):
        out.append(f"How it ended: {sp['ended_by']}.")
    det = sp.get("detours") or []
    if det:
        out.append(f"Contained diversions (time off the span's thread, under 5 minutes each, counted inside it): {len(det)}, "
                   f"{sp.get('detour_min', 0):.1f} min in all:")
        for r in det:
            where = ", ".join(dict.fromkeys((g.get("app") or "?") + (f" {g['domain']}" if g.get("domain") else "") for g in r["segments"]))
            out.append(f"  {t(r['start'])} for {dur_text(r['duration'])}: {where} (segment ids "
                       + ", ".join(dict.fromkeys(str(g["id"]) for g in r["segments"])) + ")")
    else:
        out.append("Contained diversions: none.")
    if sp.get("trimmed_start_min"):
        out.append(f"Its opening was cut off: the {sp['trimmed_start_min']:.1f} min before {t(sp['start'])} were on the same thread "
                   "but broken up, and a span has to start with 10 continuous focused minutes.")
    if sp.get("unfocused_min"):
        out.append(f"Time on the span's own thread judged UNFOCUSED (it counts as focus unless the person says otherwise): "
                   f"{sp['unfocused_min']:.1f} min, longest stretch {sp.get('unfocused_max_min', 0):.1f} min"
                   + ("; the person is being asked whether it really was unfocused" if sp.get("unfocused_ask") else "") + ".")
    return out


def span_images(conn, inside: list[dict], budget: int = SUMMARY_IMAGES) -> tuple[list, str]:
    """Screenshots of a span for a model, and the sentence that says what is attached. One of
    each distinct app/site (its longest segment), then one of every segment over a minute,
    longest first; what is left of the budget goes to further moments of the longest segments.
    Each is labelled with its segment id, time and app. The coherence check is a judgment about
    content, and Agent Coding, Terminal and Roam titles carry little of it. `inside` are the
    span's segments (stats.labelled_segments rows, one per real segment)."""
    from . import prompt_parts as pp, shots as _shots
    live = [g for g in inside if not g["idle"]]
    mom = _shots.moments(conn, [g["id"] for g in live])
    by_len = sorted((g for g in live if mom.get(g["id"])), key=lambda g: -g["duration"])
    ctx_first: dict = {}
    for g in by_len:
        ctx_first.setdefault((g["app"], g["domain"]), g)
    first = list(ctx_first.values())
    order = first + [g for g in by_len if g["duration"] >= 60 and not any(g is f for f in first)]
    images = [im for g in order[:budget] for im in pp.seg_images(mom, g, "span segment", max_moments=1, both=False)]
    used = {im["path"] for im in images}
    for g in by_len:     # more moments of the long segments while there is room
        for im in pp.seg_images(mom, g, "span segment", both=False):
            if len(images) < budget and im["path"] not in used:
                images.append(im)
                used.add(im["path"])
    content, meta = pp.image_blocks(images[:budget])
    shown_ids = {m_["label"].split("id=")[1].split(" ")[0] for m_ in meta}
    none = [g for g in live if g["duration"] >= 60 and str(g["id"]) not in shown_ids]
    note = (f"Screenshots attached above: {len(meta)}, of {len(shown_ids)} of the span's {len(live)} segments, each labelled "
            "with its segment id, the time it was taken and the app."
            + (" Segments of a minute or more with no screenshot: "
               + "; ".join(f"id={g['id']} ({_shots.why_none(conn, g['id']) if not mom.get(g['id']) else 'the image allowance was used up'})"
                           for g in none) + ". A missing screenshot says nothing about the activity." if none else ""))
    return content, note


def generate(conn, cfg: Config, sp) -> bool:
    """Write and store a summary for one finished span, and check that it is one object of
    attention (a "no" has the evaluator re-judge the span, then asks the person).

    What the call is shown, since 2026-11-20 (Eli: every model call gets all the information
    that could be relevant). Before, it had four screenshots labelled with a time only (181 of
    191 spans had more than four capture moments, median 42, and 91 spans more than four
    distinct app/site contexts), the span's lines in the short form with no ids, nothing from
    outside the span - where a wrong boundary would show - and none of what the app computed
    (54 of 191 were disqualified spans, and the call was not told). Now: the evaluator's line
    for every segment (ids, whole titles and notes, thread pointers, the person's rulings with
    their notes, signals), ten minutes before and after marked as outside, the span's computed
    facts, and a screenshot of every distinct app/site and of every segment over a minute,
    within SUMMARY_IMAGES, each labelled with its segment id and app."""
    client = _get_client()
    if client is None:
        return False
    from . import prompt_parts as pp
    from .config import local_dt   # the zone the Mac was in then, as in seg_line
    t0s = local_dt(sp["start"]).strftime("%H:%M")
    t1s = local_dt(sp["end"]).strftime("%H:%M")
    away = {g["id"] for r_ in sp["runs"] if r_["cat"] == "detour" for g in r_["segments"]}
    lo, hi = sp["start"] - CONTEXT_S, sp["end"] + CONTEXT_S
    _, rows = pp.log_lines(conn, cfg, lo, hi, phone=False)
    inside = [g for g in rows if g["clip_end"] > sp["start"] and g["clip_start"] < sp["end"]]
    marks = {g["id"]: (f"(in the span, a contained diversion, id={g['id']}) " if g["id"] in away else f"(in the span, id={g['id']}) ")
             for g in inside}
    lines, _ = pp.log_lines(conn, cfg, lo, hi, marks=marks, dated=True, default_mark="(OUTSIDE the span, context id={id}) ")
    content, shots_note = span_images(conn, inside)
    text = PROMPT.format(t0=t0s, t1=t1s, mins=sp["duration"] / 60, facts="\n".join(span_facts(sp)), shots=shots_note,
                         log="Activity log:\n" + "\n".join(lines))
    content.append({"type": "text", "text": text})
    from .classifiers.claude_vision import _rules_doc
    system = [{"type": "text", "text": ("You summarize hypothesized focus spans for a personal "
              "attention monitor, and check them against the person's own rules.\n\n" + pp.LOG_LEGEND
              + "\n\nTheir rules and definitions (authoritative):\n" + _rules_doc()
              + "\n\nAbout this person (their own words):\n" + (cfg.about_me.strip() or "(none)")),
              "cache_control": {"type": "ephemeral"}}]
    resp = llm.parse(client, site="summary", 
        model=cfg.claude_model, max_tokens=MAX_TOKENS, output_config={"effort": "medium"},
        system=system,
        messages=[{"role": "user", "content": content}],
        output_format=SpanSummaryJudgement)
    if resp.parsed_output is None:
        return False
    from . import efforts as _eff

    def _scall(eff):
        return llm.parse(client, site="summary_shadow", 
            model=cfg.claude_model, max_tokens=MAX_TOKENS, output_config={"effort": eff},
            system=system, messages=[{"role": "user", "content": content}],
            output_format=SpanSummaryJudgement)

    def _scmp(a, b):
        agree = 1.0 if a.coherent == b.coherent else 0.0
        return agree, f"coherent: {a.coherent} vs {b.coherent}"

    _eff.run_shadows(conn, cfg, client, "span_summarizer", f"span:{t0s}-{t1s}",
                     "medium", resp, _scall, _scmp,
                     prompt_text=f"span summary and coherence check, {t0s}-{t1s} ({sp['duration'] / 60:.0f} min); the prompt is not stored")
    j: SpanSummaryJudgement = resp.parsed_output
    text = j.summary.strip()
    if not text:
        return False
    cost = (resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
    with db.tx(conn):
        # replace any stale summary substantially overlapping this stretch (e.g. one written
        # for the same span before its boundaries were recalculated)
        for r in conn.execute("SELECT id, start, end FROM span_summaries WHERE start < ? AND end > ?",
                              (sp["end"] + 60, sp["start"] - 60)).fetchall():
            ov = min(sp["end"], r["end"]) - max(sp["start"], r["start"])
            if ov >= 0.5 * (r["end"] - r["start"]) or ov >= 0.5 * (sp["end"] - sp["start"]):
                conn.execute("DELETE FROM span_summaries WHERE id=?", (r["id"],))
        conn.execute("INSERT INTO span_summaries(start, end, summary, cost_usd, created, coherent, issue, title)"
                     " VALUES (?,?,?,?,?,?,?,?)",
                     (sp["start"], sp["end"], text, cost, time.time(), int(j.coherent),
                      (j.issue or None) if not j.coherent else None, j.title.strip() or None))
    log.info("summarized span %s-%s ($%.4f)%s", t0s, t1s, cost,
             "" if j.coherent else " - COHERENCE ISSUE: " + (j.issue or ""))
    return True


def ensure_summaries(conn, cfg: Config | None = None, max_new: int = 2) -> int:
    """Summarize up to max_new recently finished spans (qualifying + disqualified) that
    don't have a stored summary yet. Called from the classifier loop."""
    from . import stats
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    now = time.time()
    # The window starts on a day boundary and a span touching its first minutes is left alone.
    # It used to start at "now minus 7 days" to the second: a span crossing that moving edge was
    # cut a little shorter every cycle, no longer matched its stored summary, and was summarised
    # again and again until it had slid out (14 summaries of one 11-14 span in 19 minutes on
    # 2026-11-21, the last of an 8-second sliver; 12 of 13 spans of 11-12 lost theirs), and a
    # cut span could fail the coherence check and have its stretch re-judged.
    from .config import day_bounds, logical_date
    w0 = day_bounds(logical_date(now) - _dt.timedelta(days=7))[0]
    segs = stats.labelled_segments(conn, w0, now)  # cover the span-review queue window
    se = stats.spans_and_events(segs, conn=conn)
    done = 0
    for sp in se["focus_spans"] + se["disqualified"]:
        if done >= max_new:
            break
        if sp["start"] <= w0 + 120:
            continue
        if sp.get("ended_by") in ("ongoing", "awaiting evaluation"):
            continue
        if lookup(conn, sp["start"], sp["end"]) is not None:
            continue
        try:
            if generate(conn, cfg, sp):
                done += 1
        except Exception:
            log.exception("failed to summarize span at %s",
                          dt.datetime.fromtimestamp(sp["start"]).strftime("%H:%M"))
            break  # don't hammer the API on a persistent error
    return done

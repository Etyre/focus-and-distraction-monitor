"""Pieces of the prompts of the model calls OUTSIDE the segment evaluator (2026-11-20).

Eli: "We should also make sure that all the model calls are given all the information that could
possibly be relevant for their judgement and nothing is left out." The evaluator got this the same
day (evaluator.py: _log_lines, _seg_moments, _moment_images, _similar_precedents). The other
sites - feedback propagation, the question writer, rule drafts, span summaries, the chats - each
built their own thinner view from summaries.seg_line's short form: HH:MM with no date, titles
cut at 70 characters, notes cut at 70-90, no thread pointer, no mark on the person's rulings, no
signals, one screenshot. This module gives them the evaluator's own pieces, so every model reads
the same line for the same segment:

  log_lines         the activity log of a stretch, built by evaluator._log_lines
  LOG_LEGEND        how to read such a line (for each site's instructions)
  judged_before     what the app had judged for a segment BEFORE the person ruled on it
  compare           which of a ruling's fields differ from that, field by field
  ruling_question   the review question whose answer made a ruling, with every option
  seg_images        the capture moments of one segment, labelled, as the evaluator labels them

Every time comes from config.local_dt (the zone the Mac was in then)."""
from __future__ import annotations

import json
import logging

from .seg_models import shown_switch, stored_switch

log = logging.getLogger("prompt_parts")


def t(ts: float, fmt: str = "%H:%M:%S") -> str:
    from .config import local_dt
    return local_dt(ts).strftime(fmt)


def one(v) -> str:
    return " ".join(str(v or "").split())


LOG_LEGEND = """HOW TO READ A LOG LINE. Each computer line is, in order: its mark, with the \
segment's id; the time it began, to the second, in the time zone the person was in that day (a \
"--- weekday date ---" line gives the date wherever it changes); how long the segment lasted in \
all (seconds under two minutes); the app; "site:", "title:" and "url:" in full where the Mac \
recorded them (most apps other than browsers have no title). An "AWAY" line is ten minutes or \
more with no keyboard or mouse input (a locked screen counts as no input). "(toggl entry covering most of it: ...)" names the \
Toggl entry that ran for the largest part of the segment, or says none ran during it.
- A line with NO "judged:" part has not been judged by anyone yet.
- "judged: <content> <focused|unfocused> <switch: continuation|shift|return> ..." is the \
judgment the app holds for that segment. On a shift, ", self-distraction" means the app has it \
as seeking stimulation without a clear intent, and ", next task in the block" as the next task \
under the same task-processing intention. "<continues id=N>" is its thread pointer: the earlier \
segment it continues or returns to. "<hybrid>" marks activity inside a live discussion, \
"<flagged uncertain>" that it is flagged for the person's review. A 'transition' or 'idle' line \
has no focus tag. Segments judged before 2026-11-19 have no focus tag and carry an older switch \
tag - <switch: shift (detour)>, (self_distraction) or (focus_start) - from when the kind of \
shift was one label; no switch tag on such a line is a continuation.
- "// note:" after it is the reasoning stored with that judgment, in full. Text added to a note \
after " | " records what has happened to the judgment since it was made, for example \
"corrected via your feedback on segment N: ..." (a ruling the person made on segment N was \
carried over to this one), "does your correction of segment N apply here? ..." (flagged for \
that reason), "no question: ...", "review question archived unanswered", "spot check: the \
person says ...".
- "RULED BY THE PERSON: ..." marks a segment the person has ruled on themselves, with the fields \
they set and their note. Those fields are what happened. "stands as:" on such a line is the \
judgment in force, their fields over the app's; the app's note on that line may predate their \
ruling.
- Notes in [square brackets] are signals the Mac recorded beside the capture: how the app came \
forward ("arrived by"), how the browser reached the page ("page reached by"), input per minute, \
audio / microphone / camera. They are evidence, not verdicts, and older lines have none (clicks \
and input counts exist from 2026-11-20, page loads from 2026-11-17).
- Indented lines under a segment belong to it: "input by minute" (keys/clicks/scroll \
events/pointer moves for each clock minute, then a, m or c when audio, microphone or camera was \
on for half of it), "no input" (pauses of a minute or more inside it), and "page loads during \
this segment" (every page Chrome loaded while it was in front; the title and url on the line \
itself are those the segment began with).
- "PHONE" lines are apps on the person's iPhone, slotted in by time. Only the app and the timing \
are known of them."""


def log_lines(conn, cfg, lo: float, hi: float, marks: dict | None = None, notes: dict | None = None,
              dated: bool = False, phone: bool = True, keep: set | None = None,
              default_mark: str = "(context id={id}) ") -> tuple[list[str], list[dict]]:
    """The activity log of [lo, hi) as the evaluator reads it (evaluator._log_lines: whole title,
    URL, judgment tags, thread pointer, the person's ruling and note, signals, the whole note,
    per-minute input and page loads), plus the PHONE lines in it. `marks` {segment id: prefix}
    names particular segments; every other line gets `default_mark`. `dated` puts a date line
    wherever the calendar date changes (a prompt that spans days). `keep` limits the output to
    those segment ids (and the phone lines between them); the signals are still worked out
    against the real neighbours. Returns (lines, the segments shown: stats.labelled_segments
    rows, one per real segment)."""
    from . import evaluator as ev, stats
    segs = stats.labelled_segments(conn, lo, hi, phone=False)
    live = [s for s in segs if s["duration"] > 0]
    if not live:
        return [], []
    entries = ev._toggl_lines(conn, lo, hi, None, None)[1]
    mk = {s["id"]: (marks or {}).get(s["id"]) or default_mark.format(id=s["id"]) for s in live}
    rows = list(zip(ev._log_lines(conn, segs, mk, entries, notes=notes), live))
    if keep is not None:
        rows = [(r, s) for r, s in rows if s["id"] in keep]
        if not rows:
            return [], []
        lo, hi = min(s["clip_start"] for _, s in rows), max(s["clip_end"] for _, s in rows)
    out = [r for r, _ in rows]
    if phone:
        try:   # the phone log is another app's database; without it the log simply has no PHONE lines
            from . import phone as _phone
            picks = _phone.pickups(cfg, lo, hi)
            if picks:
                have = {r["seg_key"]: r for r in conn.execute(
                    "SELECT * FROM phone_evals WHERE start >= ? AND start < ?", (lo - 2400, hi + 2400))}
                out += [x for x in ev._phone_lines(conn, picks, set(), have) if lo <= x[0] < hi]
        except Exception:
            log.debug("no phone lines for the log", exc_info=True)
    lines, day = [], None
    for ts, _, text in sorted(out, key=lambda x: (x[0], x[1])):
        d = t(ts, "%a %Y-%m-%d")
        if dated and d != day:
            lines.append(f"--- {d} ---")
            day = d
        lines.append(text)
    return lines, [s for _, s in rows]


def seg_line(conn, cfg, segment_id: int, mark: str | None = None) -> str:
    """One segment's log line with its date in front, for a segment named from elsewhere (what a
    segment continues, the source of a propagated correction). Empty when it does not exist."""
    g = conn.execute("SELECT start, end FROM segments WHERE id=?", (segment_id,)).fetchone()
    if g is None:
        return ""
    lines, _ = log_lines(conn, cfg, g["start"] - 1, (g["end"] or g["start"]) + 1, phone=False, keep={segment_id},
                         default_mark=mark or "(id={id}) ")
    return (t(g["start"], "%a %Y-%m-%d") + " " + lines[0]) if lines else ""


# ---- the app's judgment as it stood before a ruling -------------------------------------------------

def judged_before(conn, segment_id: int, ruled_at: float) -> dict | None:
    """What the app had judged for a segment BEFORE the person ruled on it, as fields: {content,
    switch_label (stored name), antecedent_id, focused, self_distraction, next_task, hybrid,
    kind, uncertain, note, source}. The live seg_evals row when it is older than the ruling.
    When the segment was judged again since (derive_kind and every re-evaluation judge with the
    ruling in hand, 102 of 326 rulings), the answer in the transcript of the last evaluation run
    before the ruling. None when neither exists. evaluator._app_judged gives the same thing as a
    sentence; propagation and the rule drafts need the fields, to say which ones the person changed."""
    e = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    b = lambda v: None if v is None else bool(v)
    if e is not None and e["created"] <= ruled_at:
        return {"content": e["content"], "switch_label": e["switch_label"] or "continuation",
                "antecedent_id": e["antecedent_id"], "focused": b(e["focused"]),
                "self_distraction": b(e["self_distraction"]), "next_task": b(e["next_task"]), "hybrid": bool(e["hybrid"]),
                "kind": e["interruption_kind"], "uncertain": bool(e["uncertain"]), "note": e["rationale"] or "",
                "source": "the judgment stored then"}
    g = conn.execute("SELECT start FROM segments WHERE id=?", (segment_id,)).fetchone()
    if g is None:
        return None
    for run in conn.execute("""SELECT id, response, created FROM eval_runs WHERE t0 <= ? AND t1 >= ? AND created < ?
                               AND response IS NOT NULL ORDER BY created DESC LIMIT 6""",
                            (g["start"], g["start"], ruled_at)):
        try:
            js = json.loads(run["response"])
            j = next((x for x in (js.get("segments", []) if isinstance(js, dict) else js)
                      if x.get("segment_id") == segment_id), None)
        except (ValueError, AttributeError, TypeError):
            j = None
        if not j:
            continue
        sw = stored_switch(str(j.get("switch") or "").strip().lower()) or "continuation"
        ante = ((j.get("antecedents") or [{}])[0] or {}).get("segment_id") if sw in ("continuation", "return") else None
        return {"content": j.get("content"), "switch_label": sw, "antecedent_id": ante,
                "focused": b(j.get("focused")) if "focused" in j else None,
                "self_distraction": bool(j.get("self_distraction")) if "focused" in j else None,
                "next_task": bool(j.get("next_task")) if "focused" in j else None,
                "hybrid": bool(j.get("hybrid")), "kind": j.get("shift_kind") or j.get("interruption_kind"),
                "uncertain": bool(j.get("uncertain")), "note": j.get("note") or "",
                "source": f"the evaluation run of {t(run['created'], '%Y-%m-%d %H:%M')}, the last before they ruled"}
    return None


def judged_text(j: dict | None, note: bool = True) -> str:
    """judged_before's fields in the words models are given (as evaluator._app_judged writes them)."""
    if j is None:
        return "not on record (no judgment was stored for it before the ruling)"
    bits = [f"content={j['content']}", f"switch={shown_switch(j['switch_label'])}",
            f"continues id={j['antecedent_id']}" if j["antecedent_id"] else None,
            None if j["focused"] is None else ("focused" if j["focused"] else "unfocused"),
            "self-distraction" if j["self_distraction"] else None,
            "next task in the block" if j["next_task"] else None,
            "hybrid (inside a live discussion)" if j["hybrid"] else None,
            f"kind of shift (older label)={j['kind']}" if j["kind"] and j["focused"] is None else None,
            "flagged uncertain" if j["uncertain"] else None]
    return ", ".join(x for x in bits if x) + (f" - “{one(j['note'])}”" if note and j["note"] else "")


def _prev_id(conn, segment_id: int) -> int | None:
    r = conn.execute("""SELECT p.id FROM segments g JOIN segments p ON p.start < g.start AND COALESCE(p.end, p.start) > p.start
                        WHERE g.id=? ORDER BY p.start DESC LIMIT 1""", (segment_id,)).fetchone()
    return r[0] if r else None


def compare(conn, rv, j: dict | None) -> dict:
    """A ruling (seg_reviews row) against the app's judgment from before it, over EVERY field
    the person set: {"differ": [...], "match": [...], "extra": [...]}, each a list of phrases.
    Until 2026-11-20 propagation compared content, switch and focused only, so 289 of 537
    rulings were described to its judge as "The person confirmed the judgment" - 26 of them
    re-pointed the thread (event 539: the model said return with nothing named, the person said
    return to 36540) and 4 changed hybrid. `extra` holds what the app has no field for (the
    focus exception) and fields it had no judgment of."""
    from .stats import _same_thread
    out: dict = {"differ": [], "match": [], "extra": []}
    k = rv.keys()
    g = lambda f: rv[f] if f in k else None
    if j is None:
        return out

    def put(same, text):
        out["match" if same else "differ"].append(text)
    if g("content"):
        put(g("content") == j["content"], f"content (app: {j['content']}; person: {g('content')})")
    sw = g("switch_label")
    if sw:
        put(sw == j["switch_label"], f"switch (app: {shown_switch(j['switch_label'])}; person: {shown_switch(sw)})")
    if g("antecedent_id") and (sw or j["switch_label"]) in ("continuation", "return"):
        ja = j["antecedent_id"]
        if ja is None and j["switch_label"] == "continuation":
            ja = _prev_id(conn, rv["segment_id"])     # a continuation stored without a pointer continues the line before it
        same = j["switch_label"] in ("continuation", "return") and ja is not None and _same_thread(conn, g("antecedent_id"), ja)
        put(same, f"what it continues or returns to (app: {'id=' + str(ja) if ja else 'nothing named'}; person: id={g('antecedent_id')})")
    if g("hybrid") is not None:
        put(bool(g("hybrid")) == j["hybrid"], f"hybrid (app: {'yes' if j['hybrid'] else 'no'}; person: {'yes' if g('hybrid') else 'no'})")
    for f, yes, no in (("focused", "focused", "unfocused"), ("self_distraction", "self-distraction", "not a self-distraction"),
                       ("next_task", "next task in the block", "not the next task")):
        if g(f) is None:
            continue
        mine = yes if g(f) else no
        if j[f] is None:
            out["extra"].append(f"{mine} (the app's judgment then had no such field)")
        else:
            put(bool(g(f)) == j[f], f"{f.replace('_', ' ')} (app: {yes if j[f] else no}; person: {mine})")
    if g("interruption_kind") and g("focused") is None:
        from .evaluator import kind_of
        jk = j["kind"] if j["focused"] is None else kind_of(j["switch_label"], j["focused"], j["self_distraction"])
        put(g("interruption_kind") == jk, f"kind of shift, the older single label (app: {jk or 'none'}; person: {g('interruption_kind')})")
    if g("focus_exception"):
        out["extra"].append("counts as independent focus by their explicit exception (the app has no such judgment; "
                            "only the person can make this ruling)")
    return out


# ---- the review question behind a ruling -----------------------------------------------------------

def option_text(o: dict) -> str:
    """The judgment an answer option writes, in the words models are given."""
    from .summaries import verdict_text
    if o.get("phone"):
        return "a ruling on the phone segment: " + ", ".join(f"{a}={b}" for a, b in o["phone"].items())
    if o.get("sample_verdict"):
        return ("agrees with the app's judgment on that one axis" if o["sample_verdict"] == "agree"
                else verdict_text(o) or "says the app's judgment on that one axis is wrong")
    return verdict_text(o) or "(sets no field)"


def ruling_question(conn, rv):
    """(question row, its options, index of the chosen one) for the review question whose answer
    made this ruling, or None. An option can edit another segment than the one its question is
    anchored on, so the match is on the answer time and either segment (as in the evaluator's
    list of rulings)."""
    for q in conn.execute("""SELECT * FROM review_questions WHERE status='answered' AND answered_at IS NOT NULL
                             AND ABS(answered_at - ?) < 5""", (rv["created"],)):
        try:
            opts = json.loads(q["options"] or "[]")
        except ValueError:
            opts = []
        idx = q["answer_index"]
        if idx is None or not (0 <= idx < len(opts)):
            idx = next((i for i, o in enumerate(opts) if o.get("label") == q["answer"]), None)
        o = opts[idx] if idx is not None else {}
        if rv["segment_id"] in (q["segment_id"], o.get("segment_id")):
            return q, opts, idx
    return None


def question_text(q, opts: list, idx: int | None, indent: str = "  ") -> str:
    """A review question as the person saw it: context, question, every option with the judgment
    it writes, which one they chose, and their note."""
    lines = [f"{indent}review question #{q['id']} about segment id={q['segment_id']} (raised by: {q['source']}), "
             f"asked {t(q['created'], '%Y-%m-%d %H:%M')}"
             + (f", answered {t(q['answered_at'], '%Y-%m-%d %H:%M')}" if q["answered_at"] else ""),
             f"{indent}context shown: {one(q['context'])}", f"{indent}question: {one(q['question'])}", f"{indent}options:"]
    for i, o in enumerate(opts):
        on = f" (edits segment id={o['segment_id']})" if o.get("segment_id") not in (None, q["segment_id"]) else ""
        lines.append(f"{indent}  {i + 1}. “{one(o.get('label'))}”{on} -> {option_text(o)}"
                     + ("   <== THEIR ANSWER" if i == idx and q["status"] == "answered" else ""))
    if q["status"] == "unsure":
        lines.append(f"{indent}they answered: I can't tell")
    elif q["status"] == "none":
        lines.append(f"{indent}they answered: NONE OF THE ABOVE (no option says what happened)")
    elif q["status"] in ("dismissed", "archived", "declined"):
        lines.append(f"{indent}status: {q['status']} (not answered)")
    if one(q["note"]):
        lines.append(f"{indent}their note: “{one(q['note'])}”")
    return "\n".join(lines)


# ---- screenshots -------------------------------------------------------------------------------------

def seg_images(mom: dict, s, what: str, max_moments: int | None = None, both: bool = True) -> list[dict]:
    """Image entries {ts, path, label} for one segment's capture moments, chosen and labelled as
    the evaluator does it (_seg_moments: all up to three, then about one per two minutes;
    _moment_images: the active display, and the other display when `both`). `s` needs id, app,
    start and end (or end_eff); `mom` is shots.moments() for it."""
    from . import evaluator as ev, shots
    end = (s["end_eff"] if "end_eff" in s.keys() else None) or s["end"] or s["start"]
    ms = ev._seg_moments(mom, {"id": s["id"], "start": s["start"], "end_eff": end})
    if max_moments is not None:
        ms = shots.spread(ms, max_moments)
    return [im for m in ms for im in ev._moment_images(m, what, s["app"] or "", both=both)]


def fit(images: list[dict], n: int) -> list[dict]:
    """At most n of the images, evenly spread, the first and the last kept."""
    from . import shots
    return shots.spread(images, n) if len(images) > n else images


def image_blocks(images: list[dict]) -> tuple[list[dict], list[dict]]:
    from . import evaluator as ev
    return ev._image_content(images)


def similar_rulings(conn, segs: list[dict]) -> str:
    """The person's past rulings on segments like these (evaluator._similar_precedents: same
    site, or window titles sharing most of their words), each as one full line. Empty when there
    are none or the evaluator cannot be built (no credentials)."""
    try:
        from . import evaluator as ev
        return ev._get()._similar_precedents(conn, segs, skip_recent=False)
    except Exception:
        log.exception("could not gather the similar past rulings")
        return ""

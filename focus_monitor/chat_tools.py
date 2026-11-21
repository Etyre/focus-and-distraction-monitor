"""Read tools for the chats (2026-11-20): get_screenshots, get_evaluation, get_segment_detail, get_log.

Until now every tool of the day chat WROTE (set a judgment, rule on a span, open a question,
propose a rule) and the evaluator chat had none that read. When the person disputed one block,
the model could not look at it: the day chat held six screenshots sampled from a median of 352
segments with a capture, and the replayed evaluation run behind an evaluator chat had no image
of the disputed segment in 12 of 27 chat turns (segment 33153: "What did I say to Atlas when I
returned to it?" - "I can't tell you - I don't have a screenshot of that Atlas segment.").

Nothing here writes. A result is text, or text and images; images stay within SHOTS_PER_CALL
per call. `segment_detail` and `probs_text` are also what the question writer shows of a doubt.
Earlier turns' tool calls are kept with the reply (chat_messages.proposal, key "tool_trace")
and replayed on later turns (replay), so the model still has what it looked up."""
from __future__ import annotations

import datetime as dt
import json
import logging

from . import prompt_parts as pp
from .seg_models import shown_switch

log = logging.getLogger("chat_tools")

SHOTS_PER_CALL = 12          # images in one get_screenshots result
REPLAY_IMAGES = 24           # images of earlier turns' results sent again on a later turn (newest first)
TURN_TEXT_MAX = 600_000      # characters of tool results in one turn; a stored evaluator prompt alone can be 100k+
TRACE_KEY = "tool_trace"

_DAY = {"type": ["string", "null"], "description": "Logical day YYYY-MM-DD (a day runs from the day-start hour to the same hour next morning). null = the day this conversation is about."}

READ_TOOLS = [
    {"name": "get_screenshots",
     "description": ("LOOK at what was on screen. Returns the captures of one block (segment_id), or of a stretch of time "
                     "(start_hhmm to end_hhmm), each labelled with its block id, time and app; both displays when there "
                     f"were two. At most {SHOTS_PER_CALL} images per call, spread evenly; the result says how many captures "
                     "exist, so narrow the time range to see the rest. Use it before saying anything about what a block "
                     "showed that the log does not state. which='seen_by_evaluator' returns instead the images the "
                     "evaluator was shown when it judged the block. Read-only."),
     "input_schema": {"type": "object", "properties": {
         "segment_id": {"type": ["integer", "null"], "description": "The block (#id / id=N in the log)."},
         "start_hhmm": {"type": ["string", "null"], "description": "Start of a time range, HH:MM or HH:MM:SS, in the log's own times."},
         "end_hhmm": {"type": ["string", "null"]},
         "day": _DAY,
         "which": {"type": ["string", "null"], "enum": ["captures", "seen_by_evaluator", None]}},
         "required": []}},
    {"name": "get_evaluation",
     "description": ("The evaluation run behind a block's current judgment: when it was made and by which model, the "
                     "activity log and day summary the evaluator was given (its prompt, in full), its whole answer, and "
                     "the list of screenshots it was shown. Use it to see WHY a block was judged as it was and what the "
                     "evaluator could and could not see. Read-only."),
     "input_schema": {"type": "object", "properties": {"segment_id": {"type": "integer"}}, "required": ["segment_id"]}},
    {"name": "get_segment_detail",
     "description": ("Everything on record about one block: its full log line with the signals the Mac recorded, the "
                     "app's probabilities for each label and its candidates for what the block continues, the person's "
                     "ruling and note and what the app had judged before it, corrections carried over from other "
                     "rulings, every review question asked about it with the answer, and earlier chat about it. Read-only."),
     "input_schema": {"type": "object", "properties": {"segment_id": {"type": "integer"}}, "required": ["segment_id"]}},
    {"name": "get_log",
     "description": ("The activity log of another day, or of a time range, in the same line format as the log you have: "
                     "every block with its judgment, note, rulings and signals. Give a day for the whole of it, or a day "
                     "with start_hhmm and end_hhmm for part of it. Read-only."),
     "input_schema": {"type": "object", "properties": {
         "day": _DAY, "start_hhmm": {"type": ["string", "null"]}, "end_hhmm": {"type": ["string", "null"]}},
         "required": []}},
]
READ_NAMES = {x["name"] for x in READ_TOOLS}


def clock(day: dt.date, hhmm: str) -> float | None:
    """A clock time of a logical day as a timestamp, in the zone the Mac was in that day (the
    day chat used the Mac's zone of today, three hours off for days before the move east)."""
    from .config import day_bounds, local_dt, zone_at
    try:
        parts = [int(x) for x in str(hhmm).strip().split(":")[:3]]
        h, m, s = (parts + [0, 0])[:3]
    except (ValueError, AttributeError):
        return None
    t0 = day_bounds(day)[0]
    naive = local_dt(t0).replace(hour=h % 24, minute=m, second=s, microsecond=0)
    for _ in range(2):   # after midnight belongs to the same logical day
        z = zone_at(t0)
        ts = naive.replace(tzinfo=z).timestamp() if z is not None else naive.timestamp()
        if ts >= t0 - 1:
            return ts
        naive += dt.timedelta(days=1)
    return ts


def _day_of(args: dict, default: dt.date) -> dt.date:
    try:
        return dt.date.fromisoformat(args["day"]) if args.get("day") else default
    except (ValueError, TypeError):
        raise ValueError("day must be YYYY-MM-DD")


def _range(args: dict, day: dt.date) -> tuple[float, float] | None:
    from .config import day_bounds
    a, b = args.get("start_hhmm"), args.get("end_hhmm")
    if not a and not b:
        return None
    t0, t1 = day_bounds(day)
    lo = clock(day, a) if a else t0
    hi = clock(day, b) if b else t1
    if lo is None or hi is None:
        raise ValueError("start_hhmm / end_hhmm must be HH:MM or HH:MM:SS")
    if hi <= lo:
        hi += 86400
    return lo, hi


def probs_text(conn, cfg, segment_id: int, indent: str = "  ") -> str:
    """The alternatives the app weighed for a segment: each model's probabilities for content
    and switch, the combined ones, and the ranked candidates for what it continues, each with
    its log line. Empty when nothing was stored."""
    def dist(js, names=lambda k: k):
        try:
            d = json.loads(js) if isinstance(js, str) else (js or {})
        except ValueError:
            return ""
        return ", ".join(f"{names(k)} {v:.0%}" for k, v in sorted(d.items(), key=lambda x: -x[1]) if v >= 0.005)
    out = []
    who = {"claude": "the evaluator", "heuristic": "the rules model", "learned": "the model trained on their reviews"}
    for p in conn.execute("SELECT * FROM seg_predictions WHERE segment_id=? ORDER BY model", (segment_id,)):
        out.append(f"{indent}{who.get(p['model'], p['model'])}: content: {dist(p['p_content'])}; switch: {dist(p['p_switch'], shown_switch)}")
    e = conn.execute("SELECT probs FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    try:
        pr = json.loads(e["probs"]) if e is not None and e["probs"] else {}
    except ValueError:
        pr = {}
    if pr.get("content") or pr.get("switch"):
        out.append(f"{indent}combined (the vote stored with the judgment): content: {dist(pr.get('content'))}; "
                   f"switch: {dist(pr.get('switch'), shown_switch)}")
    ap = pr.get("antecedent") or {}
    if ap:
        tot = sum(ap.values()) or 1.0
        out.append(f"{indent}the evaluator's candidates for what it continues or returns to, given that it continues "
                   "something (its confidence in each):")
        for sid, p in sorted(ap.items(), key=lambda x: -x[1]):
            line = pp.seg_line(conn, cfg, int(sid)) if str(sid).lstrip("-").isdigit() else ""
            out.append(f"{indent}  {p / tot:.0%}  " + (line or f"id={sid} (not on record)"))
    return "\n".join(out)


def segment_detail(conn, cfg, segment_id: int) -> str:
    from .summaries import verdict_text
    g = conn.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
    if g is None:
        return f"error: no block with id {segment_id}"
    end = g["end"] or g["start"]
    out = [f"BLOCK id={segment_id}: {pp.t(g['start'], '%a %Y-%m-%d %H:%M:%S')} to {pp.t(end)}"
           + ("" if g["end"] else " (still open)")]
    lines, segs = pp.log_lines(conn, cfg, g["start"] - 1, end + 1, keep={segment_id}, phone=False, default_mark="(id={id}) ")
    out += lines or ["(no log line: the block has no length)"]
    e = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    if e is not None:
        out.append(f"The app's stored judgment was written {pp.t(e['created'], '%Y-%m-%d %H:%M')}"
                   + (f" by evaluation run #{e['run_id']} (get_evaluation shows it)" if e["run_id"] else " (no evaluation run on record for it)") + ".")
        if e["antecedent_id"]:
            out.append("What it is judged to continue: " + (pp.seg_line(conn, cfg, e["antecedent_id"]) or f"id={e['antecedent_id']}"))
    else:
        out.append("The app has no judgment of this block.")
    pt = probs_text(conn, cfg, segment_id)
    if pt:
        out.append("Probabilities the app's models gave:\n" + pt)
    rv = conn.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
    if rv is not None:
        out.append(f"THE PERSON'S RULING (made {pp.t(rv['created'], '%Y-%m-%d %H:%M')}"
                   + (f", through chat reply #{rv['chat_message_id']}" if rv["chat_message_id"] else "") + "): "
                   + (verdict_text(rv) or "(no field set)") + (f"; their note: “{pp.one(rv['note'])}”" if pp.one(rv["note"]) else ""))
        if rv["antecedent_id"]:
            out.append("  what they say it continues: " + (pp.seg_line(conn, cfg, rv["antecedent_id"]) or f"id={rv['antecedent_id']}"))
        j = pp.judged_before(conn, segment_id, rv["created"])
        out.append(f"  what the app had judged before they ruled ({j['source'] if j else 'no record'}): " + pp.judged_text(j))
        cmp_ = pp.compare(conn, rv, j)
        if j is not None:
            out.append("  fields of their ruling that differ from it: " + ("; ".join(cmp_["differ"]) or "none")
                       + ". Fields that match: " + ("; ".join(cmp_["match"]) or "none") + ".")
    else:
        out.append("The person has not ruled on this block.")
    for p in conn.execute("SELECT * FROM seg_propagated WHERE segment_id=? ORDER BY created", (segment_id,)):
        out.append(f"Carried over from their ruling on segment id={p['from_segment_id']} ({pp.t(p['created'], '%Y-%m-%d %H:%M')}): "
                   + ("flagged: does that ruling apply here?" if p["verdict"] == "uncertain" else
                      "applied: " + (verdict_text(p) or "(no field)")) + f" - “{pp.one(p['note'])}”")
    n = conn.execute("SELECT COUNT(*), SUM(verdict='applies') FROM seg_propagated WHERE from_segment_id=?", (segment_id,)).fetchone()
    if n[0]:
        out.append(f"Their ruling on this block was carried to {n[1] or 0} other block(s) and raised a doubt on {n[0] - (n[1] or 0)}.")
    qs = conn.execute("SELECT * FROM review_questions WHERE (segment_id=? OR options LIKE ?) AND status != 'declined' ORDER BY created",
                      (segment_id, f'%"segment_id": {segment_id},%')).fetchall()
    for q in qs:
        try:
            opts = json.loads(q["options"] or "[]")
        except ValueError:
            opts = []
        out.append(f"Review question ({q['status']}):\n" + pp.question_text(q, opts, q["answer_index"]))
    nd = conn.execute("SELECT COUNT(*) FROM review_questions WHERE segment_id=? AND status='declined'", (segment_id,)).fetchone()[0]
    if nd:
        out.append(f"{nd} doubt(s) about this block were looked at by the question writer and no question was asked.")
    if not qs and not nd:
        out.append("No review question has been asked about this block.")
    chat = conn.execute("SELECT * FROM chat_messages WHERE kind='evaluator' AND ref=? ORDER BY created, id", (str(segment_id),)).fetchall()
    if chat:
        out.append("Chat about this block (the person with the evaluator), in order:")
        for m in chat:
            who = "the person" if m["role"] == "user" else "the evaluator"
            prop = ""
            try:
                pj = (json.loads(m["proposal"]) if m["proposal"] else {}).get("judgment")
                if pj:
                    prop = f"  [proposed judgment: “{pp.one(pj.get('label'))}” -> {verdict_text(pj) or 'no field'}]"
            except ValueError:
                pass
            out.append(f"  {pp.t(m['created'], '%Y-%m-%d %H:%M')} {who}: {m['text']}{prop}")
    from . import shots
    ms = shots.moments(conn, [segment_id]).get(segment_id) or []
    out.append((f"Screenshots: {len(ms)} capture moment(s) on disk, at " + ", ".join(pp.t(m["ts"]) for m in ms)
                + ". get_screenshots shows them.") if ms else f"Screenshots: none ({shots.why_none(conn, segment_id)}).")
    return "\n".join(out)


def _screenshots(conn, cfg, args: dict, day: dt.date):
    from . import evaluator as ev, shots
    from .classifiers.claude_vision import _b64
    sid = args.get("segment_id")
    if args.get("which") == "seen_by_evaluator":
        if sid is None:
            return "error: which='seen_by_evaluator' needs a segment_id"
        e = conn.execute("SELECT run_id FROM seg_evals WHERE segment_id=?", (sid,)).fetchone()
        run = conn.execute("SELECT * FROM eval_runs WHERE id=?", (e["run_id"],)).fetchone() if e is not None and e["run_id"] else None
        if run is None:
            return f"no evaluation run is on record for block id={sid}"
        meta = json.loads(run["shots"] or "[]")
        if not meta:
            return f"evaluation run #{run['id']} was shown no screenshot at all"
        own = [m for m in meta if f"id={sid} " in m["label"] or f"id={sid})" in m["label"]]
        chosen = (own + [m for m in meta if m not in own])[:SHOTS_PER_CALL]
        blocks: list = [{"type": "text", "text": f"Evaluation run #{run['id']} ({pp.t(run['created'], '%Y-%m-%d %H:%M')}) was shown {len(meta)} "
                         f"screenshot(s), {len(own)} of them of block id={sid}. Attached: {len(chosen)}"
                         + (f" (those of the block first; the run's other {len(meta) - len(chosen)} are not attached)" if len(chosen) < len(meta) else "")
                         + ". Labels are the ones the evaluator read."}]
        for m in chosen:
            b = _b64(m["path"])
            blocks.append({"type": "text", "text": m["label"] + ("" if b else " [file no longer on disk]")})
            if b:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        return blocks
    if sid is not None:
        g = conn.execute("SELECT * FROM segments WHERE id=?", (int(sid),)).fetchone()
        if g is None:
            return f"error: no block with id {sid}"
        segs = [g]
        scope = f"block id={sid}"
    else:
        rng = _range(args, day)
        if rng is None:
            return "error: give a segment_id, or start_hhmm and end_hhmm"
        segs = conn.execute("SELECT * FROM segments WHERE idle=0 AND start < ? AND COALESCE(end, start) > ? ORDER BY start",
                            (rng[1], rng[0])).fetchall()
        scope = f"{pp.t(rng[0], '%a %Y-%m-%d %H:%M:%S')} to {pp.t(rng[1])}"
        if not segs:
            return f"no block was recorded between {scope}"
    mom = shots.moments(conn, [g["id"] for g in segs])
    by_app = {g["id"]: g["app"] or "" for g in segs}
    ms = [m for g in segs for m in mom.get(g["id"]) or []]
    if sid is None:
        ms = [m for m in ms if rng[0] - 1 <= m["ts"] <= rng[1] + 1] or ms
    if not ms:
        why = "; ".join(f"id={g['id']}: {shots.why_none(conn, g['id'])}" for g in segs[:20])
        return f"no capture on disk for {scope} ({why})"
    ms.sort(key=lambda m: m["ts"])
    shown = shots.spread(ms, SHOTS_PER_CALL)
    what = lambda m: f"block ({pp.t(m['ts'], '%a %Y-%m-%d')})"
    images = [im for m in shown for im in ev._moment_images(m, what(m), by_app.get(m["segment_id"], ""))]
    if len(images) > SHOTS_PER_CALL:   # the other display only when every moment's active display fits beside it
        images = [im for m in shown for im in ev._moment_images(m, what(m), by_app.get(m["segment_id"], ""), both=False)]
    blocks, meta = pp.image_blocks(images[:SHOTS_PER_CALL])
    head = (f"{len(ms)} capture moment(s) on disk for {scope}, at " + ", ".join(pp.t(m["ts"]) for m in ms)
            + f". Attached: {len(meta)} image(s) of {len(shown)} moment(s)"
            + (", spread evenly; ask for a narrower time range to see the others" if len(shown) < len(ms) else "") + ".")
    return [{"type": "text", "text": head}] + blocks


def _evaluation(conn, cfg, sid: int) -> str:
    g = conn.execute("SELECT * FROM segments WHERE id=?", (sid,)).fetchone()
    if g is None:
        return f"error: no block with id {sid}"
    e = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (sid,)).fetchone()
    if e is None:
        return f"block id={sid} has not been judged"
    run = conn.execute("SELECT * FROM eval_runs WHERE id=?", (e["run_id"],)).fetchone() if e["run_id"] else None
    if run is None or not run["prompt"]:
        return (f"No transcript is on record for the judgment of block id={sid} (judged before transcripts were kept, or by "
                "the small local models). What is stored: " + pp.judged_text(pp.judged_before(conn, sid, 1e12)))
    meta = json.loads(run["shots"] or "[]")
    earlier = conn.execute("SELECT COUNT(*) FROM eval_runs WHERE t0 <= ? AND t1 >= ? AND response IS NOT NULL AND id != ?",
                           (g["start"], g["start"], run["id"])).fetchone()[0]
    try:
        js = json.loads(run["response"] or "{}")
        mine = next((x for x in (js.get("segments", []) if isinstance(js, dict) else js) if x.get("segment_id") == sid), None)
    except (ValueError, AttributeError, TypeError):
        mine = None
    cur = pp.judged_text(pp.judged_before(conn, sid, 1e12))
    return "\n".join([
        f"EVALUATION RUN #{run['id']} behind the current judgment of block id={sid}: made {pp.t(run['created'], '%a %Y-%m-%d %H:%M')}"
        f" by {run['model'] or 'the evaluator model'}, covering {pp.t(run['t0'], '%Y-%m-%d %H:%M:%S')} to {pp.t(run['t1'])}, {run['n_segments']} segment(s)."
        + (f" {earlier} other run(s) have covered this block's start time (earlier judgments or re-judgments)." if earlier else ""),
        "Its answer for this block: " + (json.dumps(mine, ensure_ascii=False) if mine else "(this block is not in the run's answer)"),
        "The judgment stored now (it can differ from that answer: a ruling carried over from another block, or the app's "
        "own bookkeeping, may have changed the row since): " + cur,
        f"Screenshots it was shown ({len(meta)}; get_screenshots with which='seen_by_evaluator' attaches them):",
        *[f"  {m['label']}" for m in meta],
        "Its instructions (the system prompt) are not repeated here: they were the person's rules document, their "
        "description of themselves and, in runs since 2026-11-20, their rulings up to then.",
        "THE PROMPT IT WAS GIVEN, in full:", run["prompt"] or "",
        "ITS WHOLE ANSWER:", run["response"] or "(none stored)"])


def _log(conn, cfg, args: dict, day: dt.date) -> str:
    from .config import day_bounds
    rng = _range(args, day) or day_bounds(day)
    lines, segs = pp.log_lines(conn, cfg, rng[0], rng[1], default_mark="#{id} ", dated=True)
    if not lines:
        return f"nothing was recorded between {pp.t(rng[0], '%a %Y-%m-%d %H:%M')} and {pp.t(rng[1], '%a %Y-%m-%d %H:%M')}"
    return (f"Activity log, {pp.t(rng[0], '%a %Y-%m-%d %H:%M:%S')} to {pp.t(rng[1], '%a %Y-%m-%d %H:%M:%S')} "
            f"({len(segs)} block(s); a block that began before the range shows its real start):\n" + "\n".join(lines))


def run(conn, cfg, name: str, args: dict, day: dt.date):
    """Run one read tool. Returns the tool_result content: a string, or a list of text and image
    blocks. Never raises: an error comes back as text the model can act on."""
    try:
        d = _day_of(args, day)
        if name == "get_screenshots":
            return _screenshots(conn, cfg, args, d)
        if name == "get_evaluation":
            return _evaluation(conn, cfg, int(args["segment_id"]))
        if name == "get_segment_detail":
            return segment_detail(conn, cfg, int(args["segment_id"]))
        if name == "get_log":
            return _log(conn, cfg, args, d)
        return f"error: unknown tool {name}"
    except (KeyError, ValueError, TypeError) as e:
        return f"error: {e}"
    except Exception as e:
        log.exception("read tool %s failed", name)
        return f"error: {type(e).__name__}: {e}"


# ---- keeping and replaying what a turn looked up ---------------------------------------------------

def size(result) -> int:
    return len(result) if isinstance(result, str) else sum(len(b.get("text", "")) for b in result if b.get("type") == "text")


def trace_entry(use_id: str, name: str, args: dict, result) -> dict:
    """One tool call as it is stored with the reply. Images are stored by file name and label,
    not as data: the files are on disk."""
    if isinstance(result, str):
        return {"id": use_id, "name": name, "input": args, "result": result}
    texts = [b["text"] for b in result if b.get("type") == "text"]
    n_img = sum(1 for b in result if b.get("type") == "image")
    return {"id": use_id, "name": name, "input": args, "result": texts[0] if texts else "",
            "labels": texts[1:], "images": n_img, "args_for_images": args}


def replay(conn, cfg, traces: list[tuple[int, list]], day: dt.date) -> dict:
    """{message id: [api messages]} rebuilding each earlier reply's tool rounds from its stored
    trace: the assistant's tool calls and the results it got. Before 2026-11-20 only the text of
    earlier turns was sent again, so on the next message the model no longer had the screenshots
    or the evaluation it had just read, nor the record of which edits its own tools had made.
    Images are attached again for the newest results up to REPLAY_IMAGES; an older image result
    is replaced by its labels and a line saying the images are not re-sent."""
    budget = REPLAY_IMAGES
    with_images: set = set()
    for mid, rounds in reversed(traces):
        for r_i in range(len(rounds) - 1, -1, -1):
            for c in rounds[r_i].get("calls", []):
                if c.get("images") and c["images"] <= budget:
                    budget -= c["images"]
                    with_images.add((mid, c["id"]))
    out = {}
    for mid, rounds in traces:
        msgs = []
        for rnd in rounds:
            calls = rnd.get("calls") or []
            if not calls:
                continue
            content = ([{"type": "text", "text": rnd["said"]}] if rnd.get("said") else []) + [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("input") or {}} for c in calls]
            results = []
            for c in calls:
                res = c.get("result") or ""
                if c.get("images"):
                    again = run(conn, cfg, c["name"], c.get("args_for_images") or {}, day) if (mid, c["id"]) in with_images else None
                    if isinstance(again, list):
                        res = again
                    else:
                        res = (res + "\n" + "\n".join(c.get("labels") or [])
                               + f"\n[{c['images']} image(s) were attached here in that turn; they are not sent again. "
                               "Call get_screenshots again to look at them.]")
                results.append({"type": "tool_result", "tool_use_id": c["id"], "content": res})
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": results})
        out[mid] = msgs
    return out


def stored_traces(conn, messages: list[dict]) -> list[tuple[int, list]]:
    """(message id, trace) for the earlier assistant messages of a thread that have one. The
    dashboard sends each message's id back with the thread."""
    out = []
    for m in messages:
        if m.get("role") != "assistant" or not m.get("id"):
            continue
        r = conn.execute("SELECT proposal FROM chat_messages WHERE id=? AND role='assistant'", (m["id"],)).fetchone()
        try:
            tr = (json.loads(r["proposal"]) if r is not None and r["proposal"] else {}).get(TRACE_KEY)
        except ValueError:
            tr = None
        if tr:
            out.append((int(m["id"]), tr))
    return out

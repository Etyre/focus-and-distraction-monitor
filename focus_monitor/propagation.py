"""Feedback propagation: when the user corrects a mislabeled segment, a judgment role
considers whether that feedback applies to similar segments from the past several days -
re-labeling the ones where it clearly does, flagging for review the ones where it is
uncertain, and leaving the rest. Spans re-derive automatically from the updated judgments.

User rule (2026-09-02): "when I correct something that's mislabeled, that should spin up a
process that considers if that feedback is relevant to other segments from the past several
days, and recomputes them. If the model is uncertain whether the feedback applies to a
particular segment, it should flag that segment for review."
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import Config, load_config
from .seg_models import CONTENTS, KINDS, SWITCHES, shown_switch, stored_switch

log = logging.getLogger("propagation")

# What the judge is shown (2026-11-20; Eli: "all the model calls are given all the information that
# could possibly be relevant for their judgement and nothing is left out"). Until then a call was
# four lines about the ruling, one screenshot, and the 80 NEWEST segments of the same app, one
# short line each. Measured on the 539 events so far: the cap cut in 481 (median pool 948); for
# Chrome every Chrome segment qualified, so a median of 10% of the list was the ruled segment's
# site (event 34, x.com: 4 of 261 same-site candidates shown); in 219 of 441 recent events all 80
# were LATER than the ruled segment, because the window was counted back from the time of
# processing; 98 rulings set a thread pointer that was never printed, and 289 were called
# confirmations on three fields (event 539 re-pointed a return and was described as "confirmed");
# 46% of the candidates' notes were cut at 70 characters, which hid the marks other rulings had
# left on them.
LOOKBACK_DAYS = 4          # candidates come from this many days before AND after the ruled segment
SITTING_S = 30 * 60        # same app or site this close to the ruled segment is its own sitting: always shown
TIMELINE_S = 20 * 60       # log shown on each side of the ruled segment
PER_CALL = 40              # candidates per call: each gets its own lines and a screenshot within the image budget
MAX_CALLS = 6              # calls per ruling, so at most 240 candidates; the rest (the least similar) are not judged
WEAK_FILL = 80             # "same app, nothing else in common" candidates only fill up to this many in all
RULED_IMAGES_MAX = 20      # of a call's image budget (evaluator.IMAGE_BUDGET), for the ruled segment itself
MAX_TOKENS = 16000         # 40 verdicts with notes, and thinking counts against it (6,000 before, for 80)


def _foc(row, unset: str = "?") -> str:
    """A row's focused/unfocused judgment in words (rows judged before the redesign have none)."""
    if row is None or "focused" not in row.keys() or row["focused"] is None:
        return unset
    return "focused" if row["focused"] else "unfocused"


class PropVerdict(BaseModel):
    segment_id: int
    verdict: str = Field(description="'applies' (the ruling clearly covers this segment too), 'does_not_apply', or 'uncertain' (plausible but the person should decide - it will be flagged for their review).")
    content: str | None = Field(default=None, description="When applies: the content label the ruling gives. null leaves it as judged.")
    switch_label: str | None = Field(default=None, description="When applies and the switch judgment changes: continuation, shift (attention moved to something new), or return.")
    focused: bool | None = Field(default=None, description="When applies and the focus judgment changes: true = attention was focused in the segment, false = unfocused. null leaves it as judged.")
    self_distraction: bool | None = Field(default=None, description="Only when the ruling itself set self-distraction, and only for a candidate that is (or becomes) a shift or a return: true = sought out stimulation without a clear intent, false = went there on purpose. null leaves it as judged.")
    next_task: bool | None = Field(default=None, description="Only when the ruling itself set next-task, and only for a candidate that is (or becomes) a shift: true = the next task under the same task-processing intention. null leaves it as judged.")
    note: str = Field(description="One short sentence.")


class PropJudgement(BaseModel):
    verdicts: list[PropVerdict]


SYSTEM = """You are the feedback-propagation judge for a personal attention monitor. The person \
ruled on one segment's judgment: a correction of what the app had judged, or a confirmation of \
it. Decide, for each CANDIDATE segment (same site, a similar window, or the same app, from the \
days around it), whether the same ruling logically applies.

Be conservative in both directions: mark 'applies' only when the candidate is clearly the same \
kind of case as the ruled one (same activity, same pattern - not merely the same app); mark \
'uncertain' when it plausibly applies but a human should decide; otherwise 'does_not_apply'. \
The person's note, when present, states the general rule - apply the RULE, not surface similarity.

WHAT YOU ARE SHOWN
- THE RULING: the ruled segment's own log line; what the app had judged BEFORE the person ruled \
(the judgment they were looking at, with its note); every field the person set, including what \
they say the segment continues or returns to; which of those fields differ from the app's \
judgment and which match it; their note; and, when the ruling was an answer to a review \
question, that question with the options they did not choose.
- The activity log around the ruled segment, the person's past rulings on similar segments (a \
contrary ruling of theirs on the same kind of segment means this one does not generalise that \
far), and screenshots of the ruled segment.
- Each candidate: why it is listed, its own log line between the two lines before it and the two \
after it, the segment it is judged to continue, and a screenshot where one exists. A candidate's \
CURRENT judgment is the "judged:" part of its own line. A candidate whose note ends in \
"corrected via your feedback on segment N" already carries another ruling of the person's.
A switch judgment is about what a segment was relative to what preceded it, so compare the \
situations (what came before and after), not only the windows.

WHAT A VERDICT CAN WRITE: content, switch_label and focused; self_distraction and next_task only \
when the ruling itself set them. Leave null whatever the ruling says nothing about. A verdict \
cannot carry a thread pointer: which earlier segment a candidate continues is never copied from \
the ruled segment. When the substance of the ruling is WHICH segment this one continues or \
returns to, a candidate in the same situation can only be 'uncertain' (the person is then asked \
about it), unless its switch label itself should change. Nor can it carry the person's explicit \
"counts as independent focus" exception, which they make segment by segment.

{legend}

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        from .config import load_api_key
        load_api_key()
        c = anthropic.Anthropic()
        if not (getattr(c, "api_key", None) or getattr(c, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials")
        _client = c
    return _client


def note_event(conn, segment_id: int) -> None:
    """Queue a user correction for propagation (called by the edit endpoints)."""
    conn.execute("INSERT INTO feedback_events(segment_id, created) VALUES (?,?)",
                 (segment_id, time.time()))



def _retry_later(conn, event_id: int) -> None:
    """Put a feedback event back in the queue after a failed call, at most three times."""
    n = int(db.kv_get(conn, f"prop.fail.{event_id}") or 0) + 1
    db.kv_set(conn, f"prop.fail.{event_id}", str(n))
    if n < 3:
        conn.execute("UPDATE feedback_events SET processed=0 WHERE id=?", (event_id,))


def _apply(conn, rec) -> bool:
    """Write one propagated verdict (a seg_propagated row, or the same fields as a dict) onto the
    segment's stored judgment. False when it changes nothing."""
    sid, src, note = rec["segment_id"], rec["from_segment_id"], rec["note"]
    opt = lambda k: None if k not in rec.keys() or rec[k] is None else bool(rec[k])
    had = (conn.execute("SELECT rationale FROM seg_evals WHERE segment_id=?", (sid,)).fetchone() or [None])[0] or ""
    if rec["verdict"] == "uncertain":
        # A block whose review question the person has already settled (an answer, "can't tell",
        # "none of these") is not flagged again by a later ruling elsewhere. They had looked at it,
        # and the needs-review dot coming back made it look as if the answer had not taken
        # (Eli, 2026-11-21: segment 37204, answered "can't tell" at 04:42, flagged again at 04:47
        # by a ruling on 37283). A ruling that clearly applies still relabels it ("applies" below).
        if conn.execute("SELECT 1 FROM review_questions WHERE segment_id=? AND status IN ('answered','unsure','none')",
                        (sid,)).fetchone():
            return False
        mark = f" | does your correction of segment {src} apply here? {note}"
        conn.execute("UPDATE seg_evals SET uncertain=1, rationale=COALESCE(rationale,'') || ? WHERE segment_id=?",
                     ("" if mark in had else mark, sid))
        return True
    sets, args = [], []
    if rec["content"] in CONTENTS:
        sets.append("content=?"); args.append(rec["content"])
    switch_label = stored_switch(rec["switch_label"])  # tolerate the shown name 'shift'
    focused = None if rec["focused"] is None else bool(rec["focused"])
    cur_row = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (sid,)).fetchone()
    foc = focused if focused is not None else (
        None if cur_row is None or cur_row["focused"] is None else bool(cur_row["focused"]))
    if focused is not None:
        sets.append("focused=?"); args.append(int(focused))
    sw_new = switch_label if switch_label in SWITCHES else (cur_row["switch_label"] if cur_row else None)
    # self_distraction / next_task (2026-11-20): carried only when the ruling itself set them, and
    # they mean something only on a shift (self-distraction also on a return). Before, a ruling
    # "that was me drifting" could reach similar segments only as a bare label.
    sd_in = opt("self_distraction") if sw_new in ("interruption", "return") else None
    nt_in = opt("next_task") if sw_new == "interruption" else None
    if switch_label in SWITCHES:
        sets.append("switch_label=?"); args.append(switch_label)
        if switch_label != "interruption":
            sets.append("next_task=0")
            if switch_label != "return":   # a return keeps its own self-distraction judgment
                sets.append("self_distraction=0")
    sd = bool(cur_row["self_distraction"]) if cur_row is not None and sw_new in ("interruption", "return") else False
    if sd_in is not None:
        sets.append("self_distraction=?"); args.append(int(sd_in))
        sd = sd_in
    if nt_in is not None:
        sets.append("next_task=?"); args.append(int(nt_in))
    if (switch_label in SWITCHES or focused is not None or sd_in is not None) and foc is not None:
        # the older per-shift kind follows from the fields (evaluator.kind_of)
        from .evaluator import kind_of
        sets.append("interruption_kind=?"); args.append(kind_of(sw_new, foc, sd))
    elif switch_label in SWITCHES and switch_label != "interruption":
        sets.append("interruption_kind=NULL")
    if not sets:
        return False
    sets.append("uncertain=0")
    sets.append("rationale = COALESCE(rationale,'') || ?")
    mark = f" | corrected via your feedback on segment {src}: {note}"
    args.append("" if mark in had else mark)   # (a re-judgment that kept the old row would stack the mark)
    args.append(sid)
    conn.execute(f"UPDATE seg_evals SET {', '.join(sets)} WHERE segment_id=?", args)
    return True


def reapply(conn, segment_ids) -> int:
    """Put propagated corrections back on segments the evaluator has just judged again. A new
    judgment replaces the whole stored row, and with it went the correction carried over from
    the person's ruling on a similar segment (about 170 of 1,444 by 2026-11-20). The person's
    ruling outranks the evaluator, so it is applied again - unless that ruling has since been
    cleared."""
    n = 0
    for sid in segment_ids:
        for rec in conn.execute("""SELECT p.* FROM seg_propagated p
                                   JOIN seg_reviews r ON r.segment_id = p.from_segment_id
                                   WHERE p.segment_id=? ORDER BY p.created""", (sid,)).fetchall():
            if rec["verdict"] == "uncertain" and conn.execute(
                    "SELECT 1 FROM review_questions WHERE segment_id=? AND status != 'open' AND COALESCE(answered_at, created) >= ?",
                    (sid, rec["created"])).fetchone():
                continue   # the person has dealt with that doubt already: it is not raised again
            n += int(_apply(conn, rec))
    return n


def _title_words(s: str | None) -> set:
    import re
    return {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}


def _candidates(conn, seg, now: float) -> tuple[list[dict], int]:
    """The segments a ruling might also cover, most similar first, and how many qualified in
    all. Pool: judged segments of the same app or site that the person has not ruled on, within
    LOOKBACK_DAYS either side of the RULED SEGMENT (it was counted back from the time of
    processing: a ruling made on Friday about Monday looked at Tuesday to Friday).
    Ranked in the spirit of evaluator._similar_precedents: the segment's own sitting first
    (same app or site within SITTING_S), then the same site, then window titles sharing most of
    their words, then - for an app that records neither - the same app. "Same app, nothing else
    in common" is weak: such candidates only fill the list up to WEAK_FILL, nearest in time
    first. Each row carries `why` (the reasons it is listed) and `strong`."""
    lo, hi = seg["start"] - LOOKBACK_DAYS * 86400, min(now, seg["start"] + LOOKBACK_DAYS * 86400)
    rows = conn.execute("""
        SELECT g.* FROM segments g JOIN seg_evals e ON e.segment_id = g.id
        LEFT JOIN seg_reviews r ON r.segment_id = g.id
        WHERE r.segment_id IS NULL AND g.id != ? AND g.idle = 0 AND g.start >= ? AND g.start < ?
          AND (g.app = ? OR (g.domain != '' AND g.domain = ?))""",
        (seg["id"], lo, hi, seg["app"], seg["domain"])).fetchall()
    real = lambda d: d if d and d != "newtab" else ""     # an empty site and Chrome's new tab are not sites
    # the app's own name in a title is not a shared word ("... — Agent Coding" ends every title of that app)
    own = _title_words(seg["app"])
    sd, sw = real(seg["domain"]), _title_words(seg["title"]) - own
    titled = conn.execute("SELECT 1 FROM segments WHERE app=? AND (title != '' OR domain != '') LIMIT 1",
                          (seg["app"],)).fetchone() is not None
    out = []
    for c in rows:
        cd, cw = real(c["domain"]), _title_words(c["title"]) - own
        sc, why = 0.0, []
        if sd and cd == sd:
            sc += 4.0
            why.append(f"same site ({sd})")
        shared = sw & cw
        if sw and cw and (sw == cw or (len(shared) >= 2 and len(shared) / len(sw | cw) >= 0.34)):
            sc += 3.0 * len(shared) / len(sw | cw) + (0.5 if c["app"] == seg["app"] else 0.0)
            why.append("same window title" if c["title"] == seg["title"] else "window title shares most of its words")
        if seg["url"] and c["url"] == seg["url"]:
            sc += 1.0
            why.append("same address")
        if not sc and c["app"] == seg["app"] and not titled:
            sc = 2.0
            why.append(f"same app ({seg['app']}, which records no titles or sites)")
        gap = c["start"] - seg["start"]
        near = abs(gap) <= SITTING_S
        if near:
            why.append(f"in the ruled segment's own sitting ({abs(gap) / 60:.0f} min {'before' if gap < 0 else 'after'} it)")
        if not why:
            why.append(f"same app ({seg['app']}), nothing else in common")
        out.append(dict(c, score=sc, near=near, gap=abs(gap), why="; ".join(why), strong=near or sc >= 2.0))
    strong = sorted((c for c in out if c["strong"]), key=lambda c: (not c["near"], -c["score"], c["gap"]))
    weak = sorted((c for c in out if not c["strong"]), key=lambda c: c["gap"])
    picked = strong[:PER_CALL * MAX_CALLS]
    picked += weak[:max(0, WEAK_FILL - len(picked))]
    return picked, len(out)


def _ruling_text(conn, cfg, seg, rv) -> tuple[str, bool]:
    """THE RULING, complete on both sides, and whether it is a pure confirmation. The app's side
    is the judgment as it stood BEFORE the ruling (prompt_parts.judged_before): the live row was
    written after the ruling in 77 of 537 events - derive_kind and every re-evaluation judge with
    the ruling in hand - and 63 of those then read as "the person confirmed"."""
    from . import prompt_parts as pp
    from .summaries import verdict_text
    j = pp.judged_before(conn, seg["id"], rv["created"])
    cmp_ = pp.compare(conn, rv, j)
    confirmed = j is not None and not cmp_["differ"] and not cmp_["extra"] and bool(cmp_["match"])
    did = ("corrected the app's judgment of" if cmp_["differ"] else
           "ruled on" if j is None else
           "confirmed the app's judgment of" if cmp_["match"] else "ruled on")
    lines = [f"THE RULING (made {pp.t(rv['created'], '%a %Y-%m-%d %H:%M')}). The person {did} segment id={seg['id']}:",
             "  " + pp.seg_line(conn, cfg, seg["id"], mark="(the ruled segment, id={id}) "),
             f"What the app had judged before they ruled ({j['source'] if j else 'no record'}): " + pp.judged_text(j)]
    if j is not None and j["antecedent_id"]:
        lines.append("  the segment the app had it continuing or returning to: " + (pp.seg_line(conn, cfg, j["antecedent_id"]) or f"id={j['antecedent_id']} (no longer on record)"))
    lines.append("What the person ruled: " + (verdict_text(rv) or "(no field set)"))
    if rv["antecedent_id"]:
        lines.append("  the segment they say it continues or returns to: " + (pp.seg_line(conn, cfg, rv["antecedent_id"]) or f"id={rv['antecedent_id']} (no longer on record)"))
    lines.append("Their note: " + (f"“{pp.one(rv['note'])}”" if pp.one(rv["note"]) else "(none)"))
    if j is not None:
        lines.append("Of the fields they set, these DIFFER from what the app had judged: " + ("; ".join(cmp_["differ"]) or "none"))
        lines.append("These MATCH what the app had judged: " + ("; ".join(cmp_["match"]) or "none"))
    if cmp_["extra"]:
        lines.append("They also ruled: " + "; ".join(cmp_["extra"]))
    q = pp.ruling_question(conn, rv)
    if q is not None:
        lines.append("The ruling was their answer to this review question:\n" + pp.question_text(*q))
    return "\n".join(lines), confirmed


def _candidate_block(conn, cfg, c, no_img: dict) -> str:
    """One candidate as the judge reads it: why it is listed, its line between the two before and
    the two after, and the segment it is judged to continue. Until 2026-11-20 a candidate was
    one line built from the bare segments row: time without a date, app, 70 characters of title,
    the raw stored labels ('interruption', and the pre-2026-11-19 kind) and 70 characters of note."""
    from . import prompt_parts as pp
    before = [r["id"] for r in conn.execute("SELECT id FROM segments WHERE start < ? AND COALESCE(end, start) > start"
                                            " ORDER BY start DESC LIMIT 2", (c["start"],))]
    after = conn.execute("SELECT id, start, end FROM segments WHERE start > ? AND COALESCE(end, start) > start"
                         " ORDER BY start LIMIT 2", (c["start"],)).fetchall()
    first = conn.execute("SELECT start FROM segments WHERE id=?", (before[-1],)).fetchone()[0] if before else c["start"]
    last = max([(c["end"] or c["start"])] + [(a["end"] or a["start"]) for a in after])
    ids = set(before) | {c["id"]} | {a["id"] for a in after}
    lines, segs = pp.log_lines(conn, cfg, first - 1, last + 1, marks={c["id"]: f">>> CANDIDATE id={c['id']}: "},
                               keep=ids, default_mark="    (context id={id}) ")
    me = next((s for s in segs if s["id"] == c["id"]), {})
    out = [f"CANDIDATE id={c['id']} - {pp.t(c['start'], '%a %Y-%m-%d')}. Listed because: {c['why']}."]
    out += ["  " + x.replace("\n", "\n  ") for x in lines]
    ante = me.get("antecedent_id")
    if ante and ante not in ids:
        out.append("  what it is judged to continue: " + (pp.seg_line(conn, cfg, ante) or f"id={ante} (no longer on record)"))
    if c["id"] in no_img:
        out.append(f"  screenshot: none for this candidate ({no_img[c['id']]}). That says nothing about what it was.")
    return "\n".join(out)


def _prompt(conn, cfg, seg, rv, batch: list[dict], n_batch: int, n_batches: int, total: int, shown: int) -> tuple[list, int]:
    """The user message for one call: screenshots of the ruled segment and of each candidate,
    then the ruling, the log around it, the person's similar past rulings and the candidates.
    Returns (content blocks, number of images)."""
    from . import evaluator as ev, prompt_parts as pp, shots as _shots
    mom = _shots.moments(conn, [seg["id"]] + [c["id"] for c in batch])
    # the ruled segment: every chosen capture moment, both displays while they fit its share
    rwhat = f"the RULED ({pp.t(seg['start'], '%a %Y-%m-%d')}) segment"
    ruled = pp.seg_images(mom, seg, rwhat)
    if len(ruled) > RULED_IMAGES_MAX:
        ruled = pp.fit(pp.seg_images(mom, seg, rwhat, both=False), RULED_IMAGES_MAX)
    # each candidate: its first capture; then, while the budget lasts, the last capture of those that have more
    room = ev.IMAGE_BUDGET - len(ruled)
    cimgs, no_img = [], {}
    what = lambda c: f"CANDIDATE ({pp.t(c['start'], '%a %Y-%m-%d')}) segment"
    for c in batch:
        ms = mom.get(c["id"]) or []
        if not ms:
            no_img[c["id"]] = _shots.why_none(conn, c["id"])
        elif len(cimgs) < room:
            cimgs += ev._moment_images(ms[0], what(c), c["app"] or "", both=False)
        else:
            no_img[c["id"]] = "the call's image allowance was used up"
    for c in batch:
        ms = mom.get(c["id"]) or []
        if len(ms) > 1 and len(cimgs) < room and c["id"] not in no_img:
            cimgs += ev._moment_images(ms[-1], what(c), c["app"] or "", both=False)
    c1, m1 = pp.image_blocks(ruled)
    c2, m2 = pp.image_blocks(cimgs)
    ruling, confirmed = _ruling_text(conn, cfg, seg, rv)
    end = seg["end"] or seg["start"]
    cand_ids = {c["id"] for c in batch}
    marks = {i: f"(candidate id={i}) " for i in cand_ids}
    marks[seg["id"]] = f">>> THE RULED SEGMENT id={seg['id']}: "
    around, _ = pp.log_lines(conn, cfg, seg["start"] - TIMELINE_S, end + TIMELINE_S, marks=marks, dated=True)
    similar = pp.similar_rulings(conn, [dict(seg)])
    text = (ruling
            + f"\n\nSCREENSHOTS attached above: {len(m1)} of the ruled segment"
            + ("" if m1 else f" ({_shots.why_none(conn, seg['id'])})") + f", {len(m2)} of candidates. Each is labelled with its "
            "segment id, date and time."
            + f"\n\nACTIVITY AROUND THE RULED SEGMENT ({TIMELINE_S // 60} minutes each side; candidates of this call that fall in "
            "it are marked):\n" + "\n".join(around)
            + ("\n\n" + similar.replace("- follow these precedents:", "(each is settled; none of these segments is a candidate):")
               if similar else "\n\nThe person has no other ruling on a segment of the same site or window title.")
            + f"\n\nCANDIDATES: {len(batch)} in this call"
            + (f" (call {n_batch} of {n_batches} for this ruling)" if n_batches > 1 else "")
            + f". {total} judged segments of the same app or site within {LOOKBACK_DAYS} days of the ruled segment have no ruling "
            f"of their own; the {shown} most similar are judged"
            + (", these among them" if shown < total or n_batches > 1 else "")
            + ". Each candidate's lines are in time order; \">>> CANDIDATE\" is the candidate, the others are what came before "
            "and after it.\n\n" + "\n\n".join(_candidate_block(conn, cfg, c, no_img) for c in batch)
            + ("\n\nThe person agreed with the model here, so nothing was wrong in this segment. A candidate "
               "that is clearly the same kind of case but is judged differently should get the confirmed "
               "judgment ('applies'); one already judged the same way is 'does_not_apply'."
               if confirmed else "")
            + "\n\nFor every candidate (all of them, by id): does the ruling apply, not apply, or uncertain?")
    return c1 + c2 + [{"type": "text", "text": text}], len(m1) + len(m2)


def process_pending(conn, cfg: Config | None = None, max_events: int = 1) -> int:
    from . import efforts, prompt_parts as pp
    from .classifiers.claude_vision import _rules_doc
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    events = conn.execute("SELECT * FROM feedback_events WHERE processed=0 ORDER BY id LIMIT ?",
                          (max_events,)).fetchall()
    done = 0
    for ev in events:
        # Marked processed here, and un-marked below if the call fails: a failed call is not a
        # propagated correction (event 501 on 2026-11-14 was lost that way).
        conn.execute("UPDATE feedback_events SET processed=1 WHERE id=?", (ev["id"],))
        seg = conn.execute("SELECT * FROM segments WHERE id=?", (ev["segment_id"],)).fetchone()
        rv = conn.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (ev["segment_id"],)).fetchone()
        if not seg or not rv:
            continue
        cands, total = _candidates(conn, seg, time.time())
        if not cands:
            continue
        system = SYSTEM.format(rules=_rules_doc(), about_me=cfg.about_me.strip() or "(none)", legend=pp.LOG_LEGEND)
        client = _get_client()
        # More candidates than one call can show properly (each with its lines and a screenshot)
        # are judged over several calls, the most similar first; within a call, in time order.
        batches = [sorted(cands[k:k + PER_CALL], key=lambda c: c["start"]) for k in range(0, len(cands), PER_CALL)]
        verdicts, cost, first, failed, stopped = [], 0.0, None, False, False
        for n, batch in enumerate(batches, 1):
            content_blocks, _n_img = _prompt(conn, cfg, seg, rv, batch, n, len(batches), total, len(cands))

            def _call(eff, content_blocks=content_blocks):
                return llm.parse(client, site="propagation",
                    model=cfg.claude_model, max_tokens=MAX_TOKENS,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": content_blocks}],
                    output_format=PropJudgement,
                    output_config={"effort": eff},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")

            try:
                resp = _call(cfg.claude_effort)
            except Exception:
                log.exception("propagation call failed for event %d (call %d of %d)", ev["id"], n, len(batches))
                failed = True
                break
            cost += efforts.call_cost(resp.usage)
            if resp.parsed_output is None:
                stopped = resp.stop_reason == "contact_eli"   # (a contact stops the task for good: no retry)
                failed = not stopped
                break
            ids = {c["id"] for c in batch}
            verdicts += [v for v in resp.parsed_output.verdicts if v.segment_id in ids]
            if first is None:
                first = (resp, _call)
        if failed or stopped:
            # Nothing is written from a ruling whose calls did not all come back: half a
            # propagation would leave the earlier records of the other half deleted below.
            if failed:
                _retry_later(conn, ev["id"])
            conn.execute("UPDATE feedback_events SET cost_usd=? WHERE id=?", (cost, ev["id"]))
            continue
        n_applied = n_flagged = 0
        now = time.time()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(seg_propagated)")}
        with db.tx(conn):
            # this ruling's earlier verdicts are replaced by the ones just given: a target it
            # no longer applies to must not get the old correction back at its next re-judgment
            conn.execute("DELETE FROM seg_propagated WHERE from_segment_id=?", (seg["id"],))
            for v in verdicts:
                verdict = v.verdict.strip().lower()
                if verdict not in ("applies", "uncertain"):
                    continue
                rec = {"segment_id": v.segment_id, "from_segment_id": seg["id"], "verdict": verdict,
                       "content": v.content, "switch_label": v.switch_label, "focused": v.focused, "note": v.note,
                       # only what the ruling itself set travels (the judge is told so)
                       "self_distraction": v.self_distraction if rv["self_distraction"] is not None else None,
                       "next_task": v.next_task if rv["next_task"] is not None else None}
                if not _apply(conn, rec):
                    continue
                # Kept in a table of its own: the evaluator replaces the whole seg_evals row
                # when it judges a segment again, and reapply() puts the correction back.
                # seg_propagated has no self_distraction / next_task columns yet (db.py); until
                # it does, those two are applied now but not put back after a re-judgment.
                row = {"segment_id": v.segment_id, "from_segment_id": seg["id"], "verdict": verdict, "content": v.content,
                       "switch_label": v.switch_label, "focused": None if v.focused is None else int(v.focused),
                       "note": v.note, "created": now,
                       "self_distraction": None if rec["self_distraction"] is None else int(rec["self_distraction"]),
                       "next_task": None if rec["next_task"] is None else int(rec["next_task"])}
                keys = [k for k in row if k in cols]
                conn.execute(f"INSERT OR REPLACE INTO seg_propagated({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})",
                             [row[k] for k in keys])
                if verdict == "applies":
                    n_applied += 1
                else:
                    n_flagged += 1
            conn.execute("UPDATE feedback_events SET n_applied=?, n_flagged=?, cost_usd=? WHERE id=?",
                         (n_applied, n_flagged, cost, ev["id"]))
        log.info("propagated feedback on segment %d: %d applied, %d flagged (%d candidates of %d in %d call(s), $%.3f)",
                 seg["id"], n_applied, n_flagged, len(cands), total, len(batches), cost)
        # effort-ladder shadows for this judgment role too (the first call of the ruling: one sample per event, as before)
        def _cmp(a, b):
            pa = {x.segment_id: x.verdict for x in a.verdicts}
            pb = {x.segment_id: x.verdict for x in b.verdicts}
            keys = set(pa) & set(pb)
            if not keys:
                return 0.0, "no overlap"
            same = sum(1 for k in keys if pa[k] == pb[k])
            return same / len(keys), f"{same}/{len(keys)} verdicts match"
        if first is not None:
            efforts.run_shadows(conn, cfg, client, "feedback_propagation", f"edit:{seg['id']}",
                                cfg.claude_effort, first[0], first[1], _cmp)
        done += 1
    return done

"""Writers for the person's judgments, shared by the dashboard endpoints and the chats.

Every stored judgment carries its provenance. The evaluator's rows point at the eval_runs
transcript that produced them. The person's edits (seg_reviews, span_reviews) and model-raised
review questions carry chat_message_id when they came out of a conversation with Claude, so
any judgment can be audited back to the exchange that caused it - and the dashboard flags it.
"""
from __future__ import annotations

import json
import time

from . import db
from .seg_models import CONTENTS, KINDS, SWITCHES, stored_switch

# The person's edit of the evaluator's judgment maps onto the legacy 4-label review scheme so the
# older per-switch models keep learning from the same corrections.
NEW2OLD = {("continuation", None): "continuation", ("return", None): "task_change",
           ("interruption", "self_distraction"): "distraction",
           ("interruption", "focus_start"): "task_change",
           ("interruption", "detour"): "interruption"}


class BadEdit(ValueError):
    """An edit that names labels or rows that do not exist (the caller's input is wrong)."""


def valid_antecedent(c, segment_id: int, ante: int | None) -> int | None:
    """An antecedent must name a real segment that starts before this one; else None."""
    if ante is None or ante == segment_id:
        return None
    row = c.execute("""SELECT (SELECT start FROM segments WHERE id=?) < g.start AS earlier
                       FROM segments g WHERE g.id=?""", (ante, segment_id)).fetchone()
    return ante if row and row["earlier"] else None


def write_seg_review(c, segment_id: int, *, content: str | None = None, switch_label: str | None = None,
                     interruption_kind: str | None = None, antecedent_id: int | None = None,
                     note: str | None = None, hybrid: bool | None = None,
                     chat_message_id: int | None = None, focus_exception: bool | None = None,
                     merge_existing: bool = False, focused: bool | None = None,
                     self_distraction: bool | None = None, next_task: bool | None = None) -> bool:
    """Record the person's judgment of a segment (authoritative; replaces any earlier edit).
    `focused` is their ruling on the quality of attention in the segment (True focused, False
    unfocused, None no ruling); `self_distraction` and `next_task` are their rulings on a shift.
    Returns True when a shift (a separate thread) was asserted with nothing said about what
    followed - the caller then has the evaluator re-judge the stretch with that ruling in hand."""
    if content is not None and content not in CONTENTS:
        raise BadEdit(f"bad content label {content!r}")
    switch_label = stored_switch(switch_label)  # models are shown 'shift'
    if switch_label is not None and switch_label not in SWITCHES:
        raise BadEdit(f"bad switch label {switch_label!r}")
    if interruption_kind is not None and interruption_kind not in KINDS:
        raise BadEdit(f"bad interruption kind {interruption_kind!r}")
    if not c.execute("SELECT 1 FROM segments WHERE id=?", (segment_id,)).fetchone():
        raise BadEdit(f"no segment #{segment_id}")
    if merge_existing:
        # Incremental edit (the chats): fields left None keep the person's earlier ruling on this
        # block instead of silently reverting to the evaluator's call. A new switch label resets
        # the kind/antecedent that belonged to the old one.
        prev = c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
        if prev is not None:
            content = content if content is not None else prev["content"]
            if switch_label is None:
                switch_label, interruption_kind, antecedent_id = prev["switch_label"], prev["interruption_kind"], prev["antecedent_id"]
            hybrid = hybrid if hybrid is not None else (None if prev["hybrid"] is None else bool(prev["hybrid"]))
            _pb = lambda k: None if prev[k] is None else bool(prev[k])
            focused = focused if focused is not None else _pb("focused")
            self_distraction = self_distraction if self_distraction is not None else _pb("self_distraction")
            next_task = next_task if next_task is not None else _pb("next_task")
            focus_exception = (focus_exception if focus_exception is not None
                               else (None if prev["focus_exception"] is None else bool(prev["focus_exception"])))
            note = note or prev["note"]
    with db.tx(c):
        ante = valid_antecedent(c, segment_id, antecedent_id) if switch_label in ("continuation", "return") else None
        _i = lambda v: None if v is None else int(bool(v))
        if switch_label and switch_label != "interruption":
            next_task = None
        if switch_label and switch_label not in ("interruption", "return"):
            self_distraction = None   # the two shift judgments only mean something on a shift
        if empty_ruling(content, switch_label, interruption_kind if switch_label == "interruption" else None, ante, note,
                        hybrid, focus_exception, focused, self_distraction, next_task):
            # 2026-11-21: "save note" with an empty box, or un-clicking the only selection, stored a
            # row with every field NULL (block 36839). The block then counted as reviewed: its
            # needs-review dot went while its question was still open, and the editor said "your
            # ruling is shown on the buttons". Saying nothing is not a ruling: no row.
            c.execute("DELETE FROM seg_reviews WHERE segment_id=?", (segment_id,))
            sync_legacy(c, segment_id)
            return False
        if chat_message_id is not None:
            # What a chat replaces is kept, so undoing the chat's edit puts back the ruling that
            # was there before it (undo_seg_review). The first copy per reply wins: a reply that
            # edits the block twice still goes back to what the person had.
            was = c.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
            if was is not None and was["chat_message_id"] != chat_message_id:
                c.execute("INSERT OR IGNORE INTO review_undo(kind, key, chat_message_id, rows, created) VALUES ('seg',?,?,?,?)",
                          (str(segment_id), chat_message_id, json.dumps([dict(was)]), time.time()))
        c.execute("INSERT OR REPLACE INTO seg_reviews(segment_id, content, switch_label, interruption_kind, antecedent_id,"
                  " note, created, hybrid, chat_message_id, focus_exception, focused, self_distraction, next_task)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (segment_id, content, switch_label, interruption_kind, ante, note, time.time(),
                   None if hybrid is None else int(hybrid), chat_message_id,
                   None if focus_exception is None else int(focus_exception),
                   _i(focused), _i(self_distraction), _i(next_task)))
        if content or switch_label or hybrid is not None:
            from . import propagation
            propagation.note_event(c, segment_id)
        sync_legacy(c, segment_id)
    return switch_label == "interruption" and not interruption_kind and focused is None


def empty_ruling(content, switch_label, interruption_kind, antecedent_id, note, hybrid, focus_exception,
                 focused, self_distraction, next_task) -> bool:
    """True when a ruling says nothing at all: no field set and no note of the person's own.
    The app's own "(answered ...)" note is not something the person said."""
    said = (note or "").strip()
    if said.startswith("(answered"):
        said = ""
    return not said and all(v is None for v in (content, switch_label, interruption_kind, antecedent_id, hybrid,
                                                focus_exception, focused, self_distraction, next_task))


def sync_legacy(c, segment_id: int) -> None:
    """Make the legacy per-switch review (`reviews`, `switches.status`) follow the block's ruling.

    While the ruling maps onto one of the four old labels (NEW2OLD) the legacy review is that
    label. The first time that happens the legacy state as it was is kept (review_undo, kind
    'legacy'), and it is put back when the ruling stops mapping or is removed.

    2026-11-21: undoing a block edit left the legacy review behind. Block 36839 ruled
    "continuation", changed to "shift", then undone: `reviews` for switch 33272 stayed
    "continuation" and the switch stayed "reviewed" through both steps, feeding the Day tile,
    "Reviews so far" and the learned model. Deriving the legacy row from the ruling, instead of
    writing it beside each edit, also makes it independent of the order in which several
    answers on one block are given and cleared. A legacy review the person made directly
    afterwards (keys 1-4) is not the ruling's to take back: it is left alone."""
    sw = c.execute("SELECT id, status FROM switches WHERE to_segment=?", (segment_id,)).fetchone()
    if sw is None:
        return
    rv = c.execute("SELECT switch_label, interruption_kind, note FROM seg_reviews WHERE segment_id=?",
                   (segment_id,)).fetchone()
    old = (NEW2OLD.get((rv["switch_label"], rv["interruption_kind"] if rv["switch_label"] == "interruption" else None))
           if rv is not None else None)
    snap = c.execute("SELECT rowid, rows FROM review_undo WHERE kind='legacy' AND key=? AND chat_message_id=0",
                     (str(segment_id),)).fetchone()
    cur = c.execute("SELECT * FROM reviews WHERE switch_id=?", (sw["id"],)).fetchone()
    if old:
        now = time.time()
        was = (json.loads(snap["rows"])[0] if snap is not None
               else {"switch_id": sw["id"], "review": dict(cur) if cur is not None else None, "switch_status": sw["status"]})
        was["wrote"] = now
        c.execute("INSERT OR REPLACE INTO review_undo(kind, key, chat_message_id, rows, created) VALUES ('legacy',?,0,?,?)",
                  (str(segment_id), json.dumps([was]), now))
        note = rv["note"]
        if note and note.startswith("(answered review question: ") and note.endswith(")"):
            note = note[len("(answered review question: "):-1]   # the legacy note is the option's own words
        c.execute("INSERT OR REPLACE INTO reviews(switch_id, label, note, created) VALUES (?,?,?,?)",
                  (sw["id"], old, note, now))
        c.execute("UPDATE switches SET status='reviewed' WHERE id=?", (sw["id"],))
    elif snap is not None:
        was = json.loads(snap["rows"])[0]
        if cur is not None and abs(cur["created"] - (was.get("wrote") or 0)) < 1e-3:   # still the row the ruling wrote
            if was.get("review") is None:
                c.execute("DELETE FROM reviews WHERE switch_id=?", (sw["id"],))
            else:
                _put_back(c, "reviews", [was["review"]])
            c.execute("UPDATE switches SET status=? WHERE id=?", (was.get("switch_status"), sw["id"]))
        c.execute("DELETE FROM review_undo WHERE rowid=?", (snap["rowid"],))


def write_span_review(c, start: float, end: float, verdict: str, subtype: str | None = None,
                      note: str | None = None, chat_message_id: int | None = None) -> None:
    """The person's verdict on a hypothesized span (replaces any earlier verdict on it)."""
    from .stats import SUBTYPES
    if verdict not in ("focus", "not_focus"):
        raise BadEdit(f"bad verdict {verdict!r}")
    if subtype is not None and subtype not in SUBTYPES:
        raise BadEdit(f"bad subtype {subtype!r}")
    with db.tx(c):
        if chat_message_id is not None:   # kept for undo, like a block ruling (write_seg_review)
            was = [dict(r) for r in c.execute("SELECT * FROM span_reviews WHERE start >= ? AND end <= ?", (start - 60, end + 60))
                   if r["chat_message_id"] != chat_message_id]
            if was:
                c.execute("INSERT OR IGNORE INTO review_undo(kind, key, chat_message_id, rows, created) VALUES ('span',?,?,?,?)",
                          (f"{start:.0f}:{end:.0f}", chat_message_id, json.dumps(was), time.time()))
        c.execute("DELETE FROM span_reviews WHERE start >= ? AND end <= ?", (start - 60, end + 60))
        c.execute("INSERT INTO span_reviews(start, end, verdict, subtype, note, created, chat_message_id)"
                  " VALUES (?,?,?,?,?,?,?)", (start, end, verdict, subtype, note, time.time(), chat_message_id))


def _put_back(c, table: str, rows: list[dict]) -> None:
    for row in rows:
        cols = list(row)
        c.execute(f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                  [row[k] for k in cols])


def delete_span_review(c, start: float, end: float) -> int:
    """Withdraw the verdict on a span. A verdict a chat wrote over an earlier one goes back to
    that earlier one (it used to be lost: the span went back to the app's own call)."""
    with db.tx(c):
        cur = c.execute("SELECT * FROM span_reviews WHERE start >= ? AND end <= ? ORDER BY created DESC",
                        (start - 60, end + 60)).fetchall()
        n = c.execute("DELETE FROM span_reviews WHERE start >= ? AND end <= ?", (start - 60, end + 60)).rowcount
        for r in cur:
            if r["chat_message_id"] is None:
                continue
            u = c.execute("SELECT rowid, rows FROM review_undo WHERE kind='span' AND key=? AND chat_message_id=?",
                          (f"{r['start']:.0f}:{r['end']:.0f}", r["chat_message_id"])).fetchone()
            if u is not None:
                _put_back(c, "span_reviews", json.loads(u["rows"]))
                c.execute("DELETE FROM review_undo WHERE rowid=?", (u["rowid"],))
                break
        return n


def undo_seg_review(c, segment_id: int) -> bool:
    """Undo the ruling on a block. One a chat wrote over an earlier ruling goes back to that
    earlier ruling (True); anything else is deleted, back to the evaluator's own judgment."""
    with db.tx(c):
        cur = c.execute("SELECT chat_message_id FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
        u = None
        if cur is not None and cur["chat_message_id"] is not None:
            u = c.execute("SELECT rowid, rows FROM review_undo WHERE kind='seg' AND key=? AND chat_message_id=?",
                          (str(segment_id), cur["chat_message_id"])).fetchone()
        c.execute("DELETE FROM seg_reviews WHERE segment_id=?", (segment_id,))
        if u is not None:
            _put_back(c, "seg_reviews", json.loads(u["rows"]))
            c.execute("DELETE FROM review_undo WHERE rowid=?", (u["rowid"],))
        sync_legacy(c, segment_id)   # the legacy switch review goes back with the ruling (2026-11-21)
        return u is not None


def open_question(c, segment_id: int, source: str, context: str, question: str, options: list[dict],
                  chat_message_id: int | None = None, cost: float = 0.0) -> int:
    """Put a review question in the person's queue. Options are validated the way the generator's
    are (each maps to a concrete judgment of a segment) and the neutral pass-through reading is
    appended, so the card behaves exactly like a generated question."""
    from .questions import passthrough_applies, passthrough_option
    if not c.execute("SELECT 1 FROM segments WHERE id=?", (segment_id,)).fetchone():
        raise BadEdit(f"no segment #{segment_id}")
    dup = c.execute("SELECT id, question FROM review_questions WHERE segment_id=? AND status='open'", (segment_id,)).fetchone()
    if dup is not None:
        raise BadEdit(f"block #{segment_id} already has open review question #{dup['id']} ({dup['question'][:80]}); not opening another")
    opts = []
    for o in options:
        label = (o.get("label") or "").strip()
        sid = o.get("segment_id") or segment_id
        sw, ct = stored_switch(o.get("switch_label")), o.get("content")  # models are shown 'shift'
        if not label or (sw is not None and sw not in SWITCHES) or (ct is not None and ct not in CONTENTS):
            continue
        if ct == "transition":
            continue  # the neutral pass-through option below covers this reading
        if not c.execute("SELECT 1 FROM segments WHERE id=?", (sid,)).fetchone():
            continue
        opts.append({"label": label, "segment_id": int(sid), "switch_label": sw,
                     "antecedent_id": o.get("antecedent_id") if sw in ("continuation", "return") else None,
                     "interruption_kind": None, "content": ct, "hybrid": o.get("hybrid"),
                     "focused": o.get("focused"),
                     # on purpose / a drift, and "the next task": only mean something on a shift or return
                     "self_distraction": o.get("self_distraction") if sw in (None, "interruption", "return") else None,
                     "next_task": o.get("next_task") if sw in (None, "interruption") else None})
    if not opts:
        raise BadEdit("no valid option (each needs a label and known content/switch labels)")
    if passthrough_applies(c, segment_id):
        opts.append(passthrough_option(segment_id, opts))
    with db.tx(c):
        cur = c.execute("INSERT INTO review_questions(segment_id, source, context, question, options, cost_usd,"
                        " created, chat_message_id) VALUES (?,?,?,?,?,?,?,?)",
                        (segment_id, source, context.strip(), question.strip(), json.dumps(opts), cost,
                         time.time(), chat_message_id))
        return int(cur.lastrowid)

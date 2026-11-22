"""Exceptions to the written rules (Eli, 2026-11-21).

The person sometimes rules against what the rules say: "this should be a focus span even
though it did not follow the rule, and here is why", or the other way round. The ruling itself
takes effect as any ruling does (span_reviews, seg_reviews). This module keeps the RECORD of
it: what was ruled, what the app had under the rules, the rule it departs from (when named) and
the person's reason. The rule drafter (rule_drafts.py) reads the record and proposes a general
rule once several exceptions made on different occasions share a reason.

What counts as an exception: only what the person says is one. Usually they tell a chat model
(set_span_verdict / set_segment_judgment with exception_to_rule); they can also mark it by hand
(a checkbox beside the span verdict buttons, a button in the block editor). A ruling that merely
differs from the app's call is a correction, not an exception, and is not recorded here.
An exception stops counting (withdrawn_at) when its ruling is undone or reversed, or when it is
unmarked. Rows are never deleted.
"""
import json
import time

from . import db


def find_span(conn, start: float, end: float) -> dict | None:
    """The hypothesized span sitting on these times, as stats has it now."""
    from . import stats
    from .config import logical_date
    seen = set()
    for ts in (start, end - 1):
        d = logical_date(ts)
        if d in seen:
            continue
        seen.add(d)
        t0, t1 = stats.day_bounds(d)
        se = stats.spans_and_events(stats.labelled_segments(conn, t0, t1), conn=conn)
        best = None
        for sp in se["focus_spans"] + se["disqualified"]:
            ov = min(sp["end"], end) - max(sp["start"], start)
            if ov > 0 and ov >= 0.8 * (end - start) and ov >= 0.8 * (sp["end"] - sp["start"]) and (best is None or ov > best[0]):
                best = (ov, sp)
        if best is not None:
            return best[1]
    return None


def call_text(own: dict | None) -> str | None:
    """The app's own call on a span (stats._apply_span_reviews: own_call), in words."""
    if not own:
        return None
    if own.get("focus"):
        return f"a focus span ({own.get('subtype') or 'kind not set'})"
    return "not a focus span" + (f": {own['reason']}" if own.get("reason") else "")


def _open_span_row(conn, start, end):
    return conn.execute("SELECT * FROM rule_exceptions WHERE kind='span' AND withdrawn_at IS NULL AND start >= ? AND end <= ?"
                        " ORDER BY created DESC", (start - 60, end + 60)).fetchone()


def record_span(conn, start: float, end: float, verdict: str, subtype: str | None, own: dict | None,
                is_exception: bool, reason: str | None = None, rule: str | None = None,
                chat_message_id: int | None = None) -> int | None:
    """Called after a span verdict is written. `own` is the app's call on the span from BEFORE
    the verdict. `is_exception`: the person marked it (the card's checkbox) or told a chat model
    so. Nothing is recorded on the app's own inference: a verdict against the app's call is
    usually a correction of a misjudgment, not an exception to a rule (Eli, 2026-11-21).
    A later unmarked verdict the same way leaves a standing exception alone; one the other way
    withdraws it. Returns the standing exception's id, or None."""
    reason = " ".join((reason or "").split()) or None
    rule = " ".join((rule or "").split()) or None
    ruled = "not a focus span" if verdict == "not_focus" else f"a focus span ({subtype or 'kind as the app had it'})"
    same_way = lambda row: row["ruled"].startswith("not") == (verdict == "not_focus")
    with db.tx(conn):
        row = _open_span_row(conn, start, end)
        if not is_exception:
            if row is None:
                return None
            if not same_way(row):
                conn.execute("UPDATE rule_exceptions SET withdrawn_at=? WHERE id=?", (time.time(), row["id"]))
                return None
            conn.execute("UPDATE rule_exceptions SET ruled=? WHERE id=?", (ruled, row["id"]))
            return row["id"]
        if row is not None:   # the same exception, ruled again: keep what it already says where nothing new is given
            conn.execute("UPDATE rule_exceptions SET ruled=?, reason=COALESCE(?, reason), rule=COALESCE(?, rule),"
                         " chat_message_id=COALESCE(?, chat_message_id) WHERE id=?",
                         (ruled, reason, rule, chat_message_id, row["id"]))
            return row["id"]
        cur = conn.execute("INSERT INTO rule_exceptions(kind, start, end, ruled, app_call, rule, reason, created, chat_message_id)"
                           " VALUES ('span',?,?,?,?,?,?,?,?)",
                           (start, end, ruled, call_text(own), rule, reason, time.time(), chat_message_id))
        return cur.lastrowid


def span_verdict_removed(conn, start: float, end: float) -> None:
    """Called after a span verdict is withdrawn: the exception goes with it."""
    with db.tx(conn):
        conn.execute("UPDATE rule_exceptions SET withdrawn_at=? WHERE kind='span' AND withdrawn_at IS NULL AND start >= ? AND end <= ?",
                     (time.time(), start - 60, end + 60))


def withdraw(conn, eid: int) -> bool:
    """"Not an exception after all": the ruling itself stays."""
    with db.tx(conn):
        return conn.execute("UPDATE rule_exceptions SET withdrawn_at=? WHERE id=? AND withdrawn_at IS NULL",
                            (time.time(), eid)).rowcount > 0


def mark_segment(conn, segment_id: int, reason: str | None) -> int | None:
    """The person marks their ruling on one block as an exception, by hand."""
    from .summaries import verdict_text
    r = conn.execute("SELECT * FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
    if r is None:
        return None
    for x in active(conn):   # already marked: add the reason if one is given now
        if x["kind"] == "segments" and segment_id in x["segment_ids"]:
            if (reason or "").strip():
                set_reason(conn, x["id"], reason)
            return x["id"]
    note = r["note"] if r["note"] and not r["note"].startswith("(") else None
    return record_segments(conn, [segment_id], verdict_text(r) or "note only", reason or note, None)


def record_segments(conn, segment_ids: list[int], ruled: str, reason: str | None, rule: str | None,
                    chat_message_id: int | None = None) -> int | None:
    """A ruling on blocks that a chat model was told is an exception to a written rule."""
    rows = [conn.execute("SELECT id, start, end FROM segments WHERE id=?", (i,)).fetchone() for i in segment_ids]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    with db.tx(conn):
        cur = conn.execute("INSERT INTO rule_exceptions(kind, start, end, segment_ids, ruled, rule, reason, created, chat_message_id)"
                           " VALUES ('segments',?,?,?,?,?,?,?,?)",
                           (min(r["start"] for r in rows), max((r["end"] or r["start"]) for r in rows),
                            json.dumps([r["id"] for r in rows]), ruled, " ".join((rule or "").split()) or None,
                            " ".join((reason or "").split()) or None, time.time(), chat_message_id))
        return cur.lastrowid


def active(conn) -> list[dict]:
    """The exceptions that stand, oldest first. One on blocks whose rulings have all been
    cleared since is withdrawn here (a block ruling can be undone in several places)."""
    out = []
    for r in conn.execute("SELECT * FROM rule_exceptions WHERE withdrawn_at IS NULL ORDER BY created").fetchall():
        d = dict(r)
        d["segment_ids"] = json.loads(d["segment_ids"] or "[]")
        if d["kind"] == "segments":
            q = ",".join("?" * len(d["segment_ids"]))
            if not d["segment_ids"] or not conn.execute(f"SELECT 1 FROM seg_reviews WHERE segment_id IN ({q}) LIMIT 1",
                                                        d["segment_ids"]).fetchone():
                with db.tx(conn):
                    conn.execute("UPDATE rule_exceptions SET withdrawn_at=? WHERE id=?", (time.time(), d["id"]))
                continue
        out.append(d)
    return out


def _covered(conn) -> dict[int, dict]:
    """exception id -> the newest rule draft resting on it."""
    out = {}
    for d in conn.execute("SELECT id, status, rule, final_text, exceptions FROM rule_drafts WHERE exceptions IS NOT NULL ORDER BY created"):
        for i in json.loads(d["exceptions"] or "[]"):
            out[i] = {"draft_id": d["id"], "status": d["status"], "rule": d["final_text"] or d["rule"]}
    return out


def due(conn, last_run: float) -> bool:
    """Is it time to ask the drafter about exceptions: a new one since its last run, and at
    least three standing that no draft has rested on yet."""
    rows = active(conn)
    cov = _covered(conn)
    return any(r["created"] > last_run for r in rows) and sum(1 for r in rows if r["id"] not in cov) >= 3


def lines(conn) -> list[str]:
    """The record as the drafting model reads it."""
    from . import prompt_parts as pp, summaries
    cov = _covered(conn)
    out = []
    for r in active(conn):
        mins = (r["end"] - r["start"]) / 60
        when = f"{pp.t(r['start'], '%a %Y-%m-%d %H:%M')}-{pp.t(r['end'], '%H:%M')} ({mins:.0f} min)"
        if r["kind"] == "span":
            row = summaries.lookup(conn, r["start"], r["end"])
            what = f"the span {when}" + (f", “{pp.one(row['title'])}”: {pp.one(row['summary'])}" if row is not None and row["summary"] else "")
        else:
            what = f"blocks {', '.join('id=' + str(i) for i in r['segment_ids'])}, {when} (each is in the rulings above)"
        s = (f"X{r['id']}, made {pp.t(r['created'], '%Y-%m-%d')}" + (" through a chat" if r["chat_message_id"] else "") + f": {what}\n"
             f"    they ruled: {r['ruled']}\n"
             + (f"    under the rules the app had it as: {r['app_call']}\n" if r["app_call"] else "")
             + (f"    the rule it departs from, as named in the chat: {r['rule']}\n" if r["rule"] else "")
             + f"    their reason: " + (f"“{r['reason']}”" if r["reason"] else "(none given)"))
        if r["id"] in cov:
            s += f"\n    (already behind a drafted rule, status {cov[r['id']]['status']})"
        out.append(s)
    return out


def listing(conn) -> list[dict]:
    """For the Rules page: every standing exception, newest first, with the draft it led to."""
    cov = _covered(conn)
    out = []
    for r in reversed(active(conn)):
        r["draft"] = cov.get(r["id"])
        out.append(r)
    return out


def set_reason(conn, eid: int, reason: str) -> bool:
    with db.tx(conn):
        return conn.execute("UPDATE rule_exceptions SET reason=? WHERE id=? AND withdrawn_at IS NULL",
                            (" ".join(reason.split()) or None, eid)).rowcount > 0

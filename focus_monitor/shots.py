"""Which captures of a stretch to show a model.

Every capture moment stores one file per attached display (segment_shots.display: 0 = laptop,
1+ = external). Six call sites used to take every k-th ROW of that table; with two displays the
rows alternate d0, d1, d0, d1, so an even k showed the model the laptop screen only. On
2026-11-16 the evaluator judged three minutes of commenting on a bill draft (external monitor)
as "the Sleep Bandit page sitting still" (laptop) and the question generator asked about a
conflict that did not exist. pick() samples MOMENTS, and for each takes the display that held
the active window.
"""
from __future__ import annotations

import re

_DISPLAY = re.compile(r"_d(\d+)")


def _display_of(path: str | None) -> int | None:
    m = _DISPLAY.search(path or "")
    return int(m.group(1)) if m else None


def pick(conn, segment_ids: list[int], max_n: int) -> list[dict]:
    """Up to max_n rows {segment_id, ts, path, display}, oldest first: one per capture moment
    (the active window's display), spread evenly over the stretch. Empty when the segments
    have no per-shot history; callers keep their own first_screenshot fallback for that."""
    ids = list(segment_ids)
    if not ids or max_n <= 0:
        return []
    qs = ",".join("?" * len(ids))
    rows = [dict(r) for r in conn.execute(
        f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) ORDER BY ts, display", ids)]
    # The collector records the active display's file as the segment's first_screenshot when
    # the segment opens and as last_screenshot on every refresh after that.
    active = {r["id"]: (_display_of(r["first_screenshot"]), _display_of(r["last_screenshot"]))
              for r in conn.execute(f"SELECT id, first_screenshot, last_screenshot FROM segments WHERE id IN ({qs})", ids)}
    moments: dict[tuple, list[dict]] = {}
    for r in rows:  # the files of one moment share the timestamp prefix of their names
        moments.setdefault((r["segment_id"], r["path"].split("_")[0]), []).append(r)
    chosen, seen_seg = [], set()
    for (seg_id, _), group in moments.items():
        first_d, last_d = active.get(seg_id, (None, None))
        want = first_d if seg_id not in seen_seg else (last_d if last_d is not None else first_d)
        seen_seg.add(seg_id)
        chosen.append(next((r for r in group if r["display"] == want), group[0]))
    n = len(chosen)
    if n <= max_n:
        return chosen
    if max_n == 1:
        return chosen[:1]
    return [chosen[i] for i in sorted({round(k * (n - 1) / (max_n - 1)) for k in range(max_n)})]


# ---- every capture moment, both displays (2026-11-20) -------------------------------------------
# pick() above takes a fixed number of images for a whole stretch, so with many segments most got
# none: 12,144 of 42,445 evaluator targets (29%) were judged without a screenshot although one
# existed, and the second display (25% of capture moments have one) was never sent. The evaluator
# now asks for every segment's moments and spends a per-call image budget itself; pick() stays for
# the callers that want a handful of images.

def on_disk(path: str | None) -> bool:
    from .config import screenshot_file
    return screenshot_file(path) is not None


def moments(conn, segment_ids: list[int]) -> dict[int, list[dict]]:
    """{segment_id: [moment, ...]}, oldest first, for every capture moment whose active-display
    file is still on disk. A moment is {segment_id, ts, active: row, others: [rows]}; a row is
    {segment_id, ts, path, display}; `others` are the other displays' files of the same moment
    that are on disk. Segments recorded before per-shot history (segment_shots, from about
    2026-10-13) fall back to the first / last capture named on the segment itself."""
    ids = list(segment_ids)
    out: dict[int, list[dict]] = {i: [] for i in ids}
    if not ids:
        return out
    rows, segs = [], {}
    for k in range(0, len(ids), 500):
        part = ids[k:k + 500]
        qs = ",".join("?" * len(part))
        rows += [dict(r) for r in conn.execute(
            f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) ORDER BY ts, display", part)]
        segs.update({r["id"]: r for r in conn.execute(
            f"SELECT id, start, end, first_screenshot, last_screenshot FROM segments WHERE id IN ({qs})", part)})
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["segment_id"], r["path"].split("_")[0]), []).append(r)
    seen = set()
    for (seg_id, _), group in groups.items():
        g = segs.get(seg_id)
        first_d, last_d = (_display_of(g["first_screenshot"]), _display_of(g["last_screenshot"])) if g is not None else (None, None)
        want = first_d if seg_id not in seen else (last_d if last_d is not None else first_d)
        seen.add(seg_id)
        act = next((r for r in group if r["display"] == want), group[0])
        if not on_disk(act["path"]):
            # the active display's file is gone; another display's file of that moment still shows something
            act = next((r for r in group if on_disk(r["path"])), None)
            if act is None:
                continue
        out[seg_id].append({"segment_id": seg_id, "ts": act["ts"], "active": act, "active_known": act["display"] == want,
                            "others": [r for r in group if r is not act and on_disk(r["path"])]})
    has_rows = {r["segment_id"] for r in rows}
    for seg_id in ids:
        g = segs.get(seg_id)
        if out[seg_id] or g is None or seg_id in has_rows:
            continue
        for path, ts in ((g["first_screenshot"], g["start"]), (g["last_screenshot"], g["end"] or g["start"])):
            if path and on_disk(path) and not any(m["active"]["path"] == path for m in out[seg_id]):
                row = {"segment_id": seg_id, "ts": ts, "path": path, "display": _display_of(path) or 0}
                out[seg_id].append({"segment_id": seg_id, "ts": ts, "active": row, "active_known": True, "others": []})
    return out


def why_none(conn, segment_id: int) -> str:
    """Why a segment has no image to show: 'no capture exists' (none was ever recorded for it)
    or 'file no longer on disk' (recorded, since deleted). The evaluator is told which, so a
    missing image is never read as a fact about the activity."""
    from .config import PRIVATE_MARK
    paths = [r[0] for r in conn.execute("SELECT path FROM segment_shots WHERE segment_id=?", (segment_id,))]
    g = conn.execute("SELECT first_screenshot, last_screenshot FROM segments WHERE id=?", (segment_id,)).fetchone()
    paths += [p for p in ((g["first_screenshot"], g["last_screenshot"]) if g is not None else ()) if p]
    if not paths:
        return "no capture exists"
    if any(PRIVATE_MARK in p for p in paths):
        return "file no longer on disk (captures showing a private app such as Signal are deleted after two weeks)"
    return "file no longer on disk"


def spread(items: list, n: int) -> list:
    """`n` of `items`, evenly spaced, always with the first and the last."""
    k = len(items)
    if n >= k:
        return list(items)
    if n <= 1:
        return items[:1]
    return [items[i] for i in sorted({round(j * (k - 1) / (n - 1)) for j in range(n)})]


def nearest(conn, ts: float, before: bool, within_s: float = 600.0) -> dict | None:
    """The capture moment nearest to `ts` on one side of it (at or before / at or after), from any
    segment, no further than within_s away; None when there is none on disk. For a phone pickup:
    what was on the Mac's screen as the phone was picked up, and as it was put down."""
    q = ("SELECT segment_id, ts, path, display FROM segment_shots WHERE ts <= ? AND ts >= ? ORDER BY ts DESC LIMIT 12" if before
         else "SELECT segment_id, ts, path, display FROM segment_shots WHERE ts >= ? AND ts <= ? ORDER BY ts LIMIT 12")
    rows = [dict(r) for r in conn.execute(q, (ts, ts - within_s if before else ts + within_s))]
    tried = set()
    for r in rows:
        key = (r["segment_id"], r["path"].split("_")[0])
        if key in tried:
            continue
        tried.add(key)
        ms = [m for m in moments(conn, [r["segment_id"]]).get(r["segment_id"], [])
              if m["active"]["path"].split("_")[0] == key[1] or any(o["path"].split("_")[0] == key[1] for o in m["others"])]
        if ms:
            return ms[0]
    return None

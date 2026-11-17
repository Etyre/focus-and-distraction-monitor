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

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from .. import LABELS
from .. import db

Probs = dict[str, float]  # label -> probability, sums to 1


def normalize(p: dict[str, float]) -> Probs:
    out = {k: max(float(p.get(k, 0.0)), 1e-6) for k in LABELS}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def seg_desc(s: dict | None, now: float | None = None) -> str:
    if s is None:
        return "(none)"
    end = s["end"] or now or s["start"]
    dur = max(0, end - s["start"])
    if s.get("idle"):
        return f"[idle/away for {dur/60:.1f} min]"
    parts = [s["app"]]
    if s["title"]:
        parts.append(f"“{s['title'][:120]}”")
    if s["url"]:
        parts.append(s["url"][:160])
    t = dt.datetime.fromtimestamp(s["start"]).strftime("%H:%M:%S")
    return f"{t} ({dur/60:.1f} min) " + " | ".join(parts)


@dataclass
class SwitchContext:
    switch: dict
    from_seg: Optional[dict]
    to_seg: dict
    before: list[dict] = field(default_factory=list)   # segments preceding from_seg (oldest first)
    after: list[dict] = field(default_factory=list)    # segments following to_seg
    review_history: list[dict] = field(default_factory=list)
    from_state: str = "focus"
    focus_run_min: float = 0.0
    toggl_before: Optional[dict] = None   # Toggl entry running just before the switch
    toggl_after: Optional[dict] = None    # entry running shortly after (may be the same)
    toggl_started: Optional[dict] = None  # entry the person STARTED within ~90 s of the switch
    transit: list[dict] = field(default_factory=list)  # windows passed through on the way (one transition)

    @property
    def ts(self) -> float:
        return self.switch["ts"]

    def narrative(self) -> str:
        lines = []
        if self.from_state == "focus":
            lines.append(f"Context: before this switch the person had been in an unbroken focused span for "
                         f"about {self.focus_run_min:.0f} minutes. Whether this switch continues, interrupts, or "
                         f"distracts from that span is exactly what matters - judge carefully.")
        else:
            lines.append(f"Context: before this switch the person was already '{self.from_state}' (not in a focus span).")
        from .. import toggl as _t
        if self.toggl_before or self.toggl_after or self.toggl_started:
            lines.append("Declared intention (the person's own Toggl time tracking):")
            lines.append("  before the switch: " + _t.describe(self.toggl_before))
            if self.toggl_started:
                lines.append("  they STARTED a new Toggl entry at this switch: " + _t.describe(self.toggl_started)
                             + "  <- a deliberate task change / focus start, unless the entry itself is leisure")
            elif (self.toggl_after or {}).get("id") != (self.toggl_before or {}).get("id"):
                lines.append("  after the switch: " + _t.describe(self.toggl_after))
            lines.append("  A running Toggl entry says what they meant to be doing; compare the TO window with it.")
        lines.append("Recent activity (oldest first):")
        for s in self.before:
            lines.append("  " + seg_desc(s))
        lines.append("  FROM >>> " + seg_desc(self.from_seg))
        if self.transit:
            lines.append(f"  (then passed through {len(self.transit)} windows in quick succession while moving - "
                         f"desktops/windows/tabs - treat this as ONE transition:)")
            for s in self.transit:
                lines.append("     via " + seg_desc(s))
        lines.append("  TO   >>> " + seg_desc(self.to_seg))
        if self.after:
            lines.append("What happened next:")
            for s in self.after:
                lines.append("  " + seg_desc(s))
        return "\n".join(lines)


def build_context(conn, switch_id: int, n_before: int = 6, n_after: int = 4,
                  from_seg_id: int | None = None) -> SwitchContext | None:
    full = db.switch_full(conn, switch_id)
    if not full:
        return None
    to_seg = full["to"]
    from_seg = full["from"]
    transit: list[dict] = []
    if from_seg_id is None:
        g = conn.execute("SELECT from_segment FROM switches WHERE group_id=? AND id!=? ORDER BY ts LIMIT 1",
                         (switch_id, switch_id)).fetchone()
        from_seg_id = g[0] if g else None
    if from_seg_id is not None and from_seg and from_seg_id != from_seg["id"]:
        origin = db.segment(conn, from_seg_id)
        if origin:
            transit = [dict(r) for r in conn.execute(
                "SELECT * FROM segments WHERE start >= ? AND start < ? ORDER BY start", (origin["end"], to_seg["start"]))]
            from_seg = dict(origin)
    before = [dict(r) for r in db.segments_before(conn, from_seg["start"] if from_seg else to_seg["start"], n_before)]
    after = [dict(r) for r in db.segments_after(conn, to_seg["start"], n_after)]
    from . import base as _b  # noqa
    from .. import stats
    from .. import toggl
    state, run = stats.focus_context(conn, from_seg)
    ts = full["switch"]["ts"]
    try:
        tb, ta, tst = toggl.entry_at(conn, ts - 5), toggl.entry_at(conn, ts + 120), toggl.entry_started_near(conn, ts)
    except Exception:  # table missing on very old DBs
        tb = ta = tst = None
    return SwitchContext(switch=full["switch"], from_seg=from_seg, to_seg=to_seg, before=before, after=after,
                         from_state=state, focus_run_min=run, toggl_before=tb, toggl_after=ta, toggl_started=tst,
                         transit=transit)


class Classifier:
    name: str = "base"

    def predict(self, ctx: SwitchContext) -> tuple[Probs, str, float] | None:
        """Return (probs, rationale, cost_usd) or None to abstain."""
        raise NotImplementedError

    def learn(self, conn) -> None:
        """Update from human reviews. Called before a classification batch."""

"""Interactive chat with Claude about a single switch, for use while reviewing uncertain ones."""
from __future__ import annotations

import logging

import anthropic

from . import LABEL_HELP, db
from .classifiers.base import build_context
from .classifiers.claude_vision import _b64
from .config import Config

log = logging.getLogger("chat")

SYSTEM = """You are helping this person review their own attention-monitoring log. They are looking at ONE \
window switch and want to reason about it with you. You can see the before/after screenshots, the \
surrounding activity, their Toggl "intent" entry, and how the models classified the switch.

Answer their questions about THIS switch concisely and concretely - reference what is actually visible on \
screen and in the activity log. If they ask, say which label you'd give and why, and note honestly when it's \
genuinely ambiguous. Don't lecture; be a sharp, brief thinking partner.

The four labels:
{labels}

About this person (their own words):
{about_me}
"""

_MAXLBL = ["continuation", "interruption", "distraction", "task_change"]


def _top_label(p: dict) -> str:
    probs = {k: p[f"p_{k}"] for k in _MAXLBL}
    return max(probs, key=probs.get)


class ReviewChat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        self.system = SYSTEM.format(
            labels="\n".join(f"- {k}: {v}" for k, v in LABEL_HELP.items()),
            about_me=cfg.about_me.strip() or "(no description given)")

    def _context_blocks(self, conn, switch_id: int) -> list[dict]:
        ctx = build_context(conn, switch_id)
        full = db.switch_full(conn, switch_id)
        blocks: list[dict] = []
        before = _b64(ctx.from_seg["last_screenshot"] if ctx.from_seg else None)
        after = _b64(ctx.to_seg["first_screenshot"])
        if before:
            blocks.append({"type": "text", "text": "Screenshot BEFORE the switch:"})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": before}})
        if after:
            blocks.append({"type": "text", "text": "Screenshot shortly AFTER the switch:"})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": after}})
        preds = "\n".join(f"  - {p['model']}: {_top_label(p)} — {p['rationale']}" for p in full["predictions"])
        ens = full["ensemble"]
        review = full["review"]
        summary = ctx.narrative()
        summary += f"\n\nHow the models classified this switch:\n{preds or '  (none yet)'}"
        if ens:
            summary += f"\nEnsemble: {ens['label']}" + (" (flagged uncertain)" if ens["uncertain"] else "")
        if review:
            summary += f"\nThe person already reviewed this as: {review['label']}"
        blocks.append({"type": "text", "text": summary})
        return blocks

    def reply(self, conn, switch_id: int, messages: list[dict]) -> str:
        return _run_chat(self.client, self.cfg, self.system, self._context_blocks(conn, switch_id), messages)


def _run_chat(client, cfg, system, ctx_blocks, messages) -> str:
    api_msgs: list[dict] = []
    for i, m in enumerate(messages):
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = (m.get("text") or "").strip()
        if i == 0 and role == "user":
            api_msgs.append({"role": "user",
                             "content": ctx_blocks + [{"type": "text", "text": "\n\nMy question: " + text}]})
        else:
            api_msgs.append({"role": role, "content": text})
    if not api_msgs:
        return ""
    resp = client.messages.create(
        model=cfg.claude_model, max_tokens=1024,
        system=system,
        messages=api_msgs,
        output_config={"effort": "low"},
    )
    if resp.stop_reason == "refusal":
        return "(Claude declined to answer that.)"
    return "".join(b.text for b in resp.content if b.type == "text").strip() or "(no reply)"


SPAN_SYSTEM = """You are helping this person review their own attention-monitoring log. They are looking at ONE \
hypothesized focus span and deciding (a) whether it really was a focus span at all, and (b) which kind. \
You see the WHOLE DAY's activity log with the span under review marked in it - so you can reason about what \
led into the span, what followed it, and whether its boundaries are right - plus the day's other spans, the \
day's Toggl entries, the system's current call and prior summary of this span, and a few sampled screenshots \
from inside it.

Answer their questions about THIS span concisely and concretely - point at specific blocks, times, and \
what's visible on screen. If they ask for a verdict, say whether you'd call it a focus span, which subtype, \
and why - and note honestly when it's genuinely ambiguous. Their calibration feedback is that the system \
has been far too optimistic about what counts as focused work; do not repeat that mistake. Don't lecture; \
be a sharp, brief thinking partner.

Span subtypes: creative_work, focused_task, reading, planning, other_focused, meeting.

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}
"""


class SpanChat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        from .classifiers.claude_vision import _rules_doc
        self.system = SPAN_SYSTEM.format(rules=_rules_doc(),
                                         about_me=cfg.about_me.strip() or "(no description given)")

    def _context_blocks(self, conn, start: float, end: float) -> list[dict]:
        import datetime as dt
        from . import stats, summaries, toggl
        from .config import logical_date
        t0, t1 = stats.day_bounds(logical_date(start))
        day_segs = stats.labelled_segments(conn, t0, t1)
        se = stats.spans_and_events(day_segs, conn=conn)
        mid = (start + end) / 2
        sp = next((s for s in se["focus_spans"] + se["disqualified"]
                   if s["start"] - 1 <= mid < s["end"] + 1), None)
        if sp is None:
            raise ValueError("no span at that time")
        blocks: list[dict] = []
        sp_ids = [s["id"] for r in sp["runs"] for s in r["segments"]]
        shot_rows = []
        if sp_ids:
            qs = ",".join("?" * len(sp_ids))
            shot_rows = [dict(r2) for r2 in conn.execute(
                f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) ORDER BY ts", sp_ids)]
        if not shot_rows:
            shot_rows = [{"segment_id": s["id"], "ts": s["clip_start"], "path": s["first_screenshot"], "display": 0}
                         for r in sp["runs"] for s in r["segments"] if s.get("first_screenshot")]
        step = max(1, len(shot_rows) // 4)
        for sh in shot_rows[::step][:4]:
            b = _b64(sh["path"])
            if b:
                blocks.append({"type": "text", "text": f"Screenshot at {dt.datetime.fromtimestamp(sh['ts']).strftime('%H:%M')}{' - EXTERNAL display' if sh.get('display') else ''}:"})
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        rats = {}
        if sp_ids:
            qs = ",".join("?" * len(sp_ids))
            rats = {r2["segment_id"]: r2["rationale"] for r2 in conn.execute(
                f"SELECT segment_id, rationale FROM seg_evals WHERE segment_id IN ({qs})", sp_ids)
                if r2["rationale"]}
        call = sp.get("disq_reason") and f"NOT focused work ({sp['disq_reason']})" or sp.get("subtype")
        row = summaries.lookup(conn, sp["start"], sp["end"])

        def _t(ts, fmt="%H:%M"):
            return dt.datetime.fromtimestamp(ts).strftime(fmt)

        overview = "\n".join(
            f"  {_t(s2['start'])}-{_t(s2['end'])}: "
            + (f"DISQUALIFIED ({s2.get('disq_reason')})" if s2.get("disq_reason") else s2.get("subtype", "?"))
            + f", {s2['focus_min']:.0f} min focused"
            + ("  <== THE SPAN UNDER REVIEW" if s2 is sp else "")
            for s2 in sorted(se["focus_spans"] + se["disqualified"], key=lambda s2: s2["start"]))
        tg = toggl.entries_between(conn, t0, t1)
        tgtxt = "\n".join(
            f"  {_t(e['start'])}-{_t(e['stop'] or t1)}: {e['description'] or '(unnamed)'}"
            for e in tg) or "  (no Toggl entries at all this day)"
        # Full-day log with the span marked; elide far-away stretches to bound the prompt.
        before = [s2 for s2 in day_segs if s2["clip_end"] <= sp["start"] and s2["duration"] > 0]
        inside = [s2 for s2 in day_segs if sp["start"] <= s2["clip_start"] < sp["end"] and s2["duration"] > 0]
        after = [s2 for s2 in day_segs if s2["clip_start"] >= sp["end"] and s2["duration"] > 0]
        lines = []
        if len(before) > 120:
            lines.append(f"  ... {len(before) - 120} earlier blocks elided ...")
            before = before[-120:]
        lines += [summaries.seg_line(s2) for s2 in before]
        lines.append(">>> START OF THE SPAN UNDER REVIEW <<<")
        lines += [summaries.seg_line(s2) + (f"  // {rats[s2['id']][:80]}" if s2["id"] in rats else "")
                  for s2 in inside]
        lines.append(">>> END OF THE SPAN UNDER REVIEW <<<")
        lines += [summaries.seg_line(s2) for s2 in after[:120]]
        if len(after) > 120:
            lines.append(f"  ... {len(after) - 120} later blocks elided ...")
        text = (f"The span under review: {_t(sp['start'], '%Y-%m-%d %H:%M')}-{_t(sp['end'])} "
                f"({sp['duration'] / 60:.0f} min, {sp['focus_min']:.0f} focused, "
                f"{len(sp.get('detours', []))} contained interruption(s), ended by {sp.get('ended_by')}).\n"
                f"System's current call: {call}.\n"
                f"Prior summary of this span: {row['summary'] if row else '(none yet)'}\n\n"
                f"All hypothesized spans this day:\n{overview}\n\n"
                f"Toggl entries this day:\n{tgtxt}\n\n"
                f"The whole day's activity log (every block; state and the switch label that led into it; "
                f"the span under review is marked):\n" + "\n".join(lines))
        blocks.append({"type": "text", "text": text})
        return blocks

    def reply(self, conn, start: float, end: float, messages: list[dict]) -> str:
        return _run_chat(self.client, self.cfg, self.system, self._context_blocks(conn, start, end), messages)

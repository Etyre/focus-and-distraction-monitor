"""Span summaries: once a hypothesized focus span has ended, Claude writes a short factual
summary of what the person was actually doing in it. The summary is shown in every capture
view (timeline hover) inside that span - together with the running Toggl entry - so each
span judgment can be audited at a glance."""
from __future__ import annotations

import datetime as dt
import logging
import time

from . import db
from .config import PRICE_INPUT, PRICE_OUTPUT, Config, load_config

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


def seg_line(s) -> str:
    t = dt.datetime.fromtimestamp(s["clip_start"]).strftime("%H:%M")
    what = (s.get("app") or "") + ((" " + s["domain"]) if s.get("domain") else "")
    title = (s.get("title") or "")[:70]
    bits = [t, "%4.1fm" % (s["duration"] / 60), s.get("state", ""), what, title]
    if s.get("label") and s["label"] != "continuation":
        bits.append(f"<switch: {s['label']}>")
    if s.get("activity"):
        bits.append(f"[{s['activity']}]")
    if s.get("toggl_desc"):
        bits.append(f"(toggl: {s['toggl_desc']})")
    return "  ".join(str(b) for b in bits)


def _span_log(sp) -> str:
    lines = [seg_line(s) for r in sp["runs"] for s in r["segments"]]
    if len(lines) > 150:  # keep the prompt bounded for very long spans
        lines = lines[:75] + [f"  ... {len(lines) - 150} segments elided ..."] + lines[-75:]
    return "\n".join(lines)


PROMPT = """Below is the activity log of one hypothesized focus span ({t0}-{t1}, {mins:.0f} min).
Write a 1-2 sentence factual summary of what the person was actually doing, so they can later
audit whether this really was focused work. Name the concrete sites, apps, or documents; be
honest about drift, channel-surfing, or mixed activity; note whether a Toggl entry was running.
No preamble - just the summary.

{log}"""


def generate(conn, cfg: Config, sp) -> bool:
    """Write and store a summary for one finished span. Returns True on success."""
    client = _get_client()
    if client is None:
        return False
    t0s = dt.datetime.fromtimestamp(sp["start"]).strftime("%H:%M")
    t1s = dt.datetime.fromtimestamp(sp["end"]).strftime("%H:%M")
    prompt = PROMPT.format(t0=t0s, t1=t1s, mins=sp["duration"] / 60, log=_span_log(sp))
    resp = client.messages.create(
        model=cfg.claude_model, max_tokens=300, output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
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
        conn.execute("INSERT INTO span_summaries(start, end, summary, cost_usd, created) VALUES (?,?,?,?,?)",
                     (sp["start"], sp["end"], text, cost, time.time()))
    log.info("summarized span %s-%s ($%.4f)", t0s, t1s, cost)
    return True


def ensure_summaries(conn, cfg: Config | None = None, max_new: int = 2) -> int:
    """Summarize up to max_new recently finished spans (qualifying + disqualified) that
    don't have a stored summary yet. Called from the classifier loop."""
    from . import stats
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd:
        return 0
    now = time.time()
    segs = stats.labelled_segments(conn, now - 7 * 86400, now)  # cover the span-review queue window
    se = stats.spans_and_events(segs, conn=conn)
    done = 0
    for sp in se["focus_spans"] + se["disqualified"]:
        if done >= max_new:
            break
        if sp.get("ended_by") == "ongoing":
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

"""Span summaries: once a hypothesized focus span has ended, Claude writes a short factual
summary of what the person was actually doing in it. The summary is shown in every capture
view (timeline hover) inside that span - together with the running Toggl entry - so each
span judgment can be audited at a glance."""
from __future__ import annotations

import datetime as dt
import logging
import time

from pydantic import BaseModel, Field

from . import db
from .config import PRICE_INPUT, PRICE_OUTPUT, Config, load_config


class SpanSummaryJudgement(BaseModel):
    title: str = Field(description="3-8 words naming the OBJECT OF ATTENTION, like a document title - e.g. 'Reading METR's incident report', 'Editing elityre.com date page'. Use an epistemic marker ('likely ...') only if genuinely unsure.")
    summary: str = Field(description="1-2 factual sentences: what the person was actually doing.")
    coherent: bool = Field(description="THE CHECK: a focus span means ONE project/task/intention held throughout, with contained interruptions under 5 minutes each. Is the stretch you just described plausibly one object of attention? False whenever your own description contradicts that - e.g. the first stretch is a different activity than the rest, phrases like 'switched to' or 'from X on', or a sequence of unrelated things. (The reading->creative merge only excuses a shift when the writing grows out of the SAME material being read; moving to a different project is a different object.)")
    issue: str | None = Field(default=None, description="When coherent is false: one sentence naming the contradiction and, if apparent, where the real boundary lies.")

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
    if s.get("content"):
        bits.append(f"<{s['content']}>")
    if s.get("label") and s["label"] != "continuation":
        bits.append(f"<switch: {s['label']}>")
    if s.get("activity"):
        bits.append(f"[{s['activity']}]")
    if s.get("toggl_desc"):
        bits.append(f"(toggl: {s['toggl_desc']})")
    ia, dur2 = s.get("inact_s") or 0.0, s.get("duration") or 1.0
    if ia >= 60 and ia >= 0.3 * dur2:
        bits.append(f"(input idle {ia / 60:.0f}m of {dur2 / 60:.0f}m)")
    return "  ".join(str(b) for b in bits)


def _span_log(sp) -> str:
    lines = [seg_line(s) for r in sp["runs"] for s in r["segments"]]
    if len(lines) > 150:  # keep the prompt bounded for very long spans
        lines = lines[:75] + [f"  ... {len(lines) - 150} segments elided ..."] + lines[-75:]
    return "\n".join(lines)


PROMPT = """Below is the activity log of one hypothesized focus span ({t0}-{t1}, {mins:.0f} min).
Write a 1-2 sentence factual summary of what the person was actually DOING - the undertaking,
not the window inventory: "editing the date page of your personal site (VS Code + Terminal +
checking the rendered pages)" rather than "cycling between Code, Terminal, and Chrome". Infer
the activity from titles, content labels, and patterns, with epistemic markers ("apparently",
"likely") when inferring. Name the concrete sites, apps, or documents as evidence; be honest
about drift, channel-surfing, or mixed activity; note whether a Toggl entry was running.

Then CHECK your own description against the definition: a focus span holds ONE object of
attention throughout (window-switching within one piece of work is fine; interruptions under
5 minutes are contained). If what you just described is not plausibly one object of attention,
say so (coherent=false) and name the contradiction - this triggers review of the span's
boundaries.

{log}"""


def generate(conn, cfg: Config, sp) -> bool:
    """Write and store a summary for one finished span. The summarizer gets the evaluator's
    per-segment rationales and a few sampled screenshots, not just window titles - Terminal
    windows and untitled docs carry no activity information in their titles."""
    client = _get_client()
    if client is None:
        return False
    t0s = dt.datetime.fromtimestamp(sp["start"]).strftime("%H:%M")
    t1s = dt.datetime.fromtimestamp(sp["end"]).strftime("%H:%M")
    segs = [s for r_ in sp["runs"] for s in r_["segments"]]
    ids = [s["id"] for s in segs]
    rats = {}
    if ids:
        qs = ",".join("?" * len(ids))
        rats = {r2["segment_id"]: r2["rationale"] for r2 in conn.execute(
            f"SELECT segment_id, rationale FROM seg_evals WHERE segment_id IN ({qs})", ids)
            if r2["rationale"]}
    lines = [seg_line(s) + (f"  // {rats[s['id']][:90]}" if s["id"] in rats else "") for s in segs]
    if len(lines) > 150:
        lines = lines[:75] + [f"  ... {len(lines) - 150} segments elided ..."] + lines[-75:]
    content: list = []
    if ids:
        from .classifiers.claude_vision import _b64
        qs = ",".join("?" * len(ids))
        shots = [dict(r2) for r2 in conn.execute(
            f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) ORDER BY ts", ids)]
        step = max(1, len(shots) // 4)
        for sh in shots[::step][:4]:
            b = _b64(sh["path"])
            if b:
                where = " - EXTERNAL display" if sh.get("display") else ""
                content.append({"type": "text", "text": f"Screenshot at {dt.datetime.fromtimestamp(sh['ts']).strftime('%H:%M')}{where}:"})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
    content.append({"type": "text", "text": PROMPT.format(t0=t0s, t1=t1s, mins=sp["duration"] / 60,
                                                          log="\n".join(lines))})
    from .classifiers.claude_vision import _rules_doc
    system = [{"type": "text", "text": ("You summarize hypothesized focus spans for a personal "
              "attention monitor, and check them against the person's own rules.\n\n"
              "Their rules and definitions (authoritative):\n" + _rules_doc()
              + "\n\nAbout this person (their own words):\n" + (cfg.about_me.strip() or "(none)")),
              "cache_control": {"type": "ephemeral"}}]
    resp = client.beta.messages.parse(
        model=cfg.claude_model, max_tokens=500, output_config={"effort": "medium"},
        system=system,
        messages=[{"role": "user", "content": content}],
        output_format=SpanSummaryJudgement)
    if resp.parsed_output is None:
        return False
    from . import efforts as _eff

    def _scall(eff):
        return client.beta.messages.parse(
            model=cfg.claude_model, max_tokens=500, output_config={"effort": eff},
            system=system, messages=[{"role": "user", "content": content}],
            output_format=SpanSummaryJudgement)

    def _scmp(a, b):
        agree = 1.0 if a.coherent == b.coherent else 0.0
        return agree, f"coherent: {a.coherent} vs {b.coherent}"

    _eff.run_shadows(conn, cfg, client, "span_summarizer", f"span:{t0s}-{t1s}",
                     "medium", resp, _scall, _scmp,
                     prompt_text=PROMPT.format(t0=t0s, t1=t1s, mins=sp["duration"] / 60, log="(span log)")[:300])
    j: SpanSummaryJudgement = resp.parsed_output
    text = j.summary.strip()
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
        conn.execute("INSERT INTO span_summaries(start, end, summary, cost_usd, created, coherent, issue, title)"
                     " VALUES (?,?,?,?,?,?,?,?)",
                     (sp["start"], sp["end"], text, cost, time.time(), int(j.coherent),
                      (j.issue or None) if not j.coherent else None, j.title.strip() or None))
    log.info("summarized span %s-%s ($%.4f)%s", t0s, t1s, cost,
             "" if j.coherent else " - COHERENCE ISSUE: " + (j.issue or ""))
    return True


def ensure_summaries(conn, cfg: Config | None = None, max_new: int = 2) -> int:
    """Summarize up to max_new recently finished spans (qualifying + disqualified) that
    don't have a stored summary yet. Called from the classifier loop."""
    from . import stats
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    now = time.time()
    segs = stats.labelled_segments(conn, now - 7 * 86400, now)  # cover the span-review queue window
    se = stats.spans_and_events(segs, conn=conn)
    done = 0
    for sp in se["focus_spans"] + se["disqualified"]:
        if done >= max_new:
            break
        if sp.get("ended_by") in ("ongoing", "awaiting evaluation"):
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

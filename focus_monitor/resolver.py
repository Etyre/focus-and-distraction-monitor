"""Thread resolver: a small LLM applies judgment where the state machine used to guess.

A switch labelled "continuation" is ambiguous: it can continue the person's focused WORK, or
continue time AWAY from it (an ongoing detour, sliding back into a video, ...). Hand-coded
context-tracking rules kept mislabelling these (a read/note loop alternates two windows; a
recently-visited detour poisons returns to legitimate work). So the decision is made by a model:
Haiku sees the recent activity log, the person's Rules doc, and Toggl, and stores a verdict per
switch (thread_calls). Hand-coded logic keeps only two jobs - categorical rules the Rules doc
states outright (passive domains), and the escalation triggers below that decide WHEN judgment
is needed."""
from __future__ import annotations

import datetime as dt
import logging
import time

from pydantic import BaseModel, Field

from . import db
from .config import Config, load_config

log = logging.getLogger("resolver")

MODEL = "claude-haiku-4-5"
PRICE_IN, PRICE_OUT = 1.0 / 1e6, 5.0 / 1e6
PRICE_CW, PRICE_CR = 1.25 / 1e6, 0.1 / 1e6

SYSTEM = """You are the attention-thread resolver for a personal focus monitor. You answer ONE \
question: the person just switched windows, the switch was classified "continuation" (nothing new \
deliberately started) - so WHAT is being continued? Answer:
- "work": the TO window belongs to the person's current focused-work thread. A read/note loop \
(text <-> notes for the same piece) is all work. Returning to the work after a brief detour is work.
- "interruption": the TO window continues (or slides back into) task-adjacent time away from the \
work - reactive messages, an unrelated errand, an off-thread question.
- "distraction": the TO window continues stimulation-seeking - a video session, feeds, aimless \
browsing. Returning to an ongoing video or feed after glancing at notes is distraction, not work.

Judge from the activity log: what was the person's work thread over the last half hour, and does \
the TO window serve it? Be skeptical of "work" for entertainment/consumption content: this \
person's explicit calibration is that the system has been far too optimistic about focus.

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}

Known distraction domains (strong prior, not proof): {distraction_domains}
"""


class ThreadJudgement(BaseModel):
    reasoning: str = Field(description="1-2 sentences: what the current work thread is and why the TO window does or doesn't belong to it.")
    thread: str = Field(description="Exactly one of: 'work', 'interruption', 'distraction'.")


_resolver = None


def _get():
    global _resolver
    if _resolver is None:
        _resolver = ThreadResolver(load_config())
    return _resolver


class ThreadResolver:
    def __init__(self, cfg: Config):
        import anthropic
        from .config import load_api_key
        load_api_key()
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials")
        from .classifiers.claude_vision import _rules_doc
        self.system = SYSTEM.format(rules=_rules_doc(),
                                    about_me=cfg.about_me.strip() or "(no description given)",
                                    distraction_domains=", ".join(cfg.distraction_domains) or "(none)")

    def resolve(self, conn, switch_id: int) -> str | None:
        from . import stats, summaries
        sw = conn.execute("SELECT s.ts, g.app, g.domain, g.title, g.url FROM switches s "
                          "JOIN segments g ON g.id=s.to_segment WHERE s.id=?", (switch_id,)).fetchone()
        if not sw:
            return None
        segs = stats.labelled_segments(conn, sw["ts"] - 1800, sw["ts"] + 120)
        lines = [summaries.seg_line(s) for s in segs if s["duration"] > 0][-16:]
        prompt = ("Recent activity (oldest first; state and switch labels as currently estimated):\n"
                  + "\n".join(lines)
                  + f"\n\nSWITCH UNDER JUDGMENT at {dt.datetime.fromtimestamp(sw['ts']).strftime('%H:%M:%S')}: "
                  + f"TO {sw['app']} {sw['domain'] or ''} | {(sw['title'] or '')[:80]}\n"
                  + "What is being continued?")
        resp = self.client.beta.messages.parse(
            model=MODEL, max_tokens=300,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_format=ThreadJudgement)
        if resp.parsed_output is None:
            return None
        j: ThreadJudgement = resp.parsed_output
        thread = j.thread.strip().lower()
        if thread not in ("work", "interruption", "distraction"):
            log.warning("bad thread verdict %r for switch %d", j.thread, switch_id)
            return None
        u = resp.usage
        cost = ((u.input_tokens or 0) * PRICE_IN + (u.output_tokens or 0) * PRICE_OUT
                + (getattr(u, "cache_creation_input_tokens", 0) or 0) * PRICE_CW
                + (getattr(u, "cache_read_input_tokens", 0) or 0) * PRICE_CR)
        conn.execute("INSERT OR REPLACE INTO thread_calls(switch_id, thread, rationale, cost_usd, created)"
                     " VALUES (?,?,?,?,?)", (switch_id, thread, j.reasoning, cost, time.time()))
        log.info("switch %d: %s ($%.4f) - %s", switch_id, thread, cost, j.reasoning[:80])
        return thread


def needs_judgment(conn, cfg: Config, switch_id: int) -> bool:
    """Escalation triggers - hand-coded logic decides only WHEN judgment is needed, never what
    the judgment is. A continuation switch needs the resolver when:
    (a) the person was in a detour just before it (is this a recovery, or more detour?), or
    (b) the TO window is on a distraction/passive domain (is this really still work?), or
    (c) the TO context held a detour state in the last 30 min (is this sliding back in?)."""
    from . import stats
    e = conn.execute("SELECT label FROM ensemble WHERE switch_id=?", (switch_id,)).fetchone()
    if not e or e["label"] != "continuation":
        return False
    if conn.execute("SELECT 1 FROM thread_calls WHERE switch_id=?", (switch_id,)).fetchone():
        return False
    if conn.execute("SELECT 1 FROM reviews WHERE switch_id=?", (switch_id,)).fetchone():
        return False  # human verdict outranks the resolver anyway
    sw = conn.execute("SELECT s.ts, g.app, g.domain FROM switches s JOIN segments g ON g.id=s.to_segment"
                      " WHERE s.id=?", (switch_id,)).fetchone()
    if not sw:
        return False
    dom = (sw["domain"] or "")
    if any(dom == d or dom.endswith("." + d) for d in cfg.distraction_domains + cfg.passive_domains):
        return True
    segs = stats.labelled_segments(conn, sw["ts"] - 1800, sw["ts"] + 1)
    before = [s for s in segs if s["clip_end"] <= sw["ts"] + 1 and s["duration"] > 0]
    if before and before[-1]["state"] in ("interrupted", "distracted"):
        return True
    return any(s["state"] in ("interrupted", "distracted") and (s["app"], s.get("domain")) == (sw["app"], sw["domain"])
               for s in before)


def maybe_resolve(conn, cfg: Config, switch_id: int) -> None:
    """Called from the classifier loop after a switch is ensembled."""
    if db.spend_today(conn) >= cfg.daily_budget_usd:
        return
    if not needs_judgment(conn, cfg, switch_id):
        return
    try:
        _get().resolve(conn, switch_id)
    except Exception:
        log.exception("thread resolution failed for switch %d", switch_id)

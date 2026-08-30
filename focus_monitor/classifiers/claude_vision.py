"""Claude Opus 5 vision classifier: looks at the before/after screenshots plus context."""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from .. import LABEL_HELP, LABELS, db
from ..config import (DATA_DIR, PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT,
                      Config)
from .base import Classifier, SwitchContext, normalize, seg_desc

log = logging.getLogger("claude")


class Judgement(BaseModel):
    reasoning: str = Field(description="2-4 sentences: what the person was doing before, what they "
                                       "switched to, and why this is/isn't a change of attention.")
    p_continuation: float = Field(ge=0, le=1)
    p_interruption: float = Field(ge=0, le=1)
    p_distraction: float = Field(ge=0, le=1)
    p_task_change: float = Field(ge=0, le=1, description="'focus start': focused work begins or resumes with this switch")
    activity: str = Field(description="What the person is doing in the TO window: one of 'reading', "
                          "'writing' (composing an essay/notes/doc), 'coding', 'task_execution' (working "
                          "through discrete tasks: email, forms, admin, calendar), 'planning' (organizing "
                          "thoughts, choosing goals, metacognition, journaling, reflecting), 'browsing' "
                          "(feeds/shopping/aimless), or 'other'.")


SYSTEM_TEMPLATE = """You are an attention-monitoring assistant. A person is tracking their own focus on their Mac. \
Every time the active window changes, you judge what kind of attention event the switch was. \
You see a text log of the surrounding activity (apps, window titles, URLs, durations, and what they did \
afterwards) plus screenshots taken shortly before and shortly after the switch.

Classify the switch into exactly these categories (give probabilities that sum to 1):
{label_help}

Guidance:
- Switching windows is NOT by itself a change of attention. Reading <-> notes, editor <-> terminal, \
draft <-> reference, or asking an AI assistant a question that grows out of the current work are all \
"continuation". "continuation" also covers staying in the same non-focused state: feed -> another feed \
while distracted, or Slack -> Signal while already on a messaging detour. It means "nothing changed".
- "interruption" is leaving focused work for a quick unrelated detour (a stray question, reactive \
email/message checking, a notification) - usually short, usually followed by returning.
- "distraction" is leaving focused work to seek stimulation without an objective: feeds, aimless \
shopping, dating apps, a blocked URL. Duration is not the test - intent is.
- "task_change" means FOCUS START: focused work begins or resumes with this switch - returning to the \
task after a detour or distraction, deliberately starting a different piece of work, or genuinely \
settling into absorbed reading with note-taking after having arrived somewhere as a distraction.
- Known distraction domains are a WEAK prior. Opening a specific post or comment reached from the \
current reading thread is continuation; moving on to a different essay/thread and settling into it \
is a focus start (a "focus transition"). Only feed-scrolling / aimless browsing is distraction.
- Adding a spare thought to their DAILY NOTES / journal in Roam or Logseq is quick capture and part \
of staying focused - it is NEVER an interruption or distraction. Treat a switch into a daily-notes \
page as continuation.
- Spans are containers: a detour under 5 minutes is an event inside the span, so do not hesitate to \
call a brief glance an interruption - it will not end their span. Label what the switch IS.
- The switches that matter most are the ones that LEAVE an ongoing focus span: is this switch still the \
work (continuation), a detour (interruption), or stimulation-seeking (distraction)?
- Use the screenshots to see content: an email thread about the current project is continuation; \
an unrelated newsletter is interruption; a block-page or infinite feed suggests distraction.
- Be calibrated. If it is genuinely ambiguous, spread probability rather than guessing confidently.

About this person (in their own words):
{about_me}

Known distraction domains: {distraction_domains}
Work apps: {work_apps}. Notes apps: {notes_apps}.
"""


def _b64(path_name: str | None) -> str | None:
    if not path_name:
        return None
    p = DATA_DIR / "screenshots" / path_name
    if not p.exists():
        return None
    return base64.standard_b64encode(p.read_bytes()).decode()


def _cost(usage) -> float:
    c = (getattr(usage, "input_tokens", 0) or 0) * PRICE_INPUT
    c += (getattr(usage, "output_tokens", 0) or 0) * PRICE_OUTPUT
    c += (getattr(usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
    c += (getattr(usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ
    return c


class ClaudeVisionClassifier(Classifier):
    name = "claude"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials (set the API key from the menu bar)")
        self.examples_text = ""
        self.last_activity = None
        self.system = SYSTEM_TEMPLATE.format(
            label_help="\n".join(f"- {k}: {v}" for k, v in LABEL_HELP.items()),
            about_me=cfg.about_me.strip() or "(no description given)",
            distraction_domains=", ".join(cfg.distraction_domains) or "(none)",
            work_apps=", ".join(cfg.work_apps) or "(none)",
            notes_apps=", ".join(cfg.notes_apps) or "(none)",
        )

    def learn(self, conn) -> None:
        rows = db.reviews_with_context(conn, limit=self.cfg.few_shot_reviews)
        if not rows:
            self.examples_text = ""
            return
        lines = ["Examples of switches this person has reviewed themselves (most recent first). "
                 "Learn their personal standards from these:"]
        for r in rows:
            fr = {"app": r["from_app"] or "?", "title": r["from_title"] or "", "url": r["from_url"] or "",
                  "start": r["from_start"] or r["ts"], "end": r["from_end"], "idle": 0}
            to = {"app": r["to_app"], "title": r["to_title"], "url": r["to_url"],
                  "start": r["to_start"], "end": r["to_end"], "idle": 0}
            note = f' - note: "{r["note"]}"' if r.get("note") else ""
            lines.append(f"* {seg_desc(fr)}  ->  {seg_desc(to)}  => {r['label']}{note}")
        self.examples_text = "\n".join(lines)

    def _system_blocks(self) -> list[dict]:
        blocks = [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
        if self.examples_text:
            blocks.append({"type": "text", "text": self.examples_text, "cache_control": {"type": "ephemeral"}})
        return blocks

    def predict(self, ctx: SwitchContext):
        content: list[dict] = []
        before_img = _b64(ctx.from_seg["last_screenshot"] if ctx.from_seg else None)
        after_img = _b64(ctx.to_seg["first_screenshot"])
        if before_img:
            content.append({"type": "text", "text": "Screenshot from BEFORE the switch:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": before_img}})
        if after_img:
            content.append({"type": "text", "text": "Screenshot from shortly AFTER the switch:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": after_img}})
        content.append({"type": "text", "text": ctx.narrative() + "\n\nClassify the FROM -> TO switch."})

        try:
            resp = self.client.beta.messages.parse(
                model=self.cfg.claude_model,
                max_tokens=2048,
                system=self._system_blocks(),
                messages=[{"role": "user", "content": content}],
                output_format=Judgement,
                output_config={"effort": self.cfg.claude_effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            log.warning("rate limited: %s", e)
            return None
        except anthropic.APIStatusError as e:
            log.error("API error %s: %s", e.status_code, e.message)
            return None
        except anthropic.APIConnectionError as e:
            log.error("connection error: %s", e)
            return None

        cost = _cost(resp.usage)
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            log.warning("no parsed output (stop_reason=%s)", resp.stop_reason)
            return None
        j: Judgement = resp.parsed_output
        self.last_activity = j.activity
        probs = normalize({"continuation": j.p_continuation, "interruption": j.p_interruption,
                           "distraction": j.p_distraction, "task_change": j.p_task_change})
        return probs, j.reasoning, cost

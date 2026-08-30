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
        ctx_blocks = self._context_blocks(conn, switch_id)
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
        resp = self.client.messages.create(
            model=self.cfg.claude_model, max_tokens=1024,
            system=self.system,
            messages=api_msgs,
            output_config={"effort": "low"},
        )
        if resp.stop_reason == "refusal":
            return "(Claude declined to answer that.)"
        return "".join(b.text for b in resp.content if b.type == "text").strip() or "(no reply)"

"""Per-call usage log that the Claude usage dashboard reads.

Every model call is appended as one JSON line to ~/.claude/app-usage/Focus-monitor.jsonl, so the
subscription-backed calls (which no Anthropic API reports and which leave no transcript, since the
SDK runs with session persistence off) still show up next to the API-key ones. The app name matches
the API key's name in the Console, so the dashboard shows both under one app.

Line shape: {"ts": ISO-8601 local time with offset, "app": "Focus-monitor", "source": "agent-sdk"|"api",
"billing": "subscription"|"api", "model": id, "site": caller, "usage": {input_tokens, output_tokens,
cache_read_input_tokens, cache_creation_input_tokens}, "cost_usd": API-equivalent price}.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("usage_log")

APP_NAME = "Focus-monitor"
LOG_DIR = Path(os.environ.get("CLAUDE_APP_USAGE_DIR") or "~/.claude/app-usage").expanduser()
LOG_FILE = LOG_DIR / f"{APP_NAME}.jsonl"


def _line(ts: float, site: str, backend: str, model: str, usage, cost_usd: float) -> str:
    g = (lambda k: usage.get(k) or 0) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0) or 0)
    rec = {
        "ts": dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="microseconds"),
        "app": APP_NAME,
        "source": "agent-sdk" if backend == "subscription" else "api",
        "billing": "subscription" if backend == "subscription" else "api",
        "model": model or "unknown",
        "site": site,
        "usage": {"input_tokens": int(g("input_tokens")), "output_tokens": int(g("output_tokens")),
                  "cache_read_input_tokens": int(g("cache_read_input_tokens")),
                  "cache_creation_input_tokens": int(g("cache_creation_input_tokens"))},
        "cost_usd": round(float(cost_usd or 0.0), 6),
    }
    return json.dumps(rec, separators=(",", ":")) + "\n"


def append(ts: float, site: str, backend: str, model: str, usage, cost_usd: float) -> None:
    """Append one successful call. Never raises: logging must not break a call."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(_line(ts, site, backend, model, usage, cost_usd))
    except Exception:
        log.exception("usage log append failed")


def _last_logged_ts() -> float:
    """Timestamp of the last line in the log, or 0 when there is none."""
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace").strip().splitlines()
        for line in reversed(tail):
            try:
                return dt.datetime.fromisoformat(json.loads(line)["ts"]).timestamp()
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        pass
    return 0.0


def backfill(conn) -> int:
    """Export llm_calls rows newer than the log's last line (so history before this logging existed,
    and anything missed while the app was down, lands in the log once). Returns rows written."""
    try:
        since = _last_logged_ts()
        rows = conn.execute(
            "SELECT ts, site, backend, model, input_tokens, output_tokens, cache_read, cache_write, cost_usd"
            " FROM llm_calls WHERE ok = 1 AND ts > ? ORDER BY ts", (since,)).fetchall()
        if not rows:
            return 0
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            for ts, site, backend, model, inp, out, cr, cw, cost in rows:
                usage = {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": cr,
                         "cache_creation_input_tokens": cw}
                f.write(_line(ts, site or "", backend or "api", model or "", usage, cost or 0.0))
        log.info("usage log: backfilled %d calls from llm_calls", len(rows))
        return len(rows)
    except Exception:
        log.exception("usage log backfill failed")
        return 0

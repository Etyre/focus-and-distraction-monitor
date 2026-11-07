"""One door for every structured-output model call, with two backends behind it.

  api           the Anthropic SDK (client.beta.messages.parse), billed per token to the API key
  subscription  the Claude Agent SDK driving the Claude Code binary on the person's claude.ai
                login (Pro/Max): it draws on the subscription's usage limits, not the key

The backend is cfg.llm_backend. On the subscription, the first limit error switches every call
to the API key until the limit's reset time (parsed from the error when it names one, else
cfg.subscription_retry_min from now), after which the subscription is tried again. Any other
SDK failure falls back to the API for that one call. Chat calls that use tools stay on the API.

Every call is logged to llm_calls (backend, tokens, API-equivalent cost, latency) so the two
can be compared. The cost written into the existing cost_usd columns is the API-equivalent
price on BOTH backends, so the daily budget keeps working as a usage governor either way;
llm_calls says what was actually billed to the key.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from . import db
from .config import (DATA_DIR, PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT,
                     load_config)

log = logging.getLogger("llm")

SDK_TIMEOUT_S = 300.0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class Result:
    """What a call site reads: the same three attributes the Anthropic SDK response has."""
    parsed_output: Any
    usage: Any
    stop_reason: str | None
    backend: str = "api"
    latency_ms: int = 0
    cost_usd: float = 0.0


class SubscriptionLimit(Exception):
    def __init__(self, message: str, until: float):
        super().__init__(message)
        self.until = until


def call_cost(usage) -> float:
    """API-equivalent price of a call, from its token counts."""
    g = (lambda k: (getattr(usage, k, 0) or 0)) if not isinstance(usage, dict) else (lambda k: usage.get(k) or 0)
    return (g("input_tokens") * PRICE_INPUT + g("output_tokens") * PRICE_OUTPUT
            + g("cache_creation_input_tokens") * PRICE_CACHE_WRITE
            + g("cache_read_input_tokens") * PRICE_CACHE_READ)


# ---------------------------------------------------------------- state (shared across threads)
_lock = threading.Lock()
_conn = None
_blocked_until: float | None = None   # None = not loaded yet
_last_error: str | None = None
_cfg = None
_cfg_at = 0.0


def _db():
    global _conn
    with _lock:
        if _conn is None:
            _conn = db.connect()
        return _conn


def _config():
    """config.toml re-read every minute, so flipping llm_backend needs no restart."""
    global _cfg, _cfg_at
    if _cfg is None or time.time() - _cfg_at > 60:
        _cfg, _cfg_at = load_config(), time.time()
    return _cfg


def blocked_until() -> float:
    global _blocked_until
    if _blocked_until is None:
        try:
            _blocked_until = float(db.kv_get(_db(), "llm.subscription_blocked_until") or 0.0)
        except Exception:
            _blocked_until = 0.0
    return _blocked_until


def _block(until: float, why: str) -> None:
    global _blocked_until, _last_error
    _blocked_until, _last_error = until, why
    try:
        db.kv_set(_db(), "llm.subscription_blocked_until", str(until))
    except Exception:
        log.exception("could not persist the subscription block")
    log.warning("subscription limit hit (%s) - using the API key until %s", why,
                dt.datetime.fromtimestamp(until).strftime("%H:%M"))


def subscription_available(cfg=None) -> bool:
    cfg = cfg or _config()
    return cfg.llm_backend == "subscription" and time.time() >= blocked_until()


def _log(site: str, backend: str, model: str, effort: str | None, usage, latency_ms: int,
         ok: bool, error: str | None) -> None:
    try:
        c = _db()
        with _lock:
            c.execute("INSERT INTO llm_calls(ts, site, backend, model, effort, input_tokens, output_tokens,"
                      " cache_read, cache_write, cost_usd, latency_ms, ok, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (time.time(), site, backend, model, effort,
                       getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0,
                       getattr(usage, "cache_read_input_tokens", 0) or 0,
                       getattr(usage, "cache_creation_input_tokens", 0) or 0,
                       call_cost(usage) if usage is not None else 0.0, latency_ms, int(ok), error))
    except Exception:
        log.exception("llm_calls log failed")


# ---------------------------------------------------------------- the door
def parse(client, *, site: str = "", cfg=None, **kw) -> Result:
    """Drop-in for client.beta.messages.parse(**kw). Picks the backend; returns an object with
    parsed_output / usage / stop_reason (plus backend, latency_ms, cost_usd)."""
    cfg = cfg or _config()
    model = kw.get("model", "")
    effort = (kw.get("output_config") or {}).get("effort")
    if subscription_available(cfg):
        t0 = time.time()
        try:
            r = _sdk_parse(cfg, **kw)
            _log(site, "subscription", model, effort, r.usage, r.latency_ms, True, None)
            return r
        except SubscriptionLimit as e:
            _block(e.until, str(e))
            _log(site, "subscription", model, effort, None, int((time.time() - t0) * 1000), False,
                 f"limit: {e}"[:300])
        except Exception as e:  # noqa: BLE001 - any SDK failure: this call goes to the API
            global _last_error
            _last_error = f"{type(e).__name__}: {e}"[:300]
            log.warning("subscription call failed at %s (%s) - using the API for this call", site, _last_error)
            _log(site, "subscription", model, effort, None, int((time.time() - t0) * 1000), False, _last_error)
    t0 = time.time()
    try:
        resp = client.beta.messages.parse(**kw)
    except Exception as e:
        _log(site, "api", model, effort, None, int((time.time() - t0) * 1000), False,
             f"{type(e).__name__}: {e}"[:300])
        raise
    ms = int((time.time() - t0) * 1000)
    r = Result(resp.parsed_output, resp.usage, resp.stop_reason, "api", ms, call_cost(resp.usage))
    _log(site, "api", model, effort, resp.usage, ms, True, None)
    return r


# ---------------------------------------------------------------- the subscription backend
_LIMIT_RE = re.compile(r"(usage limit|hit your limit|limit reached|rate_limit|rate limit)", re.I)
_RESET_RE = re.compile(r"reset\w*\s*(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.I)


def _reset_time(text: str, cfg) -> float:
    """The reset time an error names ('resets 7pm (America/Los_Angeles)'), else retry_min from now."""
    m = _RESET_RE.search(text or "")
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
        h = (h % 12) + (12 if ap == "pm" else 0)
        now = dt.datetime.now()
        t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if t <= now:
            t += dt.timedelta(days=1)
        return t.timestamp() + 60
    return time.time() + 60 * float(cfg.subscription_retry_min)


def _system_text(system) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in (system or []))


def _sdk_parse(cfg, *, model, system, messages, output_format, output_config=None, **_ignored) -> Result:
    from claude_agent_sdk import (ClaudeAgentOptions, ClaudeSDKError, ResultMessage, SystemMessage,
                                  query)
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("the subscription backend takes one user message")
    content = messages[0]["content"]
    effort = (output_config or {}).get("effort")
    opts = ClaudeAgentOptions(
        system_prompt=_system_text(system), model=model, effort=effort,
        output_format={"type": "json_schema", "schema": output_format.model_json_schema()},
        tools=[], allowed_tools=[], setting_sources=[], max_turns=1, permission_mode="dontAsk",
        cwd=str(DATA_DIR),
        # The app's process carries ANTHROPIC_API_KEY for the api backend; the Claude Code
        # binary would prefer it over the login, so the child gets it blanked.
        env={"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},
        # No transcript per call: a thousand calls a day with screenshots in them would
        # otherwise pile up under ~/.claude/projects.
        extra_args={"no-session-persistence": None},
    )

    async def prompt():
        yield {"type": "user", "message": {"role": "user", "content": content}}

    async def run():
        res, limit_text = None, None
        try:
            async for m in query(prompt=prompt(), options=opts):
                if isinstance(m, SystemMessage) and m.subtype == "api_retry":
                    # The binary would back off and retry for minutes; a limit is not transient
                    # for us - the API key is right there.
                    if (m.data or {}).get("error") == "rate_limit":
                        limit_text = f"api_retry rate_limit (status {(m.data or {}).get('error_status')})"
                        break
                elif isinstance(m, ResultMessage):
                    res = m
        except ClaudeSDKError as e:
            text = str(e)
            if _LIMIT_RE.search(text):
                raise SubscriptionLimit(text[:300], _reset_time(text, cfg)) from e
            raise
        return res, limit_text

    t0 = time.time()
    res, limit_text = _run_async(asyncio.wait_for(run(), SDK_TIMEOUT_S))
    ms = int((time.time() - t0) * 1000)
    if limit_text:
        raise SubscriptionLimit(limit_text, _reset_time(limit_text, cfg))
    if res is None:
        raise RuntimeError("no result from the Claude Code binary")
    text = " ".join(str(x) for x in ((res.result or ""), *(res.errors or [])))
    if res.is_error or res.api_error_status == 429 or (res.terminal_reason == "api_error" and _LIMIT_RE.search(text)):
        if res.api_error_status == 429 or _LIMIT_RE.search(text):
            raise SubscriptionLimit(text[:300] or f"status {res.api_error_status}", _reset_time(text, cfg))
        raise RuntimeError(f"error result ({res.subtype}/{res.terminal_reason}): {text[:200]}")
    u = res.usage or {}
    usage = Usage(int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
                  int(u.get("cache_creation_input_tokens") or 0), int(u.get("cache_read_input_tokens") or 0))
    parsed = output_format.model_validate(res.structured_output) if res.structured_output is not None else None
    stop = "refusal" if res.stop_reason == "refusal" else ("end_turn" if parsed is not None else res.stop_reason)
    return Result(parsed, usage, stop, "subscription", ms, call_cost(usage))


def _run_async(coro):
    """asyncio.run, from a plain thread or from one that already has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    out: dict = {}

    def runner():
        try:
            out["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            out["e"] = e
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "e" in out:
        raise out["e"]
    return out["v"]


# ---------------------------------------------------------------- status
def status() -> dict:
    from .config import day_bounds, logical_date
    cfg = _config()
    c = _db()
    start = day_bounds(logical_date())[0]
    today = {}
    with _lock:
        rows = c.execute("SELECT backend, COUNT(*), SUM(cost_usd), SUM(ok), AVG(latency_ms) FROM llm_calls"
                         " WHERE ts >= ? GROUP BY backend", (start,)).fetchall()
    for r in rows:
        today[r[0]] = {"calls": r[1], "cost_equiv_usd": round(r[2] or 0.0, 3), "ok": r[3],
                       "avg_latency_ms": int(r[4] or 0)}
    bu = blocked_until()
    return {"backend": cfg.llm_backend,
            "subscription_blocked_until": bu if bu > time.time() else None,
            "using_now": "subscription" if subscription_available(cfg) else "api",
            "last_error": _last_error, "today": today}

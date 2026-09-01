"""Chunked segment evaluator: the model applies judgment over whole stretches of activity.

Every ~15 minutes (lazily, so after-context exists), one Claude call evaluates a chunk of
segments together, with surrounding context and sampled screenshots. For each segment it
judges:
  1. CONTENT - what kind of activity the segment is:
     creative | task | reading | planning | other | passive
     ("task" describes the activity's nature regardless of Toggl - Toggl is span-level
     evidence of deliberate focus, not evidence of task-nature.)
  2. SWITCH - what the switch INTO the segment was, attention-wise:
     continuation (no shift of the object of attention) | interruption (a shift) |
     return (coming back to the pre-interruption object)
  3. For interruptions, the KIND:
     self_distraction (stimulation-seeking; usual when the target is passive consumption) |
     focus_start (deliberately starting a new piece of focused work) |
     detour (a reactive/task-adjacent shift that is neither)

The user audits and edits these (seg_reviews); edits are authoritative for state derivation
and are fed back to the evaluator as few-shot corrections, so it learns over time."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from pydantic import BaseModel, Field

from . import db
from .config import (PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, Config,
                     load_config)

log = logging.getLogger("evaluator")

from .seg_models import CONTENTS, HEADS, KINDS, SWITCHES  # single source for the label space

LAZY_S = 15 * 60          # evaluate only segments at least this old (after-context exists)
CHUNK_S = 20 * 60         # evaluate about this much activity per call
CTX_S = 10 * 60           # context shown on each side of the chunk
MAX_SHOTS = 6


class SegJudgement(BaseModel):
    segment_id: int
    content: str = Field(description="One of: creative, task, reading, planning, other, passive, meeting, idle, transition.")
    switch: str = Field(description="The switch INTO this segment: continuation, interruption, or return.")
    interruption_kind: str | None = Field(default=None, description="Only when switch is 'interruption': self_distraction, focus_start, or detour.")
    uncertain: bool = Field(default=False, description="True only when you are genuinely unsure of the content or switch judgment and a human should check it. Use sparingly - these are flagged for the person to review.")
    note: str = Field(description="One short sentence of reasoning.")


class ChunkJudgement(BaseModel):
    segments: list[SegJudgement]


SYSTEM = """You are the segment evaluator for a personal attention monitor. You receive a stretch \
of one person's computer activity - an ordered log of segments (one active window each), with \
surrounding context before and after, Toggl entries, and sampled screenshots - and you judge every \
segment marked EVALUATE. For each one output:

1. content - what the activity IS:
   - creative: making something (writing an essay/doc, coding, video editing).
   - task: executing discrete tasks (email, forms, errands, admin, logistics). Judge the \
activity's nature only - whether a Toggl timer is running is NOT part of this judgment.
   - reading: reading posts/papers/books. (Whether notes are being taken matters for spans, not \
for this label - reading without notes is still 'reading' content, unless it is really passive \
browsing/consumption.)
   - planning: metacognition - journaling, prioritizing, reviewing goals, deciding what to work on.
   - other: real engaged activity fitting none of the above.
   - passive: passive consumption - watching videos, scrolling feeds, browsing without an \
objective, reading comics/entertainment. When in doubt between 'other' and 'passive', lean passive: \
this person's calibration is that the system has been far too optimistic about focus.
   - meeting: a live conversation - video call (Zoom/Meet/FaceTime), call notes or live transcript \
(e.g. Otter) during one.
   - idle: the window was open but keyboard/mouse input was idle for most of the segment and \
nothing suggests engaged reading or watching - the person was likely away or disengaged. \
Log lines show "(input idle ...)" when recorded. A mostly-input-idle segment can NOT be task, \
creative, or planning (those require interaction); it can still be reading or passive if they \
were plausibly reading/watching the screen.
   - transition: rapidly passing through desktops, windows, or tabs on the WAY to a specific \
target - travel, not attention. Typically a chain of very short segments ending somewhere the \
person settles. Transitions are never attention shifts in themselves; their switch is \
'continuation' of whatever the person is heading toward.
2. switch - did the switch INTO this segment shift the person's OBJECT OF ATTENTION?
   - continuation: same object (window-hopping within one piece of work counts - text<->notes, \
editor<->terminal, stepping through desktops to find something; also staying within one ongoing \
distraction session).
   - interruption: the object of attention changed.
   - return: coming back to the object they were on before an interruption.
3. interruption_kind - ONLY for switch='interruption':
   - self_distraction: seeking stimulation without clear intent (usually when the target content \
is passive; going to a blocked site; absent-mindedly opening feeds/shopping/dating apps).
   - focus_start: deliberately starting a NEW piece of focused work (often marked by starting a \
Toggl entry, opening a todo list, or settling into new material with notes).
   - detour: a reactive or task-adjacent shift that is neither - answering a message, a stray \
question, a quick errand.

Judge each segment in the context of the whole stretch - what came before AND after it. A flip to \
a notes app mid-video does not make the video session work; a quick message check inside a writing \
session is a detour, not the end of the work.

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}

{examples}"""


_ev = None


def _get():
    global _ev
    if _ev is None:
        _ev = ChunkEvaluator(load_config())
    return _ev


class ChunkEvaluator:
    def __init__(self, cfg: Config):
        import anthropic
        from .config import load_api_key
        load_api_key()
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials")
        self._examples_stamp = None
        self._system = None

    def _examples(self, conn) -> str:
        rows = conn.execute("""
            SELECT r.segment_id, r.content, r.switch_label, r.interruption_kind, r.note,
                   g.app, g.domain, g.title
            FROM seg_reviews r JOIN segments g ON g.id = r.segment_id
            ORDER BY r.created DESC LIMIT 20""").fetchall()
        if not rows:
            return ""
        lines = ["The person has corrected past judgments - follow these precedents:"]
        for r in rows:
            what = f"{r['app']} {r['domain'] or ''} | {(r['title'] or '')[:60]}"
            verdict = " / ".join(v for v in (r["content"], r["switch_label"], r["interruption_kind"]) if v)
            note = f" ({r['note']})" if r["note"] else ""
            lines.append(f"- {what} -> {verdict}{note}")
        return "\n".join(lines)

    def _system_blocks(self, conn) -> list[dict]:
        stamp = conn.execute("SELECT COALESCE(MAX(created),0) FROM seg_reviews").fetchone()[0]
        if self._system is None or stamp != self._examples_stamp:
            from .classifiers.claude_vision import _rules_doc
            self._system = SYSTEM.format(rules=_rules_doc(),
                                         about_me=self.cfg.about_me.strip() or "(no description given)",
                                         examples=self._examples(conn))
            self._examples_stamp = stamp
        return [{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}]

    def evaluate_chunk(self, conn, t0: float, t1: float, extra: str | None = None) -> int:
        """Evaluate all real segments starting in [t0, t1). Local models predict first (and are
        stored for track-record scoring); segments where the locals have EARNED confident
        deferral skip the LLM; the rest go into one LLM call, and the resolved judgment is the
        accuracy-weighted ensemble of all models. Returns how many segments were judged."""
        from . import seg_models, stats, summaries, toggl
        from .classifiers.claude_vision import _b64
        segs = stats.labelled_segments(conn, t0 - CTX_S, t1 + CTX_S)
        target = [s for s in segs if t0 <= s["clip_start"] < t1 and not s["idle"]
                  and not s.get("neutral") and s["duration"] > 0]
        if not target:
            return 0
        # -- local models + deferral decision ---------------------------------------------
        sc = seg_models.scores(conn)
        real = [s for s in segs if not s["idle"] and not s.get("neutral") and s["duration"] > 0]
        local: dict = {}   # id -> (per-model preds, combined dist per head)
        deferred: set = set()
        now = time.time()
        for s in target:
            i = real.index(s)
            prev = real[i - 1] if i > 0 else None
            lp = seg_models.local_predict(conn, self.cfg, s, prev)
            for m, p in lp.items():
                seg_models.store_prediction(conn, s["id"], m, p)
            combined = {h: seg_models.combine_head({m: lp[m][h] for m in lp},
                                                   seg_models.weights_for(sc, h, list(lp)))
                        for h in HEADS}
            local[s["id"]] = (lp, combined)
            if extra is None and seg_models.can_defer(sc, combined, self.cfg):
                deferred.add(s["id"])
        for s in target:
            if s["id"] not in deferred:
                continue
            _, comb = local[s["id"]]
            c = max(comb["content"], key=comb["content"].get)
            sw = max(comb["switch"], key=comb["switch"].get)
            k = max(comb["kind"], key=comb["kind"].get) if sw == "interruption" else None
            conn.execute(
                "INSERT OR REPLACE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, probs, created)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (s["id"], c, sw, k, 0, "(deferred to local models - earned track record)",
                 json.dumps(comb), now))
        target = [s for s in target if s["id"] not in deferred]
        if not target:
            log.info("chunk %s-%s: all %d segments deferred to local models ($0)",
                     dt.datetime.fromtimestamp(t0).strftime("%H:%M"),
                     dt.datetime.fromtimestamp(t1).strftime("%H:%M"), len(deferred))
            conn.execute("INSERT INTO eval_runs(t0, t1, n_segments, cost_usd, created) VALUES (?,?,?,?,?)",
                         (t0, t1, len(deferred), 0.0, now))
            return len(deferred)
        target_ids = {s["id"] for s in target}
        lines = []
        for s in segs:
            if s["duration"] <= 0:
                continue
            mark = f"EVALUATE id={s['id']}: " if s["id"] in target_ids else "(context) "
            lines.append(mark + summaries.seg_line(s))
        tg = toggl.entries_between(conn, t0 - CTX_S, t1 + CTX_S)
        tgtxt = "\n".join(f"  {dt.datetime.fromtimestamp(e['start']).strftime('%H:%M')}"
                          f"-{dt.datetime.fromtimestamp(e['stop'] or time.time()).strftime('%H:%M')}: "
                          f"{e['description'] or '(unnamed)'}" for e in tg) or "  (none)"
        content: list[dict] = []
        qs = ",".join("?" * len(target_ids))
        shot_rows = [dict(r) for r in conn.execute(
            f"SELECT segment_id, ts, path, display FROM segment_shots WHERE segment_id IN ({qs}) ORDER BY ts",
            list(target_ids)).fetchall()]
        if not shot_rows:  # segments recorded before per-shot history existed
            shot_rows = [{"segment_id": s["id"], "ts": s["clip_start"], "path": s["first_screenshot"], "display": 0}
                         for s in target if s.get("first_screenshot")]
        app_of = {s["id"]: s["app"] for s in target}
        step = max(1, len(shot_rows) // MAX_SHOTS)
        shot_meta: list[dict] = []
        for r2 in shot_rows[::step][:MAX_SHOTS]:
            b = _b64(r2["path"])
            if b:
                where = f" - EXTERNAL display {r2['display']}" if r2.get("display") else ""
                label = f"Screenshot of segment id={r2['segment_id']} ({dt.datetime.fromtimestamp(r2['ts']).strftime('%H:%M')} {app_of.get(r2['segment_id'], '')}{where}):"
                shot_meta.append({"path": r2["path"], "label": label})
                content.append({"type": "text", "text": label})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        prompt_text = (f"Toggl entries in this window:\n{tgtxt}\n\nActivity log:\n" + "\n".join(lines)
                       + (f"\n\nIMPORTANT - {extra}" if extra else "")
                       + "\n\nJudge every segment marked EVALUATE (all of them, by id).")
        content.append({"type": "text", "text": prompt_text})
        system_text = self._system_blocks(conn)[0]["text"]
        resp = self.client.beta.messages.parse(
            model=self.cfg.claude_model, max_tokens=8000,
            system=self._system_blocks(conn),
            messages=[{"role": "user", "content": content}],
            output_format=ChunkJudgement,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default")
        cost = ((resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
                + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
                + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)
        n = 0
        if resp.parsed_output is not None:
            now = time.time()
            per_seg_cost = cost / max(len(target), 1)
            with db.tx(conn):
                run_id = conn.execute(
                    "INSERT INTO eval_runs(t0, t1, n_segments, cost_usd, created, system, prompt, shots, response, model)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (t0, t1, 0, cost, now, system_text, prompt_text, json.dumps(shot_meta),
                     resp.parsed_output.model_dump_json(), self.cfg.claude_model)).lastrowid
                for j in resp.parsed_output.segments:
                    if j.segment_id not in target_ids:
                        continue
                    c = j.content.strip().lower()
                    sw = j.switch.strip().lower()
                    k = (j.interruption_kind or "").strip().lower() or None
                    if c not in CONTENTS or sw not in SWITCHES or (k is not None and k not in KINDS):
                        log.warning("bad judgement for segment %d: %s/%s/%s", j.segment_id, c, sw, k)
                        continue
                    # Claude's discrete judgment as (pseudo-)probabilities, so it can be scored
                    # and weighted alongside the local models.
                    top_p = 0.55 if j.uncertain else 0.85
                    cp = {"content": seg_models._uniformish(CONTENTS, c, top_p),
                          "switch": seg_models._uniformish(SWITCHES, sw, top_p),
                          "kind": seg_models._uniformish(KINDS, k, top_p) if k else seg_models._uniformish(KINDS)}
                    seg_models.store_prediction(conn, j.segment_id, "claude", cp, per_seg_cost)
                    # Resolved judgment = accuracy-weighted ensemble of every model's output.
                    lp, _ = local.get(j.segment_id, ({}, None))
                    allp = dict(lp); allp["claude"] = cp
                    comb = {h: seg_models.combine_head({m: allp[m][h] for m in allp},
                                                       seg_models.weights_for(sc, h, list(allp)))
                            for h in HEADS}
                    rc = max(comb["content"], key=comb["content"].get)
                    rsw = max(comb["switch"], key=comb["switch"].get)
                    rk = max(comb["kind"], key=comb["kind"].get) if rsw == "interruption" else None
                    unc = int(j.uncertain or max(comb["content"].values()) < 0.5
                              or max(comb["switch"].values()) < 0.5)
                    conn.execute(
                        "INSERT OR REPLACE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, probs, created, run_id)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (j.segment_id, rc, rsw, rk, unc, j.note, json.dumps(comb), now, run_id))
                    n += 1
                # A target the model omitted must not stall the queue forever: store an
                # explicit default the user can audit and correct.
                judged = {j.segment_id for j in resp.parsed_output.segments}
                for s in target:
                    if s["id"] not in judged:
                        conn.execute(
                            "INSERT OR IGNORE INTO seg_evals(segment_id, content, switch_label, interruption_kind, rationale, created, run_id)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (s["id"], "other", "continuation", None, "(model omitted this segment; defaulted)", now, run_id))
                conn.execute("UPDATE eval_runs SET n_segments=? WHERE id=?", (n + len(deferred), run_id))
        log.info("chunk %s-%s: %d/%d segments judged, %d deferred ($%.3f)",
                 dt.datetime.fromtimestamp(t0).strftime("%H:%M"),
                 dt.datetime.fromtimestamp(t1).strftime("%H:%M"), n, len(target), len(deferred), cost)
        return n + len(deferred)


def refine_incoherent(conn, cfg: Config | None = None, max_spans: int = 1) -> int:
    """Model judgments are not sacred: when the coherence check catches a span containing two
    objects of attention, RE-EVALUATE its segments with that finding in hand and correct them
    (the user's own edits always take precedence in state derivation and are never the target).
    Bounded: a stretch already refined twice in 24h falls back to asking the user instead."""
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    rows = conn.execute("SELECT * FROM span_summaries WHERE coherent=0 AND refined=0"
                        " ORDER BY created DESC LIMIT ?", (max_spans,)).fetchall()
    n = 0
    for r in rows:
        conn.execute("UPDATE span_summaries SET refined=1 WHERE id=?", (r["id"],))
        prior = conn.execute(
            "SELECT COUNT(*) FROM span_summaries WHERE refined=1 AND id != ? AND start < ? AND end > ? AND created >= ?",
            (r["id"], r["end"], r["start"], time.time() - 86400)).fetchone()[0]
        if prior >= 2:
            log.info("refinement budget exhausted for %s - leaving for the user",
                     dt.datetime.fromtimestamp(r["start"]).strftime("%H:%M"))
            continue
        try:
            extra = (f"a whole-span coherence review of this stretch found: {r['issue']} "
                     f"Re-judge with that in hand. If a NEW piece of work begins mid-stretch, "
                     f"mark the segment where it begins as switch='interruption', "
                     f"interruption_kind='focus_start'.")
            self_n = _get().evaluate_chunk(conn, r["start"] - 1, r["end"] + 1, extra=extra)
            log.info("coherence-driven re-evaluation of %s-%s: %d segments re-judged",
                     dt.datetime.fromtimestamp(r["start"]).strftime("%H:%M"),
                     dt.datetime.fromtimestamp(r["end"]).strftime("%H:%M"), self_n)
            n += 1
        except Exception:
            log.exception("refinement failed")
    return n


def chat_reply(conn, cfg: Config, segment_id: int, messages: list[dict]) -> str:
    """Talk to "the one who made the decision": reconstitute the EXACT evaluation conversation
    (same system prompt, same user message with the same screenshots, its actual response), then
    continue it with the reviewer's questions - so mistakes can be triangulated and the
    instruction that would have prevented them identified. Falls back to an informed stand-in
    for segments judged by the local models (deferral) or before transcripts were kept."""
    import json as _json
    from .classifiers.claude_vision import _b64
    ev_row = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    run = None
    if ev_row is not None and ev_row["run_id"]:
        run = conn.execute("SELECT * FROM eval_runs WHERE id=?", (ev_row["run_id"],)).fetchone()
    seg = conn.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
    who = f"segment id={segment_id} ({seg['app']} | {(seg['title'] or '')[:60]})" if seg else f"segment id={segment_id}"
    ev = _get()
    if run is not None and run["system"] and run["response"]:
        system = [{"type": "text", "text": run["system"], "cache_control": {"type": "ephemeral"}}]
        content: list[dict] = []
        for m in _json.loads(run["shots"] or "[]"):
            b = _b64(m["path"])
            if b:
                content.append({"type": "text", "text": m["label"]})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        content.append({"type": "text", "text": run["prompt"]})
        api_msgs: list[dict] = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": run["response"],
                                               "cache_control": {"type": "ephemeral"}}]}]
        preface = (f"[The person you monitor is now reviewing your judgments above, in particular "
                   f"{who}. Answer their questions about what you saw and why you judged as you did; "
                   f"if you now think you were wrong, say so plainly and say what instruction or "
                   f"information would have led you to the right judgment.]\n\n")
    else:
        # No transcript: an informed stand-in built from what actually decided this segment.
        preds = conn.execute("SELECT model, p_content, p_switch, p_kind FROM seg_predictions WHERE segment_id=?",
                             (segment_id,)).fetchall()
        ptxt = "\n".join(f"  {p['model']}: content={p['p_content']} switch={p['p_switch']}" for p in preds) or "  (none stored)"
        vtxt = (f"resolved: content={ev_row['content']}, switch={ev_row['switch_label']}, "
                f"kind={ev_row['interruption_kind']}, rationale: {ev_row['rationale']}") if ev_row else "(no evaluation recorded)"
        system = [{"type": "text", "text": ev._system_blocks(conn)[0]["text"], "cache_control": {"type": "ephemeral"}}]
        api_msgs = []
        preface = (f"[No verbatim transcript exists for the evaluation of {who} - it was judged "
                   f"before transcripts were kept, or deferred to the local (non-LLM) models. "
                   f"What is on record:\n{vtxt}\nPer-model probability outputs:\n{ptxt}\n"
                   f"Help the person triangulate whether the judgment was right.]\n\n")
    for i, m in enumerate(messages):
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = (m.get("text") or "").strip()
        if role == "user" and preface:
            text = preface + text
            preface = ""
        api_msgs.append({"role": role, "content": text})
    resp = ev.client.messages.create(model=(run["model"] if run is not None and run["model"] else cfg.claude_model),
                                     max_tokens=1024, system=system, messages=api_msgs,
                                     output_config={"effort": "low"})
    if resp.stop_reason == "refusal":
        return "(Claude declined to answer that.)"
    return "".join(b.text for b in resp.content if b.type == "text").strip() or "(no reply)"


class AuditFlag(BaseModel):
    segment_id: int
    field: str = Field(description="Which judgment looks wrong: content, switch, or kind.")
    suggested: str = Field(description="The label you would give instead.")
    reason: str = Field(description="One short sentence.")


class DayAudit(BaseModel):
    flags: list[AuditFlag] = Field(description="ONLY segments whose judgment looks wrong. Empty list if the day looks right.")


AUDIT_PROMPT = """Below is one full day of this person's activity, with the resolved judgments \
(state, [content=...], switch labels) each segment received - some judged by cheap local models. \
Re-check the day as a whole. List ONLY the segments whose content or switch judgment looks wrong, \
with what you'd say instead. These get flagged for the person to review, so precision matters more \
than recall - do not nitpick defensible calls. An empty list is the expected answer for a clean day.

{log}"""


def daily_audit(conn, cfg: Config | None = None) -> int:
    """Once per day: one LLM sweep over yesterday's resolved judgments; suspected errors are
    flagged (uncertain=1) for the user's review queue - labels are never silently changed."""
    from . import stats, summaries
    from .config import logical_date
    cfg = cfg or load_config()
    if not cfg.daily_audit:
        return 0
    day = logical_date() - dt.timedelta(days=1)
    key = day.isoformat()
    if conn.execute("SELECT 1 FROM audit_runs WHERE day=?", (key,)).fetchone():
        return 0
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    t0, t1 = stats.day_bounds(day)
    segs = [s for s in stats.labelled_segments(conn, t0, t1)
            if not s["idle"] and not s.get("neutral") and s["duration"] > 0]
    evaluated = [s for s in segs if s.get("content")]
    if not evaluated:
        conn.execute("INSERT INTO audit_runs(day, n_flags, cost_usd, created) VALUES (?,?,?,?)",
                     (key, 0, 0.0, time.time()))
        return 0
    lines = [f"id={s['id']} [content={s.get('content') or '?'}] " + summaries.seg_line(s) for s in segs]
    ev = _get()
    resp = ev.client.beta.messages.parse(
        model=cfg.claude_model, max_tokens=4000,
        system=ev._system_blocks(conn),
        messages=[{"role": "user", "content": AUDIT_PROMPT.format(log="\n".join(lines))}],
        output_format=DayAudit,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default")
    cost = ((resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
            + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
            + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)
    n = 0
    ids = {s["id"] for s in evaluated}
    with db.tx(conn):
        if resp.parsed_output is not None:
            for f in resp.parsed_output.flags:
                if f.segment_id not in ids:
                    continue
                if conn.execute("SELECT 1 FROM seg_reviews WHERE segment_id=?", (f.segment_id,)).fetchone():
                    continue  # the user already ruled on this one
                conn.execute("UPDATE seg_evals SET uncertain=1, rationale = COALESCE(rationale,'') || ? WHERE segment_id=?",
                             (f" | daily audit: {f.field} looks like '{f.suggested}' - {f.reason}", f.segment_id))
                n += 1
        conn.execute("INSERT INTO audit_runs(day, n_flags, cost_usd, created) VALUES (?,?,?,?)",
                     (key, n, cost, time.time()))
    log.info("daily audit %s: %d flag(s) ($%.3f)", key, n, cost)
    return n


def run_pending(conn, cfg: Config | None = None, max_chunks: int = 1, lookback_s: float = 36 * 3600) -> int:
    """Evaluate the oldest unevaluated stretch that is at least LAZY_S old. Called from the
    classifier loop; also usable for backfill with a longer lookback and more chunks."""
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    horizon = time.time() - LAZY_S
    done = 0
    for _ in range(max_chunks):
        row = conn.execute("""
            SELECT MIN(g.start) FROM segments g
            LEFT JOIN seg_evals e ON e.segment_id = g.id
            WHERE g.start >= ? AND g.start < ? AND g.idle = 0 AND g.neutral = 0
              AND COALESCE(g.end, g.start) > g.start
              AND COALESCE(g.end, g.start) <= ? AND e.segment_id IS NULL""",
            (time.time() - lookback_s, horizon, horizon)).fetchone()
        if not row or row[0] is None:
            break
        t0 = row[0] - 1
        t1 = min(t0 + CHUNK_S, horizon)
        try:
            if _get().evaluate_chunk(conn, t0, t1) == 0:
                break  # nothing judged: don't spin on the same window
        except Exception:
            log.exception("chunk evaluation failed (%s)", dt.datetime.fromtimestamp(t0))
            break
        done += 1
    return done

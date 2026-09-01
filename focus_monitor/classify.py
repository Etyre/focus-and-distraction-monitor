"""Classification loop: runs pending switches through the ensemble with an adaptive Claude budget."""
from __future__ import annotations

import logging
import random
import time

from . import LABELS, db, ensemble, toggl
from .classifiers import (ClaudeVisionClassifier, HeuristicClassifier, LearnedClassifier,
                          build_context)
from .config import Config, load_config

log = logging.getLogger("classify")


class ClassifierService:
    def __init__(self, cfg: Config | None = None, use_claude: bool = True):
        self.cfg = cfg or load_config()
        self.conn = db.connect()
        self.local = [HeuristicClassifier(self.cfg), LearnedClassifier()]
        self.claude = None
        if use_claude:
            try:
                self.claude = ClaudeVisionClassifier(self.cfg)
            except Exception as e:  # e.g. no credentials
                log.warning("Claude classifier disabled: %s", e)
        self._review_count = -1
        self.toggl = toggl.TogglSync(self.conn)

    # -- learning ------------------------------------------------------------------------------
    def refresh_models(self):
        n = self.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if n != self._review_count:
            for m in self.local + ([self.claude] if self.claude else []):
                m.learn(self.conn)
            self._review_count = n
            log.info("models refreshed from %d reviews", n)

    # -- budget / frugality policy ---------------------------------------------------------------
    def should_call_claude(self, local_probs: dict | None, scores: dict, leaving_focus: bool) -> tuple[bool, str]:
        """Claude budget goes first to switches that leave an ongoing focus span."""
        if not self.claude:
            return False, "claude unavailable"
        spent = db.spend_today(self.conn)
        if spent >= self.cfg.daily_budget_usd:
            return False, f"daily budget reached (${spent:.2f})"
        # The chunk evaluator is now the primary judgment; per-switch Opus calls only get a
        # slice of the budget (they mainly sharpen review-queue ranking).
        if spent >= self.cfg.switch_claude_budget_fraction * self.cfg.daily_budget_usd:
            return False, "budget reserved for the chunk evaluator"
        local_conf = max(local_probs.values()) if local_probs else 0.0
        if not leaving_focus:
            # Already distracted/interrupted: cheap to get wrong. Local models suffice if confident,
            # and Claude is only used while there is plenty of budget left for the important cases.
            if spent >= self.cfg.low_priority_budget_fraction * self.cfg.daily_budget_usd:
                return False, "low priority: budget reserved for focus-span switches"
            if local_conf >= 0.8 and random.random() >= self.cfg.audit_sample_rate / 2:
                return False, "low priority: local models confident"
            return True, "low priority"
        learned = scores.get("learned")
        confident_local = local_conf >= self.cfg.frugal_confidence_threshold
        if (learned and learned["n"] >= self.cfg.frugal_min_reviews
                and learned["accuracy"] >= self.cfg.frugal_accuracy_threshold and confident_local):
            if random.random() < self.cfg.audit_sample_rate:
                return True, "audit sample"
            return False, f"frugal: local models accurate ({learned['accuracy']:.0%}) and confident"
        return True, "accuracy mode"

    # -- trivial switches: never worth Claude's or the user's time -------------------------------
    def trivial_reason(self, ctx) -> str | None:
        fr, to = ctx.from_seg, ctx.to_seg
        if not fr:
            return None
        if fr["app"] == to["app"] and fr["domain"] == to["domain"]:
            return "same app and site"
        window = ctx.ts - self.cfg.working_pair_window_min * 60
        n = self.conn.execute("""
            SELECT COUNT(*) FROM switches s JOIN segments f ON f.id=s.from_segment JOIN segments t ON t.id=s.to_segment
            WHERE s.ts >= ? AND s.ts < ? AND ((f.app=? AND f.domain=? AND t.app=? AND t.domain=?)
                                            OR (f.app=? AND f.domain=? AND t.app=? AND t.domain=?))""",
            (window, ctx.ts, fr["app"], fr["domain"], to["app"], to["domain"],
             to["app"], to["domain"], fr["app"], fr["domain"])).fetchone()[0]
        if n >= 3:
            return f"working pair (bounced {n}x in the last {self.cfg.working_pair_window_min:.0f} min)"
        return None

    def flags_today(self) -> int:
        from .config import day_bounds, logical_date
        t0 = day_bounds(logical_date())[0]
        return self.conn.execute("SELECT COUNT(*) FROM ensemble e JOIN switches s ON s.id=e.switch_id "
                                 "WHERE e.uncertain=1 AND s.ts>=?", (t0,)).fetchone()[0]

    # -- one switch --------------------------------------------------------------------------------
    def classify_switch(self, switch_id: int, force_claude: bool = False) -> dict | None:
        ctx = build_context(self.conn, switch_id)
        if not ctx:
            return None
        preds, rationales, costs = {}, {}, {}
        trivial = None if force_claude else self.trivial_reason(ctx)
        for m in self.local:
            r = m.predict(ctx)
            if r:
                preds[m.name], rationales[m.name], costs[m.name] = r
        scores = ensemble.model_scores(self.conn)
        weights = ensemble.weights_from_scores(scores, ["claude"] + [m.name for m in self.local])
        local_probs = ensemble.combine(preds, weights)[0] if preds else None
        # Only leaving a *real* focus span (>= min_focus_span_min) is high stakes; a flip away from a
        # 40-second stint is not an interruption of anything.
        leaving_focus = ctx.from_state == "focus" and ctx.focus_run_min >= self.cfg.min_focus_span_min
        importance = (1.0 + min(ctx.focus_run_min / 30.0, 2.0)) if leaving_focus else 0.3
        if trivial:
            # Working pairs and same-site hops are continuation by construction.
            preds["trivial"] = {"continuation": 0.94, "interruption": 0.02, "distraction": 0.02, "task_change": 0.02}
            rationales["trivial"], costs["trivial"] = trivial, 0.0
            call, why = False, f"trivial: {trivial}"
        else:
            call, why = (True, "forced") if force_claude else self.should_call_claude(local_probs, scores, leaving_focus)
        activity = None
        if call:
            r = self.claude.predict(ctx)
            if r:
                preds["claude"], rationales["claude"], costs["claude"] = r
                activity = getattr(self.claude, "last_activity", None)
            else:
                why += " (claude failed)"
        if not preds:
            return None
        if trivial:
            weights = dict(weights, trivial=3.0)
        probs, used = ensemble.combine(preds, weights)
        to = ctx.to_seg
        if self.cfg.is_daily_notes(to["app"], to["title"], to["url"]):
            # Logging a spare thought to daily notes is capture, part of focus - never a break.
            moved = probs["interruption"] + probs["distraction"]
            probs["continuation"] += moved
            probs["interruption"] = probs["distraction"] = 0.0
            rationales["daily_notes"] = "daily-notes capture (never an interruption/distraction)"
        label = max(probs, key=probs.get)
        # Mark ambiguity on every focus-leaving switch; the review SET is chosen later as the
        # top-N most uncertain/highest-stakes per day (stats.select_for_review), not first-come.
        uncertain = (leaving_focus and not trivial
                     and ensemble.is_uncertain(probs, preds, self.cfg.uncertain_threshold_focus))
        now = time.time()
        with db.tx(self.conn):
            for m, p in preds.items():
                self.conn.execute(
                    """INSERT INTO predictions(switch_id, model, p_continuation, p_interruption, p_distraction,
                       p_task_change, rationale, cost_usd, created) VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(switch_id, model) DO UPDATE SET p_continuation=excluded.p_continuation,
                       p_interruption=excluded.p_interruption, p_distraction=excluded.p_distraction,
                       p_task_change=excluded.p_task_change, rationale=excluded.rationale,
                       cost_usd=excluded.cost_usd, created=excluded.created""",
                    (switch_id, m, p["continuation"], p["interruption"], p["distraction"], p["task_change"],
                     rationales[m], costs[m], now))
            self.conn.execute(
                """INSERT OR REPLACE INTO ensemble(switch_id, p_continuation, p_interruption, p_distraction,
                   p_task_change, label, uncertain, weights, importance, focus_run_min, activity, created)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (switch_id, probs["continuation"], probs["interruption"], probs["distraction"],
                 probs["task_change"], label, int(uncertain), ensemble.dumps_weights(used),
                 importance, ctx.focus_run_min, activity, now))
            self.conn.execute("UPDATE switches SET status='classified' WHERE id=? AND status='pending'", (switch_id,))
        log.info("switch %d -> %s (%.0f%%)%s [%s; %s; $%.3f]", switch_id, label, 100 * probs[label],
                 " UNCERTAIN" if uncertain else "",
                 f"leaving {ctx.focus_run_min:.0f}-min focus" if leaving_focus else f"from {ctx.from_state}",
                 why, costs.get("claude", 0.0))
        return {"label": label, "probs": probs, "uncertain": uncertain, "why": why}

    # -- transitions: a chain of quick hops is one switch ---------------------------------------------
    def group_transition(self, switch_id: int) -> int | None:
        """If `switch_id` starts a burst of quick hops, mark the intermediates 'transit' and return the
        terminal switch (the one into the window where the person settled). Returns None if the hop
        chain isn't complete yet (still moving), or switch_id itself if there is no burst."""
        sw = self.conn.execute("SELECT * FROM switches WHERE id=?", (switch_id,)).fetchone()
        if sw["group_id"]:
            return None if sw["status"] == "transit" else sw["group_id"]
        origin = db.segment(self.conn, sw["from_segment"])
        chain = [dict(sw)]
        cur = db.segment(self.conn, sw["to_segment"])
        while cur is not None:
            end = cur["end"] or time.time()
            if end - cur["start"] >= self.cfg.transit_seconds:
                break  # settled here
            nxt = self.conn.execute("SELECT * FROM switches WHERE from_segment=? ORDER BY ts LIMIT 1", (cur["id"],)).fetchone()
            if nxt is None:
                if cur["end"] is None:
                    return None  # still moving - wait
                break            # short stay then idle/neutral: settle here
            chain.append(dict(nxt))
            cur = db.segment(self.conn, nxt["to_segment"])
        if len(chain) == 1:
            return switch_id
        settled = cur
        # Hunted around and came back to the same window: judge the quick hops individually.
        if origin and settled and origin["app"] == settled["app"] and origin["domain"] == settled["domain"]:
            return switch_id
        terminal = chain[-1]["id"]
        with db.tx(self.conn):
            for c in chain[:-1]:
                self.conn.execute("UPDATE switches SET status='transit', group_id=? WHERE id=?", (terminal, c["id"]))
            self.conn.execute("UPDATE switches SET group_id=? WHERE id=?", (terminal, terminal))
        log.info("transition: switches %s grouped into %d (%d hops)", [c["id"] for c in chain], terminal, len(chain) - 1)
        return terminal

    # -- loop ------------------------------------------------------------------------------------
    def pending(self) -> list[int]:
        cutoff = time.time() - self.cfg.lookahead_seconds
        ids = [r[0] for r in self.conn.execute(
            "SELECT id FROM switches WHERE status='pending' AND ts < ? ORDER BY ts LIMIT 50", (cutoff,))]
        out, seen = [], set()
        for sid in ids:
            t = self.group_transition(sid)
            if t is not None and t not in seen:
                st = self.conn.execute("SELECT status FROM switches WHERE id=?", (t,)).fetchone()[0]
                if st == "pending":
                    out.append(t)
                    seen.add(t)
        return out

    def run_once(self) -> int:
        self.refresh_models()
        n = self.toggl.sync()
        if n:
            log.info("toggl: synced %d entries", n)
        ids = self.pending()
        for sid in ids:
            try:
                self.classify_switch(sid)
            except Exception:
                log.exception("failed to classify switch %d", sid)
        try:
            from . import evaluator
            evaluator.run_pending(self.conn, self.cfg)  # supersedes the per-switch thread resolver
            evaluator.daily_audit(self.conn, self.cfg)  # once a day; a fast no-op otherwise
            evaluator.refine_incoherent(self.conn, self.cfg)  # auto-correct model-vs-model findings
        except Exception:
            log.exception("chunk evaluation failed")
        try:
            from . import summaries
            summaries.ensure_summaries(self.conn, self.cfg)
        except Exception:
            log.exception("span summaries failed")
        try:
            from . import questions
            questions.generate_pending(self.conn, self.cfg)
        except Exception:
            log.exception("question generation failed")
        return len(ids)

    def run(self, interval: float = 15.0):
        log.info("classifier started (claude=%s)", bool(self.claude))
        while True:
            try:
                self.run_once()
            except Exception:
                log.exception("classification batch failed")
            time.sleep(interval)

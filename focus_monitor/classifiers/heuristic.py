"""Rule-based classifier with per-app/domain priors learned from reviews (naive Bayes-ish)."""
from __future__ import annotations

from collections import Counter, defaultdict

from .. import LABELS, db
from ..config import Config
from .base import Classifier, SwitchContext, normalize


class HeuristicClassifier(Classifier):
    name = "heuristic"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dest_counts: dict[str, Counter] = defaultdict(Counter)   # key -> Counter(label)
        self.pair_counts: dict[str, Counter] = defaultdict(Counter)

    # ----- learning -------------------------------------------------------------------------
    def learn(self, conn) -> None:
        self.dest_counts.clear()
        self.pair_counts.clear()
        for r in db.reviews_with_context(conn):
            self.dest_counts[self._key(r["to_app"], r["to_domain"])][r["label"]] += 1
            self.pair_counts[self._pair(r["from_app"], r["from_domain"], r["to_app"], r["to_domain"])][r["label"]] += 1

    @staticmethod
    def _key(app: str, domain: str) -> str:
        return f"{app}|{domain}"

    @classmethod
    def _pair(cls, fa, fd, ta, td) -> str:
        return cls._key(fa or "", fd or "") + "->" + cls._key(ta, td)

    # ----- rules ------------------------------------------------------------------------------
    def _rule_probs(self, ctx: SwitchContext) -> dict[str, float]:
        to, fr = ctx.to_seg, ctx.from_seg
        text = f"{to['title']} {to['url']}".lower()
        dom = to["domain"]
        p = {"continuation": 0.55, "interruption": 0.25, "distraction": 0.12, "task_change": 0.08}

        if self.cfg.is_daily_notes(to["app"], to["title"], to["url"]):
            return {"continuation": 0.90, "interruption": 0.0, "distraction": 0.0, "task_change": 0.10}
        if any(pat.lower() in text for pat in self.cfg.blocked_page_patterns):
            return {"continuation": 0.03, "interruption": 0.07, "distraction": 0.88, "task_change": 0.02}
        if dom and any(dom == d or dom.endswith("." + d) for d in self.cfg.distraction_domains):
            # weak prior: a specific post reached from the current thread is usually work
            p = {"continuation": 0.30, "interruption": 0.12, "distraction": 0.40, "task_change": 0.18}
        elif fr and to["app"] in self.cfg.notes_apps and fr["app"] not in self.cfg.notes_apps:
            p = {"continuation": 0.80, "interruption": 0.08, "distraction": 0.02, "task_change": 0.10}
        elif fr and fr["app"] in self.cfg.notes_apps and to["app"] in self.cfg.work_apps:
            p = {"continuation": 0.78, "interruption": 0.10, "distraction": 0.02, "task_change": 0.10}
        elif fr and fr["app"] == to["app"] and fr["domain"] == to["domain"]:
            p = {"continuation": 0.70, "interruption": 0.15, "distraction": 0.08, "task_change": 0.07}
        elif to["app"] in ("Mail", "Messages", "Slack", "Discord", "WhatsApp") or dom in ("mail.google.com",):
            p = {"continuation": 0.15, "interruption": 0.55, "distraction": 0.20, "task_change": 0.10}

        # Coming back from a non-focus state into work/notes apps = focus start, not continuation
        if ctx.from_state in ("interrupted", "distracted") and (to["app"] in self.cfg.work_apps or to["app"] in self.cfg.notes_apps):
            p = {"continuation": 0.15, "interruption": 0.05, "distraction": 0.05, "task_change": 0.75}
        # Toggl intent: starting an entry at the switch = declared focus start; a running work entry
        # makes the destination more likely to be the work (unless it's a known distraction site).
        if ctx.toggl_started:
            p = {"continuation": 0.15, "interruption": 0.05, "distraction": 0.05, "task_change": 0.75}
        elif ctx.toggl_before and p["distraction"] < 0.5:
            p["continuation"] += 0.15
        # Bounce-back: FROM segment was short and the segment after TO returns to the earlier place
        dur_to = (to["end"] or ctx.ts) - to["start"]
        if ctx.after and fr and ctx.after[0]["app"] == fr["app"] and ctx.after[0]["domain"] == fr["domain"] and dur_to < 90:
            p["interruption"] += 0.15
        # Long dwell + notes app afterwards => absorbed
        if dur_to > 15 * 60 and ctx.after and ctx.after[0]["app"] in self.cfg.notes_apps:
            p["continuation"] += 0.2
        return p

    # ----- predict ----------------------------------------------------------------------------
    def predict(self, ctx: SwitchContext):
        rules = normalize(self._rule_probs(ctx))
        to, fr = ctx.to_seg, ctx.from_seg
        # Blend with learned priors (Laplace smoothed), weight grows with evidence.
        learned = Counter()
        n = 0
        pc = self.pair_counts.get(self._pair(fr["app"] if fr else "", fr["domain"] if fr else "", to["app"], to["domain"]))
        dc = self.dest_counts.get(self._key(to["app"], to["domain"]))
        for c, w in ((pc, 2.0), (dc, 1.0)):
            if c:
                for k in LABELS:
                    learned[k] += w * c[k]
                n += w * sum(c.values())
        probs = dict(rules)
        rationale = "rules"
        if n > 0:
            alpha = min(0.8, n / (n + 6.0))
            lp = normalize({k: learned[k] + 0.5 for k in LABELS})
            probs = {k: (1 - alpha) * rules[k] + alpha * lp[k] for k in LABELS}
            rationale = f"rules blended with {n:.0f} reviewed examples for this app/domain"
        return normalize(probs), rationale, 0.0

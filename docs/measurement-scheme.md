# Focus Monitor — Measurement & Rating Scheme

> **`Rules and heuristics for Focus monitor.md` is the canonical spec and takes priority over this file.**
> This document is the implementation companion — it explains how the rules there are computed, and
> covers details that doc doesn't yet clarify.

*Living document. This is the whole scheme in one place so we can review it end to end.
Values in **bold** come from `config.toml`; the parenthetical is the current setting.*

Config right now: min span 15 min · detour-break 5 min · away 10 min · transit 20s · daily budget $15 · event counts off.

---

## 1. What we are measuring

**Phase 1 (now): focus spans.** How long you hold attention on one task/thread/project without it
breaking, and *when* attention is stable vs. broken. Interruption and self-distraction **counts are
computed but hidden** (`count_events` = false); phase 2 turns them on once spans
are trustworthy.

Ultimate targets (spec): longest span/day, total focus time, # interruptions, # self-distraction
events, total time self-distracted.

---

## 2. Raw capture (collector)

- Poll every **2s**: frontmost app + window title, and the active tab URL for
  Chrome/Chromium/Safari. Nothing is keylogged.
- A (app, title, URL) held ≥ **3s** becomes a **segment**; the change
  between segments is a **switch**. Screenshot at each switch, refreshed every
  **60s**, deleted after **14 days**.
- **Activity** = macOS `HIDIdleTime` (time since last key/mouse/trackpad event).
- **Away**: no input ≥ **10 min**, or screen locked → an *away* segment,
  **backdated to when inactivity began**. A wall-clock gap between polls (the computer slept) counts
  as inactivity too, so a span ends when you actually stopped — not when we noticed on wake.
- **Short break**: **1–10 min** pause is logged
  (hatched on the timeline) but does **not** break a span. Under
  1 min: ignored.
- **Neutral windows** (`neutral_patterns`, e.g. presence-check form): recorded, never a switch/event,
  transparent to spans. `ignore_apps` are skipped entirely.

---

## 3. Grouping switches

- **Transitions**: a chain of windows each held < **20s** (desktops → window →
  tab) = one transition, judged once as *origin → where you settled*; the middle stops are "transit"
  and inherit the verdict. Exception: hunt around and land back where you started → the quick hops are
  judged individually (a 3-second glance from deep focus can still be an interruption).
- **Trivial** (auto-continuation, never modelled): same app + same site; or a **working pair** — A↔B
  bounced ≥ 3× within **30 min** (editor↔terminal, reading↔notes).

---

## 4. Four labels for a switch

- **continuation** — nothing changed: reading↔notes, editor↔terminal, tab-to-tab on one thread, a
  related side-question; also *staying* in a non-focus state (feed→feed, Slack→Signal).
- **interruption** — leaving focused work for a quick unrelated detour (stray question, reactive
  message check, notification).
- **distraction** — leaving focused work to seek stimulation with no objective (feed-scrolling,
  aimless shopping, dating apps, blocked URL).
- **focus start** (internally `task_change`) — focused work begins/resumes: return after a detour,
  deliberately starting a different task, or settling into absorbed reading/notes. Moving from one
  reading thread to another (a **focus transition**) is a focus start, **not** a distraction.

Calibrations encoded so far:
- Distraction domains are a **weak prior only**. A specific post/comment reached from the current
  thread is continuation; only feed-scrolling / aimless browsing is distraction.
- **Unfocused** is the term for off-task time that is not a brief in-span detour - a longer stretch of
  random browsing / not being on a task (shown as "unfocused", not "focus broken").
- **Daily-notes capture**: switching to your Roam/Logseq **daily notes** to log a spare thought is
  never an interruption or distraction - it is part of focus (a hard rule, applied after the models).
- **Toggl** entries = declared intention: the running entry at each switch, and whether you *started*
  one at that moment (→ focus start), go to every model.

---

## 5. Labels → spans (nested model)

A **span is a container**, and **one threshold (5 min) governs every diversion**:
- A run of **focus** segments on one task builds a span.
- A **diversion** = any contiguous time **away from the span's task** — a distraction, an interruption,
  **or a hop to different work** (all merged into one; durations add up).
- A diversion **< 5 min** stays *inside* the span as one flagged attention-interruption event (▽ on the
  timeline); the span continues, and returning to the task continues the same focus block.
- A diversion **≥ 5 min** ends the span, retroactively at the moment you left the task; if the diverting
  work is itself ≥ the span minimum it becomes its own span. **Going idle (≥ 10 min)** or a recording
  gap also ends the span.
- There is **no instant "task change" break**: a deliberate switch to different work is just another
  diversion subject to the 5-minute rule. "Same task" = returning to the same window/site, a
  continuation-type focus, or a task-change the model isn't confident about (`P(task_change) < 0.7`);
  a human review of a switch as a task change is authoritative.
- Returning to the **same app/site** after a brief detour is a **resume**, not a new task. A
  **continuation** back onto a work/notes surface (a `focus_domains` site like roamresearch.com, a
  work/notes app, or the exact place you were focused) also **resumes focus** - a brief glance never
  "leaks" its interrupted/distracted state forward onto the work you return to.
- **Same Toggl task = same span (evidence, not override).** A running Toggl entry states *intent*, not
  that you were focused. It suppresses a "focus start (new task)" split only when (a) the model is NOT
  confident it is a real task change (`P(task_change) < 0.7`), (b) the switch was not human-reviewed as
  a task change (a human review always wins), and (c) it is a focus run - Toggl never relabels
  interrupted/distracted time. A real break (detour >= 5 min, away, gap) still ends the span first, so
  Toggl cannot glue two spans across a genuine distraction even under one running entry.
- A span counts only with **≥ 15 min of focus**; shorter focused stints show
  faintly and are not spans.

Each span records **what ended it**. "Length" = wall-clock; "focused" excludes contained detours.
Daily: longest span, total focus minutes (excl. contained detours); hidden in phase 1 —
interruption/distraction counts, self-distracted minutes.

---

## 6. Where judgement is spent

Switches that **leave an ongoing focus span** matter most: prioritised for Claude, weighted more in
scoring, and the only ones ever shown to you.
- **Claude budget**: all switches at first; once the learned model reaches ≥
  85% over ≥ 40 reviews and the local models are
  confident, only a 10% audit sample goes to Claude. Non-focus-leaving switches
  use Claude only under 60% of budget. Hard cap
  **$15/day**.
- **Review queue**: ≤ **8/day**, highest-stakes first. Your review overrides the
  model and retrains it.

---

## 7. Models (ensemble)

- **heuristic** — rules + per-app/domain priors from your reviews.
- **learned** — logistic regression on switch features + Toggl signals, once ≥ 15 reviews.
- **claude** — Opus 5 vision: before/after screenshots + narrative + Toggl intent + your recent
  reviews as examples.

Weight ∝ `exp(-2 × log-loss)` on your last 200 reviews, importance-weighted toward focus-leaving
switches. Flagged for review when top probability < **75%**
(focus-leaving) or Claude disagrees with the ensemble.

---

## 8. Open questions / to revisit

- Toggl weighting strength (currently a hint; starting an entry = focus start).
- Auto-merging adjacent spans that share a Toggl task but were split by a "focus start".
- Phase 2: turn on interruption / self-distraction **counts**.
- Shrinking the distraction-domain list further as the learned model improves.

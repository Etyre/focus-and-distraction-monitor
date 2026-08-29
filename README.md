# Focus & Distraction Monitor

Watches the active window on your Mac, screenshots on every window switch, and uses an
ensemble of models (Claude Opus 5 vision + a rules model + a logistic-regression model that
learns from your reviews) to judge whether each switch was a **continuation of focus**, an
**attention interruption**, a **self-distraction**, or a deliberate **task change**. A local
dashboard shows full-absorption spans on a timeline, daily metrics, a review queue for
uncertain cases, and model accuracy over time.

## Setup

```bash
cd focus-and-distraction-monitor
# .venv already created with Python 3.12; otherwise:
#   /opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # put this in ~/.zshrc
.venv/bin/python -m focus_monitor check  # verifies permissions + API access
```

macOS will prompt for permissions the first time; grant your terminal app:
* **Screen Recording** (screenshots and window titles)
* **Accessibility** and **Automation → System Events / Google Chrome** (AppleScript)

Then edit `config.toml` — especially `about_me`, `distraction_domains`, and
`blocked_page_patterns` (the text your blocker's block page shows).

## Run as a standalone app (recommended)

```bash
.venv/bin/python build_app.py                  # -> dist/Focus Monitor.app
open "dist/Focus Monitor.app"                  # or drag to /Applications; add to Login Items
```

Because a Finder-launched app doesn't see your shell environment, store the API key where the
app can find it - in the Keychain (preferred):

```bash
security add-generic-password -s FocusMonitor -a anthropic -w   # prompts for the key
```

or in `~/Library/Application Support/FocusMonitor/env` as `ANTHROPIC_API_KEY=sk-ant-...`.
The bundle references this project directory (its venv and `config.toml`), so rebuild after moving
the project. Dev alternative without the bundle: `.venv/bin/python -m focus_monitor app`.

An eye appears in the menu bar while the monitor is running (👁 ⏸ when paused, 👁 ⚠︎ if the
collector stops responding). Click it for **Open Dashboard** (a native window - no browser),
**Review Uncertain Switches**, today's numbers, review-queue size, Claude spend, Pause/Resume and
Quit. The dashboard is also reachable in a browser at http://127.0.0.1:8790.

Headless alternative: `run` (same three services, no icon). Pieces can also run separately:
`collect`, `classify [--once] [--no-claude]`, `web`, `stats --days 14`.
Start at login: System Settings → General → Login Items → add `Focus Monitor.app`. (A launchd
plist is also in `launchd/` for the headless variant.)

## How it works

* **Collector** polls every 2 s. A new (app, title, URL) that persists ≥ 3 s becomes a new
  *segment* and a *switch*; a screenshot is taken then and refreshed every 60 s. Keyboard/mouse
  activity is read from macOS's `HIDIdleTime` (time since last input - nothing is keylogged). No input
  for ≥ 8 min, or a locked screen, becomes an *away* segment starting when the inactivity began (this
  ends a focus span); pauses of 1-8 min are logged as *short breaks* and shown hatched on the timeline
  but don't break spans.
* **Classifier** waits 2 min after a switch (so it can see what you did next), then asks the
  models. Each yields probabilities for the four labels; the ensemble weights them by
  `exp(-2 × log-loss)` on your last 200 reviews. Switches are flagged *uncertain* when the top
  probability < 0.65 or Claude disagrees with the ensemble.
* **Adaptive frugality**: Claude is called for every switch until the learned model reaches
  ≥ 85 % accuracy on ≥ 40 reviews *and* the local models are ≥ 85 % confident on the switch;
  then only a 10 % audit sample goes to Claude. A hard daily cap (`daily_budget_usd`, default $15)
  applies regardless.
* **Learning**: reviews retrain the logistic model and the per-app/domain priors immediately, and
  the 40 most recent reviews (with your notes) are shown to Claude as examples.
* **Labels** (on switches): `continuation` = the same state continues (reading↔notes, but also
  feed→feed while distracted or Slack→Signal while on a detour); `interruption`; `distraction`;
  `focus start` (internally `task_change`) = focused work begins or resumes here. A segment's state
  (focus / interrupted / distracted / away) follows from the label of the switch into it. Focus spans =
  maximal runs of focused segments; a distraction *event* = maximal run of distracted segments.
* **Phase 1 = focus spans.** `count_events = false` hides interruption/distraction counts everywhere;
  the labels are still produced because they decide where a span breaks. The Day view shows each
  span with what ended it (interrupted / distracted / away / new task) and the Toggl entries inside it.
* **Nested spans** (`break_min`, `min_focus_span_min`): a focus span is a container. Detours
  (interrupted/distracted) shorter than 5 min are events *inside* the span and don't end it; a detour of
  5+ min, being away, a recording gap, or a deliberate focus start ends it. A span must contain ≥ 15 min
  of focus. "Length" is wall-clock; "focused" excludes contained detours.
* **Toggl as intent**: set your Toggl Track API token from the menu (or Keychain account `toggl`).
  Entries sync every 2 min; the running entry at each switch - and whether you *started* an entry at
  the switch - is given to every model as your declared intention, and shown as a row on the timeline.
* **Transitions**: a chain of windows each held < 20 s (`transit_seconds`) - scrolling desktops,
  picking a window, picking a tab - is one transition, judged once as *origin → where you settled*;
  the stops in between are recorded as transit and inherit that verdict. If you hop around and land
  back in the window you left, the hops are judged individually (a 3-second glance can still be an
  interruption).
* **You are asked about very few switches.** Same-site hops and "working pairs" (A↔B bounced 3+ times
  in 30 min) are never questioned or sent to Claude. Only switches that *leave* an ongoing focus span
  can be flagged, at most `max_reviews_per_day` (8) per day, highest-stakes first (longer preceding
  focus = higher stakes). Everything else is auto-accepted. Model accuracy is likewise weighted toward
  focus-leaving switches.

**Neutral windows** (`neutral_patterns` in `config.toml`, e.g. a presence-check form) are recorded
but never produce a switch or event, and are transparent to focus spans - a 30-second check in the
middle of an hour of writing leaves a single one-hour span.

Labels live on switches; your review always overrides the model. See `docs/measurement-scheme.md` for the full measurement & rating scheme.

Data is in
`~/Library/Application Support/FocusMonitor/` (`focus.db` + `screenshots/`, pruned after 14 days).

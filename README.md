# Focus & Distraction Monitor

Watches the active window on your Mac, screenshots on every window switch, and uses an
ensemble of models (Claude Opus 5.5 vision + a rules model + a logistic-regression model that
learns from your reviews) to judge whether each switch was a **continuation of focus**, an
**attention interruption**, a **self-distraction**, or a deliberate **task change**. A local
dashboard shows full-absorption spans on a timeline, daily metrics, a review queue for
uncertain cases, and model accuracy over time. Built for Eli to measure his own focus spans
against his own rules (`Rules and heuristics for Focus monitor.md`); spec written 2026-08-25,
first commit 2026-08-29, in continuous development since.

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
  The synced table is also served read-only at `GET /api/toggl/current` and `GET /api/toggl/entries?days=N`
  (added 2026-11-11) so the executive assistant reads Toggl from here instead of polling Toggl itself; its
  calls were exhausting the hourly API quota the two apps share. "Current" lags Toggl by up to the 5-minute
  sync interval; each row carries `synced`.
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
* **Agreement with your judgments** (top of the Models tab, `/api/agreement`): of the verdicts you
  have given, how many the app already had right vs. how many you corrected, overall, over the last
  30 days, and by day. A segment verdict agrees when every field you set matches what the app had
  judged *before* you ruled (content, switch relation, thread); when the app re-judged a segment after
  your ruling, the judgment you saw is recovered from the evaluation transcript (`eval_runs`). Span
  verdicts count a confirmed span as agreement and a rejected one as a correction.
* **Neutral windows** (`neutral_patterns` in `config.toml`, e.g. a presence-check form) are recorded
  but never produce a switch or event, and are transparent to focus spans - a 30-second check in the
  middle of an hour of writing leaves a single one-hour span.
* Labels live on switches; your review always overrides the model. See `docs/measurement-scheme.md`
  for the full measurement & rating scheme.

* **Phone lane.** Under every timeline (Day, span, question and capture views) a second lane shows
  which app was on the iPhone's screen, with a tick per unlock, read from the
  [phone-unlock-logger](../phone-unlock-logger)'s database (`phone.py`, read-only; path in
  `phone_db`). The phone pushes its log to the Mac on its own schedule, so the lane usually runs
  hours behind the Mac lane; when it ends before the window does, the lane says so ("phone data
  ends 12:12"). The stream only says which app was on screen, not whether the phone was being
  touched; the proxy is iOS Auto-Lock. A segment the phone ended by locking itself, i.e. one that
  ends in a lock at almost exactly the Auto-Lock interval, means the app was left open on an idle
  phone and barely used: it is flagged `put_down` and drawn hollow (dashed) in the lane. The
  interval is inferred from the data (`phone.autolock_s`: lock-ended durations spike in a
  15-second band above one of iOS's Auto-Lock options), so changing the setting needs no code
  change; it takes a couple of days of self-locks to show up. Display only for now: the evaluator
  does not see the phone yet.

### The app, the launcher and restarts

An eye appears in the menu bar while the monitor is running (👁 ⏸ when paused, 👁 ⚠︎ if the
collector stops responding). Click it for **Open Dashboard** (a native window - no browser),
**Review Uncertain Switches**, today's numbers, review-queue size, Claude spend, Pause/Resume and
Quit. The dashboard is also reachable in a browser at http://127.0.0.1:8790.

The bundle's launcher (`dist/Focus Monitor.app`, built by `build_app.py`) supervises the Python
process: if it crashes, or stops ticking for 3 minutes while the Mac is awake (macOS froze it solid
once, for three days), it is killed and restarted, and the event is written to
`~/Library/Application Support/FocusMonitor/launcher.log`. The app also opts out of App Nap. Quitting
from the menu is a clean exit and is not restarted. Because a Finder-launched app doesn't see your
shell environment, the API key is read from the Keychain (see Common tasks) or from
`~/Library/Application Support/FocusMonitor/env`. The bundle references this project directory (its
venv and `config.toml`), so rebuild after moving the project.

After editing source, `.venv/bin/python -m focus_monitor restart` makes the running app relaunch
itself (the dashboard HTML is read once at startup); a dashboard window that was open comes back on
the same tab. `--if-stale` makes it a no-op unless a source file is newer than the process. The Claude
Code Stop hook in `.claude/settings.json` used to run exactly that after every turn; it is disabled
(since 2026-10-06) because a restart wipes in-app state mid-review. Code that does not compile is
refused rather than restarted into.

Headless alternative: `run` (same three services, no icon). Pieces can also run separately:
`collect`, `classify [--once] [--no-claude]`, `web`, `stats --days 14`.

### Model calls: two backends and the LLM inbox

`focus_monitor/llm.py` is the one door for every structured-output model call (vision classifier,
segment evaluator, daily audit, span summaries, review questions, feedback propagation, thread
resolver, effort judge). `llm_backend` in `config.toml` picks who pays: `"api"` = the API key per
token; `"subscription"` = the claude.ai login through the Claude Code binary (Agent SDK), falling
back to the key while a usage limit is in force. The three interactive chats (switch, span, day) and
the evaluator chat use the API directly because they use tools.

Every system prompt starts with the shared preamble from `~/Documents/agent-coding/shared/`
(`llm_contact.system_prompt`), every call that reaches the API carries the `contact_eli` tool, and
every response goes through `llm_contact.check`. If the model asks to talk to Eli (the tool, or an
`@ELI:` line - the only route on the subscription backend, which runs without tools), the contact is
recorded in the LLM inbox, Eli is notified, and that task stops: a structured call returns
`parsed_output=None` with `stop_reason="contact_eli"` (callers treat it as "no answer" and do not
retry), a chat shows a notice with the inbox id instead of a reply. Read and answer with `llm-inbox`.

## Files

| File | Purpose |
|---|---|
| `focus_monitor/__main__.py` | Entry point: `collect`, `classify`, `web`, `app`, `run`, `stats`, `check`, `restart`. Sets up logging. |
| `focus_monitor/config.py` | `Config` dataclass, `config.toml` loader, paths (data dir, log, Keychain lookup), pricing constants. |
| `focus_monitor/phone.py` | Read-only view of the phone-unlock-logger's database: per-app phone segments and unlocks for a window, app-name map, and how far the phone data reaches. |
| `focus_monitor/collector.py` | Active-window polling, screenshots, idle detection, segments and switches. |
| `focus_monitor/classify.py` | Classifier service loop: waits the lookahead, runs the ensemble, flags uncertain switches. |
| `focus_monitor/classifiers/` | `claude_vision.py` (Claude + screenshots), `heuristic.py` (rules), `learned.py` (logistic model), `base.py` (switch context). |
| `focus_monitor/ensemble.py` | Weights the classifiers by recent log-loss. |
| `focus_monitor/evaluator.py` | Segment evaluator (content / switch / kind per segment), daily audit, evaluator chat. |
| `focus_monitor/llm.py` | The one door for structured model calls: api vs. subscription backend, limit handling, cost log, shared preamble + `contact_eli` + inbox check. |
| `focus_monitor/usage_log.py` | Appends every call to `~/.claude/app-usage/Focus-monitor.jsonl` for the Claude usage dashboard. |
| `focus_monitor/chat.py` | Interactive chats about one switch, one span, or a whole day (the day chat edits records through tools). |
| `focus_monitor/questions.py` | Writes review questions for doubted segments; ranks the queue. Each question is about one dimension (thread, content, or hybrid); a doubt uncertain along two gets two questions on the same segment, and options that would land the same judgment (blank fields count as "as the model has it") are merged. Every question's options are the model's readings plus the app's own pass-through option ("just a window I passed through"), which is left out when the segment was plainly in use: held longer than `transit_seconds` and given under 10% chance of being a transition by the evaluator. |
| `focus_monitor/propagation.py` | Applies a correction to similar recent segments (feedback propagation judge). |
| `focus_monitor/resolver.py` | Attention-thread resolver: which thread a switch continues. |
| `focus_monitor/summaries.py` | Span summaries and coherence checks. |
| `focus_monitor/efforts.py` | Effort-ladder shadow calls and the effort judge (which effort level is good enough). |
| `focus_monitor/edits.py`, `seg_models.py`, `stats.py`, `toggl.py`, `db.py` | Edits with provenance, label space, metrics/spans, Toggl sync, SQLite schema and helpers. |
| `focus_monitor/web.py` | FastAPI server for the dashboard and its API (port 8790). |
| `focus_monitor/dashboard.html` | The whole dashboard (Day, Captures, Trends, Review, Models, Efforts). Drafts in every text field persist in `localStorage`. |
| `focus_monitor/menubar.py`, `icons.py` | The menu bar eye (rumps) and its icons. |
| `build_app.py`, `build.sh` | Build `dist/Focus Monitor.app` (signed native launcher wrapping the venv). |
| `install.sh` | Idempotent install: build, register the Login Item, relaunch; `--launchd` for the headless variant. |
| `test.sh` | End-to-end self-test against a throwaway data dir; never touches the running app. |
| `launchd/com.elityre.focus-and-distraction-monitor.plist` | launchd agent for the headless variant (installed by `install.sh --launchd`). |
| `config.toml` / `config.example.toml` | Your settings and profile (untracked) / the committed template. |
| `Rules and heuristics for Focus monitor.md` | The canonical rules; read at runtime and shown to every model as authoritative. |
| `docs/measurement-scheme.md` | How the rules are computed; implementation companion to the rules doc. |
| `AI focus and distraction monitor spec.md` | The original spec (2026-08-25). |
| `.claude/settings.json` | The (disabled) Claude Code Stop hook that restarted the app after edits. |
| `requirements.txt` | anthropic, claude-agent-sdk, fastapi, uvicorn, pillow, scikit-learn, pydantic, rumps, pyobjc (WebKit, Quartz). |

## Runtime locations

* Config: `./config.toml` in this folder (`FOCUS_MONITOR_CONFIG` overrides). Re-read on start;
  `llm_backend` is re-read every minute.
* Secrets: Keychain service `FocusMonitor`, accounts `anthropic` (API key) and `toggl` (Toggl token);
  fallback `~/Library/Application Support/FocusMonitor/env` (`ANTHROPIC_API_KEY=sk-ant-...`). A shell
  `ANTHROPIC_API_KEY` wins when present.
* Data: `~/Library/Application Support/FocusMonitor/` (`FOCUS_MONITOR_DATA` overrides): `focus.db`,
  `screenshots/` (full size; pruned after `screenshot_retention_days`, or kept for good when that is
  0, the setting since 2026-11-11), `screenshots/archive/` (a 640px copy of every capture for good -
  except captures that show a `private_apps` window such as Signal, marked `_priv` in the filename,
  which are deleted from both tiers after `private_retention_days` (14) so disappearing messages
  never outlive the chat), `heartbeat`, `icons/`, `reopen_dashboard.json`.
* Logs: `~/Library/Logs/focus-and-distraction-monitor.log` (app log, timestamped, rotated at 5 MB x 3;
  `FOCUS_MONITOR_LOG` overrides; before 2026-11-07 it was `Application Support/FocusMonitor/app.log`),
  `~/Library/Application Support/FocusMonitor/launcher.log` (crash/freeze restarts by the native launcher),
  `~/Library/Logs/focus-and-distraction-monitor.launchd.log` (stdout/stderr of the headless variant only).
* Usage log for the Claude usage dashboard: `~/.claude/app-usage/Focus-monitor.jsonl`.
* LLM inbox (contacts from the model): `~/Library/Application Support/LLMInbox/messages.jsonl`, via `llm-inbox`.
* Installed copy: `dist/Focus Monitor.app` in this folder, registered as a Login Item (System
  Settings → General → Login Items). The bundle points at this folder's `.venv` and `config.toml`.

## Common tasks

```bash
cd ~/Documents/agent-coding/focus-and-distraction-monitor
# .venv already created with Python 3.12; otherwise:
#   /opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# API key where a Finder-launched app can find it (preferred: Keychain; prompts for the key)
security add-generic-password -s FocusMonitor -a anthropic -w
# or: echo 'ANTHROPIC_API_KEY=sk-ant-...' > ~/Library/Application\ Support/FocusMonitor/env
export ANTHROPIC_API_KEY=sk-ant-...      # for dev runs from a shell (put this in ~/.zshrc)

.venv/bin/python -m focus_monitor check  # permissions + model access
./test.sh                                # end-to-end self-test in a temp data dir (no model calls, no port 8790)

./build.sh                               # -> dist/Focus Monitor.app   (same as .venv/bin/python build_app.py)
./install.sh                             # build + Login Item + relaunch if running; safe to rerun
open "dist/Focus Monitor.app"            # or drag to /Applications; add to Login Items
.venv/bin/python -m focus_monitor app    # dev alternative without the bundle (menu bar eye)
.venv/bin/python -m focus_monitor run    # headless: collector + classifier + web, no icon

.venv/bin/python -m focus_monitor restart            # running app relaunches itself with the new code
.venv/bin/python -m focus_monitor restart --if-stale # only if a source file is newer than the process

tail -f ~/Library/Logs/focus-and-distraction-monitor.log
tail -f ~/Library/Application\ Support/FocusMonitor/launcher.log
.venv/bin/python -m focus_monitor stats --days 14
.venv/bin/python -m focus_monitor classify --once --no-claude   # local models only

# Login Item by hand
osascript -e 'tell application "System Events" to make login item at end with properties {path:"'"$PWD"'/dist/Focus Monitor.app", hidden:false}'
osascript -e 'tell application "System Events" to delete login item "Focus Monitor"'

# Headless variant under launchd instead of the Login Item (mutually exclusive)
./install.sh --launchd
launchctl bootout gui/$(id -u)/com.elityre.focus-and-distraction-monitor   # stop it

# Quit the menu bar app: Quit from the eye menu (clean exit, the launcher does not restart it).
# Kill by PID if it is stuck: pgrep -f "Focus Monitor.app/Contents/MacOS" | xargs kill
```

macOS will prompt for permissions the first time; grant your terminal app (for dev runs) or
"Focus Monitor" (the bundle):
* **Screen Recording** (screenshots and window titles)
* **Accessibility** and **Automation → System Events / Google Chrome** (AppleScript)

Then edit `config.toml` - especially `about_me`, `distraction_domains`, and
`blocked_page_patterns` (the text your blocker's block page shows).

## Settings that matter

* **Phone lane**: needs the phone-unlock-logger installed and running (its launchd agent reads the
  Biome stream every 2 hours) and, on the iPhone, Screen Time → Share Across Devices on.
* **iPhone Auto-Lock** (Settings → Display & Brightness → Auto-Lock): 2 minutes, set 2026-11-11
  (it was 5 minutes before). Shorter is better for the phone lane: a phone left open on the desk
  shows as use for at most this long before it locks itself and the segment is flagged `put_down`.
  Attention Aware Features on means looking at the phone also counts as use, which is right.

* System Settings → Privacy & Security: **Screen Recording**, **Accessibility**, **Automation**
  (System Events, Google Chrome) for "Focus Monitor" (and for your terminal app when running from a shell).
* System Settings → General → **Login Items**: "Focus Monitor" (added by `install.sh`).
* `config.toml` `[classifier]`: `llm_backend` (`subscription` needs a Claude Code login, `api` needs the
  key), `claude_model`, `claude_effort`, `daily_budget_usd` / `total_budget_usd`, `defer_*`,
  `max_reviews_per_day`, `question_archive_days` (3; an open review question older than this is
  archived, 0 = never). `[profile]`: `about_me`, `distraction_domains`, `blocked_page_patterns`,
  `private_apps`, `neutral_patterns`, `ignore_apps`. `[day] day_start_hour` (a day runs 4am-4am).
* Toggl Track API token: from the eye menu, or Keychain `FocusMonitor` / `toggl`.
* LLM inbox notifications: `~/Library/Application Support/LLMInbox/config.env` (see `shared/README.md`).
* Nightblock closes `api.anthropic.com` midnight-6am; model calls fail then and are retried later.

## Gotchas learned

* Models moved on 2026-11-16: Opus 5 to Opus 5.5 (`claude_model`) and, for the thread resolver, Haiku 4.5
  to Haiku 5.5. Every Opus call already set its effort, so only the model name and the price constants in
  `config.py` changed (costs logged before that date used Opus 5 prices). Haiku 5.5 thinks by default and
  thinking shares `max_tokens`, so the resolver went from 300 to 2000 with `effort: low`. Checked with one
  real call per model through both backends. `config.toml` is untracked, so a fresh checkout needs its
  `claude_model` set too; the running app keeps the old models until `restart`.

* A Finder-launched app has no shell environment: the API key must be in the Keychain or the `env`
  file, not just `~/.zshrc`.
* The bundle hard-codes this project directory (venv + `config.toml`); moving the project means
  `./build.sh` again.
* The launcher runs Python from inside the signed bundle so macOS attributes Screen Recording /
  Accessibility / Automation to "Focus Monitor", not to Python; a bare script is denied silently.
* macOS once froze the process solid for three days without killing it; the launcher now restarts on a
  stalled heartbeat (3 min while awake), and the app opts out of App Nap.
* A blank dashboard window with a healthy server (`curl http://127.0.0.1:8790/` returns the page) is
  the web view, not the app. On 2026-11-14 WebKit killed its hung GPU process and the window's
  long-lived web content process never painted again (`FailedToAcquireBufferSpan` every 15 s in
  `/usr/bin/log show --predicate 'processID == <app pid> AND subsystem == "com.apple.WebKit"'`), so
  every open showed an empty window. Since then each open from closed builds a new web view and
  retires the old process: close the window and open it again. Note `log` alone is a zsh builtin
  and prints nothing useful; call `/usr/bin/log`.
* Review questions left open for `question_archive_days` (3) are archived, not deleted
  (`questions.archive_stale_questions`, every 30 min in the classifier loop; first run 2026-11-14
  archived 1,472 of 1,529). The row keeps everything and gets `status='archived'` + `archived_at`;
  the segment's needs-review flag is cleared (its rationale gets "review question archived
  unanswered" appended) and the model's judgment stands. Age counts from `created`, so reopening an
  answer on a question older than that sends it to the archive at the next pass. To see them:
  `sqlite3 focus.db "SELECT id, question FROM review_questions WHERE status='archived'"`.
* Review questions are shown in time order, oldest first (since 2026-11-16), so next/prev move to
  the neighbouring question on the same day. Before that the priority tier came first, and "next"
  jumped between days. A needs-review dot on the Day timeline opens its question inside that queue
  (the switch card only when the flag has no open question), so the arrows carry on from there.
* The review-question queue (`/api/questions/queue`) returns every open question, so anything
  computed per question there is multiplied by the backlog. With ~460 priority questions open
  (2026-11-14) the per-question timelines made it take 25 s and 10 MB, and the Review tab sat on an
  empty card that whole time. The queue now carries no timelines; each card fetches its own from
  `/api/questions/{id}/context` when shown (and warms the next one).
* A segment can be flagged for review with no switch into it (the first segment after a wake from
  sleep, e.g. 34062). Its "needs review" dot has only a question to open; treating the missing
  switch id as a switch drew an empty card.
* The dashboard HTML is read once at startup: after editing it, `restart`. A restart wipes in-app state
  mid-review, which is why the automatic Stop hook was disabled.
* `restart` refuses code that does not compile, so the launcher never crash-loops on a typo.
* The subscription backend (Claude Code binary) takes exactly one user message, no tools, and would
  prefer `ANTHROPIC_API_KEY` over the login if it saw it, so the child gets it blanked; session
  persistence is off so a thousand screenshot-laden calls a day do not pile up under `~/.claude/projects`.
  A usage-limit error switches every call to the API key until the reset time named in the error.
* Subscription calls are invisible to the Anthropic usage reports, so `usage_log.py` writes every call
  to `~/.claude/app-usage/Focus-monitor.jsonl` for the usage dashboard (with a backfill from `llm_calls`).
* `contact_eli` cannot be offered on the subscription backend (no tool plumbing through the Agent SDK
  one-shot call); the preamble's `@ELI:` marker is detected there instead, including inside the JSON of a
  structured answer.
* Structured calls that come back without parsed output (refusal, max_tokens, or a `contact_eli` call)
  are simply "no answer" at every call site; the next cycle may look at the same segment again, but the
  same request is never retried in place.
* Meetings never count as fully absorbed focus; they are their own tier/metric, not "total focus".
* Screenshots of `private_apps` (Signal) are deleted from both tiers after `private_retention_days`
  so disappearing messages never outlive the chat. Nothing else is ever deleted: since 2026-11-11
  the full-size captures are kept too (`screenshot_retention_days = 0`, about 5 GB a month), because
  the plan is to get the evaluator to predict Eli's judgments well and then re-run it over all the
  past data. After that rerun, set a retention again and let the full-size tier go; the 640px
  archive stays.
* Before 2026-11-07 the launchd label was `com.focusmonitor` and the plist wrote to `/tmp`; the file
  is now `launchd/com.elityre.focus-and-distraction-monitor.plist`. Nothing was installed under the old
  label (the app runs as a Login Item), so no `bootout` was needed.
* A doubt the question generator looks at and passes over must be written down, or it comes back.
  Until 2026-11-12 the only thing that marked a doubt as handled was a question row, so every
  doubt the model declined (or whose thread question the thread-mass gate dropped) was gathered
  and sent again on the next pass, about every 18 s, all night: 669 generator calls between
  midnight and 4 AM for one question. Now a `declined` row in `review_questions` records the
  pass-over (the queue never shows it; re-evaluating the segment deletes it) and the
  needs-review flag is cleared with it.
* The phone lane lags: the iPhone syncs its foreground log to the Mac on its own schedule (often
  hours, sometimes a day), and the logger only collects every 2 hours. An empty phone lane for
  this afternoon means "not synced yet", not "no phone use" - the lane prints where its data ends.
* Nothing in the phone data records touches, scrolling or the backlight; only which app was in
  the foreground. The Auto-Lock interval is the one tell: in 30 days of 5-minute data, 40 of
  1,349 lock-ended segments fell in 300-310 s (self-locks) against single digits in the
  neighbouring 15-second bands. Under 5 min = active until locked by hand; at the interval = put
  down; longer = ambiguous (either active to the end or idle for the final interval).

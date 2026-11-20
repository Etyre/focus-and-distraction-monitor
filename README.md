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
  for ≥ 10 min (`idle_seconds`) becomes an *away* segment starting when the inactivity began (this
  ends a focus span); pauses of 1-10 min are logged as *short breaks* and shown hatched on the timeline
  but don't break spans. A locked screen is time without input like any other: it becomes away
  only when the total reaches 10 minutes (until 2026-11-20 a lock was away at once, and on battery
  the display sleeps 2 minutes after the last input). Days run 4am to 4am in the time zone the
  Mac was in at the time (`timezones.json` in the data folder, kept by the collector).
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
  The chunk evaluator sees your segment rulings as precedents two ways (`evaluator._examples`,
  `_similar_precedents`): since 2026-11-20 ALL of them are in its cached system prompt, each
  with its context (`_rulings_block`), and each call adds a short list pointing at the ones
  that resemble the segments being judged (a real site or title match). Until 2026-11-19 it
  saw only the latest 20 of ~300.
* **Judgments** (redesigned 2026-11-19 to match the Rules doc). For every segment the evaluator
  gives: its content type; whether attention in it was **focused or unfocused**; and the switch
  into it, as `continuation`, `shift` (attention moved to something new) or `return`, with a
  pointer to the earlier segment it continues. On a shift it adds two yes/no judgments:
  **self-distraction** (sought out stimulation without a clear intent) and **next task** (the
  next task under the same task-processing intention). From these the app works out the rest:
  * A *thread* is a chain of segments linked by those pointers; a shift starts a new one.
  * A *focus span* is focused time on one thread. It starts with 10 continuous focused minutes.
  * A *diversion* (attention-interruption event) is any time inside a span that is off the
    span's thread, focused or not. A quick purposeful purchase is a focused segment and still a
    diversion. Under 5 minutes it is counted inside the span; at 5 minutes the span ended.
  * Unfocused time on the span's own thread counts as focus ("I may still be thinking"). When
    it is more than 20% of a span, or one stretch of 10+ minutes, the span panel asks whether
    it really was unfocused; "yes" makes that time a diversion. An open question of this kind
    counts in the Day page's "To review" number and puts the span in the Review queue, even
    if you already confirmed the span.
  * A *focused task block* is several threads joined by next-task shifts, under a running Toggl
    entry. Diversions are measured against the block.
  * A *self-distraction event* is a self-distraction shift at any time of day; each counts once.
    A *bounce-away* is the one shift by which you leave a focus span, when that shift is a
    self-distraction (`stats.self_distraction_counts`). Writing → email → Bluesky → writing is
    one attention-interruption event; it is a bounce-away only if the move to email was itself
    a self-distraction, and a self-distraction later in the same time away is an event but not
    a bounce-away. The tile, the Trends chart and the menu bar show self-distraction events.
  * A span that opens with 10 focused minutes of creative work keeps the kind "creative work"
    when the same thread turns into task-mode work, even when task time dominates and no Toggl
    entry is running (`_opens_creative`). Before 2026-11-19 such a span was thrown out whole as
    "task activity without a Toggl entry". Task work on another thread is still time away.
  * A brief look at a todo list to pick the next task belongs to that task's thread: the
    evaluator prompt puts the shift on the todo-list segment and makes the task a continuation.
  * Passing through windows on the way to something new: the pass-throughs are transitions
    (continuations) and the shift goes on the arrival segment, the same as for a return.
  * Going back to a distraction visited a few minutes earlier is a `return`, and it carries its
    own self-distraction judgment (the only return that can). When the model judges that visit
    stimulation-seeking, it counts as another self-distraction event, and as a bounce-away if
    it is what left the span.
  * A video watched in service of the current task (a tutorial on the thing you are stuck on,
    then back to the task) is part of the span: the evaluator labels it with the task's content,
    as a focused continuation. For segments judged under the redesign this also overrides the
    `passive_domains` rule (youtube.com is otherwise never focus).
  * Phone segments (2026-11-19). `stats.labelled_segments` returns segments with the phone
    segments that were time away carved in, so a Mac segment cut by one appears twice with the
    same id, and the phone segment itself has a negative id and no switch. Code that needs one
    row per real segment (the evaluator's own log, the daily audit) passes `phone=False`. A
    database that has no `phone_evals` table yet (opened read-only before the app has migrated
    it) is read as "no phone use". Re-judging the computer segments along with late phone data
    was tried and dropped: one call cost $0.13 instead of $0.02 and rewrote 23 settled judgments.
    A phone segment the model leaves out is stored as a continuation, so it is not queued again.
    Pickups older than 3 days when their data arrives are not queued (`PHONE_LOOKBACK_S`).
    The first draft judged a pickup as a whole, with its own short rulebook; it called most 3-10 s
    looks at the app the phone unlocked onto a shift. Judging each app segment by the ordinary
    rules lets those be pass-throughs.
    Carving can remove a span: the opening 10 unbroken focused minutes are measured after the
    phone time is cut out.
    What the phone log cannot show (checked 2026-11-19 against 40 days of it): it is a stream of
    "app in focus" records, with no record for the screen waking or the lock screen itself. A
    3-second "Oura" pickup is not a lock-screen glance: a start record with no transition reason
    is the phone being unlocked onto the app that was open when it was locked. There were 110
    single-app pickups under 6 seconds in 30 days, 94 of that kind.
  * A touch-back of 15 seconds or less followed by leaving again does not reset the 5-minute
    clock (`TOUCH_BACK_S`): the absences on either side are one absence.
  * One attention-interruption event per time you leave the span: everything passed through
    while away is merged into one diversion (`_assemble`).
  Segments judged before the redesign have no focused/unfocused field. They are read the older
  way: a four-way label on the switch (`continuation`, `interruption` = detour, `distraction`,
  `focus start`, internally `task_change`), with the state inherited through continuations.
* **Phase 1 = focus spans.** `count_events = false` hides interruption/distraction counts everywhere;
  the labels are still produced because they decide where a span breaks. The Day view shows each
  span with what ended it (interrupted / distracted / away / new task) and the Toggl entries inside it.
* **Nested spans** (`break_min`, `min_focus_span_min`): a focus span is a container. Detours
  (interrupted/distracted) shorter than 5 min are events *inside* the span and don't end it; a detour of
  5+ min, being away, a recording gap, or a deliberate focus start ends it. A span must contain ≥ 10 min
  of focus (`min_focus_span_min`; it was 15 until 2026-11-17, against the Rules doc's 10), including one
  continuous block of 10 min with no contained detour (`min_unbroken_min`). "Length" is wall-clock; "focused" excludes contained detours. The span
  starts at that block: a choppy opening before the first clean 10 minutes is cut off and kept as short focus.
  The opening block has to be time judged focused.
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
* **Follow-up after a correction** (`evaluator.followup_question`, `/api/questions/{id}/followup`):
  when you answer a review question against the app's judgment (or say "No" to a spot check),
  the review card replays that segment's evaluation run and lets the model ask one short question
  about your reason; it shows in the question's chat a few seconds later. It replies "No
  question." (nothing is shown) when your answer or note already explains it, and no call is made
  when your answer agreed with the app or the question was about the phone.
* **The question writer sees your past rulings** too (`questions.generate_pending`): the same
  similarity-picked precedents as the evaluator, with the instruction to drop a doubt that a past
  ruling already answers.
* **Rules drafted from your answers** (`rule_drafts.py`): once 8 new rulings have come in (and at
  most every 6 hours, one model call, about $0.20), the evaluator model reads your newest 150
  rulings next to the Rules doc and drafts up to 3 lines for patterns that at least three rulings
  share and the doc does not state. Drafts appear on the **Rules** tab (the tab
  shows a count while any wait) as "Proposed rule" cards. Under each card are the review answers
  it rests on: the capture, when and where it was, the question you were asked and your answer,
  your ruling next to what the app had said, your note, and a link to open the capture. Edit the
  wording if you like, then "add to my rules" (appended under "Lessons from
  reviews" in the Rules doc, which every model reads) or "no, not a rule". Decided drafts are
  listed at the bottom of the tab. Rejected drafts are
  shown to later runs so they are not proposed again; no new drafts while 5 are waiting.
* **Spot checks** (`questions.spot_checks`, `spot_checks_per_day`, default 3): a few times a day
  a segment whose judgment the evaluator was confident of is drawn at random (chance proportional
  to its length; at most one per 2 hours) and asked about in the review queue. Each check asks
  about one axis only, also drawn at random: the kind of activity, focused or unfocused, or the
  switch ("Spot check: the app judged the 3 min from 10:50 as a continuation of what came before.
  Is that right?"). Above the question is a 2-3 sentence summary of what was happening, written
  by one small model call (about $0.04; it is told not to hint at the verdict); if that call
  fails the card shows a bare where-and-when line instead. "Yes" stores that axis as your own ruling. "No" on focused is the opposite ruling; "No"
  on the other two flags the segment so the usual question about what it was gets written.
  (That flag carries a mark in the reasoning, "spot check: the person says ...", which exempts it
  from the short-thread floor: without it 86 of 283 eligible segments would have had the "No"
  dropped with no follow-up. Clearing the answer removes the mark and the verdict.)
  Outcomes are in `agreement_samples` (with the `axis` asked), and the
  first tile on the agreement panel gives the share agreed, by week. This is the unbiased
  accuracy figure: the other questions are about segments the app already doubted.
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
  change; it takes a couple of days of self-locks to show up.
* **Phone app segments are judged like computer segments** (since 2026-11-19; Eli: "The app
  should make the same segment judgements of open phone apps as it does of computer segments").
  `phone.pickups` groups the phone log into pickups: one per stretch of use between locks
  (records under 60 s apart; a put-down record counts for its first 10 s; under 3 s of use in
  all is ignored). Each pickup carries its app segments in order. The evaluator sees every app
  segment as a `PHONE` line in the activity log, slotted in by time, marked "(phone unlocked)" on
  the first of a pickup and "(first Oura open today)" on an app's first use of the day, and judges
  it with the fields of a computer segment: content, focused, switch (continuation / shift /
  return), and on a shift self-distraction and next-task. Only the app and the timing are known;
  there are no phone screenshots. Judgments are stored in `phone_evals`, one row per app segment.
  When a row is stored the app works out `away`: a shift is time away from what was on the Mac,
  and so is a continuation of a phone segment that was away; a continuation of the Mac thread, or
  a pass-through (content `transition`: the few seconds on whatever app the phone unlocked onto),
  is not.
  A phone segment that is away is cut out of the Mac segment it overlaps (`stats._carve_phone`):
  it becomes a block of its own on the timeline (app "iPhone", the phone app as its site, a
  negative id), the Mac window before and after it stays one thread, and the phone time is time
  away from the span. So one trip to the phone is one attention-interruption event, 5 minutes of
  it ends the span, and each shift judged a self-distraction is a self-distraction event (and a
  bounce-away, when it is what left the span). A phone segment that is not away, or not judged
  yet, changes nothing.
  Phone time that is on track (a continuation of the computer thread, e.g. a call that is the
  task) counts as part of the span: nothing changes while the Mac records activity, and where
  the Mac went idle meanwhile that piece continues the last active segment instead of ending
  the span (`stats._phone_on_track`). Phone time that was a shift is always time away and never
  starts a span of its own; phone-only focus (meditation, cardio intervals) is tracked in Toggl.
  Phone judgments are low confidence (the model sees an app name and timing, nothing else) and
  the phone is secondary: a phone block's tooltip says so, and a phone judgment becomes a review
  question only when it is load-bearing (`questions.phone_questions`): reading that phone time
  the other way (away from the work, or part of it) would move the day's independent focus time
  or longest span by `PHONE_Q_MIN` (10) minutes or more, and the phone use lasted 30 s or more
  and was not a pass-through. The question is a fixed one with four
  answers, written without a model call, and anchored on the computer segment the phone time
  falls in (so that segment shows the needs-review dot, and its own doubts wait until the phone
  question is settled). The answer is stored in `phone_reviews` and overrides `phone_evals`.
  What matters from the phone is interruptions and self-distraction events.
  A glance at the lock screen is an interruption and never a self-distraction (Eli's rule); that
  is set in code (`evaluator.store_lock_glance`), not asked of the model. The phone log records
  a lock-screen look only when you swipe to the widgets page without unlocking
  (`phone.lock_glances`: 6 on their own in 40 days, 14 more on the way into an unlock, which
  belong to that pickup); a plain glance leaves no record.
  The phone log arrives hours to a day after the fact, so most pickups are judged late:
  `evaluator.queue_phone_rejudge` (every classifier cycle; does work only when the phone data
  has advanced, or every 30 minutes) puts each stretch holding an unjudged pickup into
  `reeval_queue` with the reason "phone data arrived", and `run_reevaluations` judges the phone
  segments in it against the activity already judged. The computer segments are shown as context
  and are not judged again. A day's numbers can therefore change when the phone data lands.

### Signals beside the captures (`signals.py`, since 2026-11-19)

macOS says that an app came forward, never why. So the collector also records what can be
observed around each switch, and the evaluator reads it as notes in [brackets] at the end of
each line of its activity log (the Captures card shows the same notes as "Signals:"). None of it
needs a permission beyond those the app already has. `collect_signals = false` in `config.toml`
turns it all off; pausing tracking pauses it too.

| Signal | How it is read | Stored in |
|---|---|---|
| Keys, clicks, scroll and pointer-move events per 10 s | the system's per-type event counters (`CGEventSourceCounterForEventType`). Counts only: no key is ever read | `activity_samples` |
| Audio playing, microphone on, camera on | CoreAudio / CoreMediaIO "device is running somewhere", every poll | `activity_samples` (seconds on) |
| What the last click before a switch landed on | four times a second a watcher thread checks the click counter; when it has gone up, the app whose window is under the pointer (Dock, a notification banner, the previous app...) | `ui_events` (every click), `segment_arrivals` (per segment) |
| Whether a key was pressed just before a switch | the key counter, same thread | `segment_arrivals.key_age` |
| A notification banner on screen | a Notification Center window above the desktop layer. It says that a banner showed, not which app sent it | `ui_events`, `segment_arrivals.banner_age` |
| How Chrome reached each page | every profile's `History` file, read every 30 s (typed, link from a page, new tab from a page, link from another app, bookmark, address-bar search, back/forward) | `page_visits` |

From these the log says, per segment: "arrived by: a click on the Dock / on a notification
banner / inside <the previous app> / the keyboard, no click / no click and no key press", "a
notification banner appeared N s before", "page reached by: ...", "input: N keys/min ...", and
"audio playing / microphone on / camera on N% of the time". Section 6 of the evaluator's prompt
tells it how to weigh them: evidence, not verdicts.

Not collected, because each needs Full Disk Access for the app and could not be tested without
it: which app sent a notification (`~/Library/Group Containers/group.com.apple.usernoted/db2/db`)
and whether a Focus mode is on (`~/Library/DoNotDisturb/DB/Assertions.json`).

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
`collect`, `classify [--once] [--no-claude]`, `web`, `stats --days 14`, `rejudge-wrong-display` (see Gotchas).

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

Every call that reaches the API also carries the second shared tool, `note_to_eli` (added 2026-11-20),
which leaves Eli a message without stopping anything: the note lands in the same inbox (marked
`via: note`, `halts: false`, a quieter notification) and the task carries on. Structured calls, the
switch and span chats and the evaluator chat make one request and have no tool loop, so
`llm.with_notes` (the shared `llm_contact.with_notes`) records the note, answers the call with
"Delivered to Eli. Nothing has stopped" and makes the same request again for the answer (up to three
times; each request is logged in `llm_calls` and the cost is summed). The day chat runs its own tool
loop, so its dispatcher returns that acknowledgement like any other tool result. A note is never read
as the answer; text the model wrote in the same turn as a note stays in a chat reply. A response
that carries both a note and a `contact_eli` call records both and stops as above. The subscription
backend has no tools, so no note tool there.

## Files

| File | Purpose |
|---|---|
| `focus_monitor/__main__.py` | Entry point: `collect`, `classify`, `web`, `app`, `run`, `stats`, `check`, `restart`. Sets up logging. |
| `focus_monitor/config.py` | `Config` dataclass, `config.toml` loader, paths (data dir, log, Keychain lookup), pricing constants. |
| `focus_monitor/phone.py` | Read-only view of the phone-unlock-logger's database: per-app phone segments and unlocks for a window, pickups (what the evaluator judges), app-name map, and how far the phone data reaches. |
| `focus_monitor/collector.py` | Active-window polling, screenshots, idle detection, segments and switches. |
| `focus_monitor/classify.py` | Classifier service loop: waits the lookahead, runs the ensemble, flags uncertain switches. |
| `focus_monitor/classifiers/` | `claude_vision.py` (Claude + screenshots), `heuristic.py` (rules), `learned.py` (logistic model), `base.py` (switch context). |
| `focus_monitor/ensemble.py` | Weights the classifiers by recent log-loss. |
| `focus_monitor/evaluator.py` | Segment evaluator (content / switch / kind per segment), daily audit, evaluator chat. |
| `focus_monitor/llm.py` | The one door for structured model calls: api vs. subscription backend, limit handling, cost log, shared preamble + `contact_eli` / `note_to_eli` + inbox check. |
| `focus_monitor/usage_log.py` | Appends every call to `~/.claude/app-usage/Focus-monitor.jsonl` for the Claude usage dashboard. |
| `focus_monitor/chat.py` | Interactive chats about one switch, one span, or a whole day (the day chat edits records through tools). |
| `focus_monitor/shots.py` | Picks which captures of a stretch a model is shown: one per capture moment, the display that held the active window, spread evenly. Used by the evaluator, audit, question generator, propagation, summaries and chat. |
| `focus_monitor/questions.py` | Writes review questions for doubted segments; ranks the queue. Each question is about one dimension (thread, content, or hybrid); a doubt uncertain along two gets two questions on the same segment, and options that would land the same judgment (blank fields count as "as the model has it") are merged. Every question's options are the model's readings plus the app's own pass-through option ("just a window I passed through"), which is left out when the segment was plainly in use: held longer than `transit_seconds` and given under 10% chance of being a transition by the evaluator. |
| `focus_monitor/propagation.py` | Applies a correction to similar recent segments (feedback propagation judge), and applies it again after a re-judgment. |
| `focus_monitor/resolver.py` | Attention-thread resolver: which thread a switch continues. |
| `focus_monitor/summaries.py` | Span summaries and coherence checks. |
| `focus_monitor/efforts.py` | Effort-ladder shadow calls and the effort judge (which effort level is good enough). |
| `focus_monitor/edits.py`, `seg_models.py`, `stats.py`, `toggl.py`, `db.py` | Edits with provenance, label space, metrics/spans, Toggl sync, SQLite schema and helpers. |
| `focus_monitor/signals.py` | Extra signals beside the captures: input counts, audio/mic/camera, what was clicked before a switch, notification banners, how Chrome reached each page (tables `activity_samples`, `ui_events`, `segment_arrivals`, `page_visits`). |
| `focus_monitor/rule_drafts.py` | Drafts Rules-doc lines from repeated review rulings; you approve each (tables `rule_drafts`, `rule_draft_runs`). |
| `focus_monitor/span_selftest.py` | Self-test of the span rules on hand-built days (`python -m focus_monitor.span_selftest`; run by `test.sh`). |
| `focus_monitor/web.py` | FastAPI server for the dashboard and its API (port 8790). |
| `focus_monitor/dashboard.html` | The whole dashboard (Day, Captures, Trends, Review, Models, Efforts). Drafts in every text field persist in `localStorage`. |
| `focus_monitor/menubar.py`, `icons.py` | The menu bar eye (rumps) and its icons. |
| `build_app.py`, `build.sh` | Build `dist/Focus Monitor.app` (signed native launcher wrapping the venv). |
| `install.sh` | Idempotent install: build, register the Login Item, relaunch; `--launchd` for the headless variant. |
| `focus_monitor/llm_selftest.py` | Self-test of the LLM inbox wiring at every call site with a fake client: a note carries on, `contact_eli` and `@ELI:` stop (`python -m focus_monitor.llm_selftest`; run by `test.sh`). |
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
.venv/bin/python -m focus_monitor.span_selftest   # just the span rules, on hand-built days (a few seconds)
.venv/bin/python -m focus_monitor.llm_selftest    # just the LLM inbox wiring: fake client, throwaway inbox, no network

./build.sh                               # -> dist/Focus Monitor.app   (same as .venv/bin/python build_app.py)
./install.sh                             # build + Login Item + relaunch if running; safe to rerun
open "dist/Focus Monitor.app"            # or drag to /Applications; add to Login Items
.venv/bin/python -m focus_monitor app    # dev alternative without the bundle (menu bar eye)
.venv/bin/python -m focus_monitor run    # headless: collector + classifier + web, no icon

.venv/bin/python -m focus_monitor restart            # running app relaunches itself with the new code
.venv/bin/python -m focus_monitor restart --if-stale # only if a source file is newer than the process
nohup .venv/bin/python -m focus_monitor restart --when-idle >/dev/null 2>&1 &   # wait for 2 idle minutes, then restart if stale

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
* **Screen Recording** (screenshots, and window titles for the apps in `title_apps`)
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

* **Never commit a `.venv` symlink from a worktree (2026-11-20).** A worktree needs the main
  checkout's environment, so a `.venv` symlink gets made inside it. `.gitignore` had `.venv/`,
  which matches a directory but not a symlink, so `git add -A` committed it; merging that branch
  made git replace main's real (ignored) `.venv` with a symlink to itself, deleting the
  environment. The pattern is now `.venv`. In a worktree, add files by name. The app bundle
  reaches its packages through `.venv/lib/python3.12/site-packages`, so a missing environment
  means the app cannot start again after its next restart.

* **Audit, 2026-11-20 (seven reviewers; numbers, raw record, what the models are shown).** The
  open questions it raised are in `OPEN-DECISIONS.md`.
  * *Short pauses.* A raw idle segment under `idle_seconds`, and time the evaluator or the person
    labelled content "idle", is a pause (`pause=True` in `stats._labelled_segments`): it takes the
    state of what came before, stays inside a running span, never opens one, resets the opening
    10 minutes, and is not unfocused time. Contiguous pause time reaching `idle_seconds` is time
    away. 807 of 1,541 idle segments were under 10 minutes and each ended a span; 17 spans held
    5+ minutes of idle-labelled "focus".
  * *Time zones.* `config.local_dt`, `logical_date` and `day_bounds` use the zone in force at the
    time. Never use `datetime.fromtimestamp` for a day boundary or a time a model reads. The Mac
    went from Pacific to Eastern at 2026-11-08 22:47 Eastern (where the log's offset changes).
  * *Old unjudged tails.* `spans_and_events` stops at the last judged segment only when the
    unjudged tail is recent (`FRONTIER_RECENT_S`, 4 h). It used to do so on past days too and
    dropped the rest of the day from spans and unfocused minutes (32 of 87 days).
  * *Reading gate.* Content "creative" in a reading span counts as working with the material.
  * *Toggl.* Coverage is seconds of overlap (`_toggl_cover`), not "an entry was running at the
    segment's start"; with overlapping entries the latest-started wins, as in `toggl.entry_at`.
    The old end of a running entry is the end of the newest judged segment, so a night of failed
    syncs no longer leaves judgments standing under an entry that had stopped.
  * *Rulings from before the `focused` field.* A person's kind or focus-exception ruling is turned
    into focused / self-distraction in `stats.verdict()` when the evaluator's row has `focused`,
    so a re-judgment cannot take the state away from them.
  * *The evaluator's evidence.* Up to `IMAGE_BUDGET` (60) screenshots a call, every segment to
    judge first; a stretch with more segments than fit is judged over several calls
    (`last_end`). Until now every call had 8 images and 29% of targets were judged with no
    screenshot though one existed. Both displays are sent. A re-judgment never replaces a
    judgment whose screenshots are gone: files exist only from 2026-10-27, and the history
    re-judgment queues only days from the first file on.
  * *Evaluator lines.* `summaries.seg_line(s, full=True)`: full title, URL and note, seconds,
    page loads, input by minute. A line with no judgment shows no state word, old `<switch>` tag
    or `[activity]`: those were the per-switch classifier's guesses and read as judgments. A
    ruled line says RULED BY THE PERSON with the note. Context is 45 minutes before, up to the
    time of the call after, plus a DAY SO FAR block.
  * *Budget.* `db.spend_today` leaves out what the subscription served: one evaluator call is now
    $0.50 to $1.30 at API prices, and counted in full it would stop all judging by the afternoon.
  * *Collector.* Tab-title counters ("(1) Notifications / X") no longer start a segment
    (`same_window`). A restart while away moves the captures of every segment it cuts back.
  * *Review answers.* An option can carry `self_distraction` (on purpose / a drift) and answering
    writes it. Options written before 2026-11-20 do not have it.
  * *The daily audit* runs in pieces over several cycles (`kv` `audit.progress.<day>`), every
    audited segment with its screenshots.
* **Bug hunt, 2026-11-20.** Things that looked right and were not:
  * Second pass the same day (a review of that day's own fixes). Re-evaluations: a failed call
    waits five minutes (`kv` `reeval.retry_after`) and is retried for as long as the error lasts;
    three empty answers pass over that one chunk, not the day's whole queue row (an hour offline
    would have closed every day of the history re-judgment as done). History calls never fall
    back to the API key (`llm.only_subscription`). `repair_outvoted` no longer raises the doubt
    flag (it would have asked 339 questions about September). The `seg_propagated` migration
    records only the fields the source ruling set, a ruling's old records are dropped when it
    propagates again, and marks are not stacked. An empty first-pass answer is recorded, held ten
    minutes, and after three tries the stretch gets a flagged default instead of a call every 15
    seconds. An older day's span summaries are kept when it is re-judged (`ensure_summaries`
    rewrites only the last 7 days).
  * Everything judged before the 03:34 restart on 2026-11-20 was judged at low effort. The
    history re-judgment is built but not started: `python -m focus_monitor rejudge-history --start`
    queues one `reeval_queue` row per day (reason "judged at low effort"); the running app then
    works through them newest first, after all other work, at most one call a minute
    (`--every`), never on the API key (it waits while the subscription is at its limit), and
    outside the daily budget (`eval_runs.budget_exempt`). `--pause` / `--resume` stop and continue
    it; with no flag it prints progress. The judgments as they stood are copied to
    `seg_evals_low_effort` first.
  * A focus span running past 4am was assembled as two pieces, one per day, and each had to
    qualify alone. `spans_and_events` now looks up to 8 hours past the edge of its window when
    activity crosses it (`_whole_spans`), and the piece of a qualifying span that lies inside the
    window skips the length, opening-block and Toggl gates. Each day still counts only its own
    minutes (Eli, 2026-11-20). Two days in 90 changed, by about 6 focus minutes each.
  * Undoing a chat's edit deleted the whole ruling, including one the person had made before the
    chat touched it. What a chat replaces is now kept in `review_undo` (blocks and span verdicts),
    and undo puts it back; a second undo then removes that earlier ruling too.
  * `/api/span/queue` took 6 to 7 seconds: `phone.autolock_s` scanned four days of the phone log
    once per span. It is cached per six hours of the day for ten minutes (2 s cold, 0.2 s warm).
  * The dashboard had the day's start hour written in as 4; it now asks `/api/day-start`.
  * A ruling that agrees with the model was described to the propagation judge as a correction.
    It is now called a confirmation.
  * Segments left unjudged at a 4am rollover before the one-day look-back existed are judged by a
    one-off backfill in `run_pending` (one chunk per quiet cycle; `kv` `backfill.unjudged` when done).
  * A new judgment replaces the whole `seg_evals` row, and a correction propagated from a ruling
    on a similar segment lived only in that row, so every re-judgment erased it. Propagated
    corrections are now also kept in `seg_propagated` and applied again after the evaluator writes
    (`propagation.reapply`), unless the ruling they came from was cleared. A one-off migration
    records the ones still present and re-queues recent rulings whose corrections were lost.
  * The stored switch is the evaluator's own answer. It used to be the weighted vote with the two
    local models, which could outvote an unsure evaluator; the other fields stayed the
    evaluator's, so a row could say "continuation" under a note reading "Left X for Y" (497
    rows). A disagreement now only flags the segment. `evaluator.repair_outvoted` put the
    evaluator's switch back on the old rows once; the values it replaced are in
    `seg_evals_outvoted`. Content was never outvoted and is still the vote.
  * A re-evaluation that came back with no judgments was marked done and retired the open
    questions for that stretch. It is retried up to three times and the questions stay.
  * Phone time with nothing on the Mac in it is "distracted" when its pickup began as a
    self-distraction, and plain unfocused time otherwise (Eli, 2026-11-20). `_detour_label` used to
    want a self-distraction shift inside the stretch itself, so the later pieces of such a pickup,
    cut off by a pass-through app or a second of idle Mac, were counted as interruptions.
  * The evaluator's context lines had no segment ids, so a segment continuing one could not point
    at it; the evaluator named a later segment and the pointer was dropped (195 "dropping invalid
    antecedent" warnings in a week). Context lines are now "(context id=N)".
  * The Rules tab compared the ruling and the app's judgment as strings; a ruling on one axis
    that agreed with the app looked like a correction. `evidence()` now returns `corrected`,
    compared field by field.
  * The review queue leaves out questions not written yet. If ALL of them were unwritten the tab
    said "Nothing to review" for good; it now loads the first one that becomes ready. And after
    a late question was shown, "next" replayed every card after its place in time (`qPastEnd`).
  * **The evaluator ran at effort "low" from 2026-08-31 to 2026-11-20** while `config.toml` asked
    for "high": the call had the level written into it. Its shadow calls were told the primary was
    "high" and ran "low" and "medium", so the Efforts tab's evaluator rows from that period compare
    low with low, not with high. Judgments from before the fix were made at low effort.
  * Whatever was unjudged at the 4am rollover was never judged (the evaluator looked back to the
    start of today only): 162 segments in 30 days. It now looks back one more day.
  * A long segment starting just after a short one was judged with it, on its first ten minutes.
    A first pass now takes only segments that ended 15 minutes ago.
  * A span could bridge idle time when the person had left the task just before going away
    (27 spans since August). `span_selftest` has the case.
  * A model answering in a fixed JSON shape can only write `@ELI:` inside a field; the contact
    check reads response text line by line and missed it. `llm.contacted` now hands such fields
    over as text of their own.
  * Pause was kept in memory only; any restart resumed recording. It is in `kv` now.
  * The block editor on a capture card sent the evaluator's content, switch and hybrid as the
    person's ruling with every click, and dropped the thread pointer and note of an earlier
    answer. It sends only what the person has ruled.
  * Keys 1-4 on a span card reviewed whichever capture had been opened last (`current` was not
    cleared). 1,008 captures without a first screenshot could not be opened at all.
  * Mini-timeline blocks opened the capture of their transition GROUP (`switch_id`), not their own
    (`own_switch_id`): a different segment for 27% of blocks.
  * `git add <folder>` is not safe here when another Claude session is working in the same
    checkout: stage files (or hunks) by name.
* **Signals (2026-11-19): what was tried and what the data showed.**
  * A listen-only event tap on mouse clicks can be created without any permission, but macOS
    delivers it no events unless the app has Input Monitoring. The click and key COUNTERS and
    the pointer position need no permission, so the watcher polls those instead. A click is
    placed against the window list from the poll before (a quarter second earlier), because the
    click itself often changes what is on screen. It must not be older than that: the list covers
    only the current Space, and with a 1-2 s old list a swipe to another Space followed by a click
    was reported as "a click inside" the app just left (found in the first hour of live data,
    2026-11-20). Reading the list four times a second costs about 2 ms each time.
  * "No visit in Chrome's History" means "the tab was already open" only if History has been
    read for that stretch. The stretch read so far is kept in `kv` (`signals.visits_from`,
    `signals.visits_to`); judging it by the newest visit's time made the note appear or vanish
    depending on whether some later page happened to load.
  * SQLite's LIKE has no escape character unless the query says `ESCAPE`; without it, escaping
    `_` in an address made the pattern match nothing.
  * While a banner shows, Notification Center puts up a window the size of the whole display
    (layer 21), so "which window is under the pointer" would say Notification Center for every
    click. Only its top right corner (420 x 260) counts as the banner. Its desktop widgets are
    separate windows on a negative layer.
  * Other apps float windows above the normal layer that clicks pass through (Superwhisper keeps
    a 240 x 160 one mid-screen), so above layer 0 only the system's own surfaces count as click
    targets: the Dock, the menu bar, Spotlight and the banner corner.
  * Chrome's transition values do not mean what their names say. "Typed" without the
    from-address-bar flag is a link opened from another app (28 of 39 such visits in a week began
    with another app in front). "Reload" is mostly a single-page site (X) moving between its own
    pages, so it is not reported. "Link" with the from-API flag is as often an extension.
  * Chrome keeps `History` locked while it runs; opening it with `?mode=ro&immutable=1` reads it
    anyway. A read that races a write can fail; the next round (30 s) picks it up.
  * Microphone "on" is also true during voice dictation, not only on calls.

* **The app's abstractions were rebuilt to match the Rules doc (2026-11-19).** Before, the
  evaluator gave each shift a "kind" (focus start / detour / self-distraction) and every later
  segment inherited that through continuations. That one field was doing four jobs: saying
  whether time was focused, whether it was a diversion, whether it was a self-distraction, and
  where a span could start. The doc keeps these apart, so the app now does too:
  * `seg_evals` / `seg_reviews` have `focused`, `self_distraction`, `next_task` (NULL on rows
    from before the change). `evaluator.SegJudgement` asks for them; `shift_kind` is gone from
    the model's output. The old `interruption_kind` column is still written, derived by
    `evaluator.kind_of`, so the local models, old corrections and old code paths keep working.
  * `stats.labelled_segments` takes a segment's state from `focused` when it has one. "Passive
    is never focus" and "correspondence without Toggl is never focus" are still vetoes.
    A pass-through or input-idle segment keeps inheriting the state before it: a pause in
    focused work stays focused, and only the collector's 10-minute idle ends a span.
  * `stats._runs3` adds *soft* runs: unfocused time that continues a thread. `_assemble`
    counts them as focus inside a running span on that thread; they cannot start a span or
    count toward its opening 10 minutes. Time the person ruled unfocused is never soft.
  * `stats._same_task` reads the evaluator's judgment before the "same window" shortcut for
    these segments (Rules doc: hand-coded rules may veto, never affirm): a shift is a new
    thread even in the same window, unless it is a next-task shift under a running Toggl entry.
  * "Self-distraction" is about intent, not content. A planned video break is passive and
    unfocused but not a self-distraction, so passive content no longer forces "distracted" on
    redesigned rows; a stretch of time away counts as a self-distraction when it contains a
    self-distraction shift.
  * Names: the app had "bounce-aways" as the all-day count and "self-distractions" as the
    in-span count; the doc has them the other way round. `daily_metrics` now returns
    `self_distraction_events` (all day) and `bounce_aways` (in spans).
  * A return to something left more than 10 minutes ago is stored as a shift (`RETURN_MAX_S`).
  * Hold-off: the evaluator may answer with `hold_from`, leaving the end of a chunk to be
    judged later (table `eval_holds`, retried after 15 minutes, allowed for an hour).
  History is untouched: every daily number for the 28 days before the change came out
  identical under the new code (checked field by field). Checked with `python -m
  focus_monitor.span_selftest` (ten hand-built days), two real evaluator calls on a copy of the
  database, and a simulated hold. Not checked with real model calls: the question generator,
  day chat and propagation prompts, which were reworded to the new fields.

* **"Shift" and "interruption" are two things (renamed 2026-11-17).** The Rules doc defines an
  attention-interruption event as leaving the object of a focus span. The evaluator's switch
  label of the same name meant something else: attention moved to something new at this
  switch, at any time of day. One word for both left the evaluator unsure what a switch could
  "interrupt" when no span was running. Models and the dashboard now call the switch label a
  **shift** (kinds: detour, self-distraction, focus start); "interruption" is kept for the
  span-level event that the app derives and counts. Only the wording changed: the stored value
  is still `interruption` (`seg_evals`, `seg_reviews`, the four-way `ensemble` labels), and
  `seg_models.shown_switch` / `stored_switch` translate at the boundary, so nothing was
  migrated. The evaluator's output field is now `shift_kind`; `eval_runs.response` rows from
  before the change say `interruption` / `interruption_kind`. Every model prompt uses the new word: the evaluator, question generator, day chat and
  propagation say `shift` / `shift_kind` and are translated back on the way in (`edits.py`
  accepts either). The older per-switch classifier keeps its four class names (`interruption`
  there is the class for a detour) with a sentence explaining the naming.

* **A span needs one continuous 10-minute block (rule from Eli, 2026-11-17).** The break rule
  alone (an absence under `break_min` does not end a span) turns project A 4 min / B 3 min /
  A 4 min ... into one long span on A with B as contained interruptions, which the Rules doc
  says is not a focus span. `stats.spans_and_events` now disqualifies a span whose longest
  stretch with no contained interruption is under `min_unbroken_min` (10; "no continuous block
  of 10 minutes"); meetings are exempt, and a span review still overrides it. This lives in the
  span logic because the evaluator cannot express it: the true labels (left A, returned to A)
  are what assemble the span. Measured on the 21 days before the change: 12 of 83 non-meeting
  spans (242 of about 3,250 focus minutes) fall under the bar; they spend 22% of their length
  on diversions against 5% for the rest. A bar of 5 minutes would have removed none.
  Together with lowering `min_focus_span_min` from 15 to 10 the same day, the 21 days go from
  103 spans to 130 (about 3,250 to 3,490 focus minutes).
* **Where "return" goes when the way back passes through other windows (2026-11-19).** The doc
  said navigating back is a continuation and also that coming back is a return, so it was unclear
  which segment carries the `return` label. Eli's rule: the windows passed through are
  transitions (switch `continuation`), and `return` goes on the segment where he arrives back on
  the old thread. The Rules doc and the evaluator's instructions both say so now.
* **A focus span starts with its 10 continuous minutes (2026-11-19).** The rule above only
  required a 10-minute block somewhere in the span, so a span could open with several broken
  minutes and the clean block could come later. Eli's rule is stricter: "every focus span has
  to start with at least 10 consecutive focused minutes on the focused object."
  `stats.spans_and_events` now cuts each span's start forward to its first unbroken
  `min_unbroken_min` block (`trimmed_start_min` on the span records how much); the cut opening
  becomes short focus, or a disqualified span if it holds 10 focus minutes of its own. Meetings
  are exempt, and a span you confirmed in a review keeps its reviewed bounds. A review of a
  span that loses more than 20% of its length no longer matches it (reviews need 80% overlap),
  so that span shows as unreviewed again. Measured on the 21 days before the change: no span
  is lost, 25 of 110 non-meeting spans start later, and focus time drops from 2,996 to 2,851
  minutes (about 5%); the longest span of the day gets shorter on 7 of 21 days.
* **Two Agent Coding projects were always "the same task".** `_same_task` returned true for the
  same app and site before looking at anything else, and Agent Coding has no site. For apps in
  `title_apps` a different window title now skips that shortcut, so the thread pointer and the
  evaluator's judgment decide. Segments recorded before 2026-11-17 have no title and behave as
  before.

* **Non-browser window titles were never recorded (found 2026-11-17).** The collector asks
  System Events for `name of front window`, inside a `try` that swallows the failure, and it has
  returned nothing for every non-browser app since the first day: every such segment has an
  empty title (browser titles come from the browser's own AppleScript). So a change of project
  inside Agent Coding, which does put the project in its window title, never started a new
  segment, and the evaluator could tell projects apart only from screenshots. Apps listed in
  `title_apps` (default `["Agent Coding"]`) now get their title from the window list
  (`CGWindowListCopyWindowInfo`), which needs only Screen Recording. It is a list, not every
  app, on purpose: a new title starts a new segment, so an app whose title ticks (a timer)
  would shred the timeline, and Signal's title names the conversation. Titles in the window
  list come back empty for a process without Screen Recording, so `focus_monitor check` run
  from a terminal that lacks it shows no title even when the app records one.

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
* A flagged segment whose question is still being written is left out of the Review queue (since
  2026-11-19; before, a "being written" placeholder card sat in its place in time). It joins the
  queue when the question is ready: the page re-checks every 20 s, and one that lands behind the
  card you are on is shown after the last question. Opening such a segment on purpose from the
  Day tab still shows its placeholder.
* Review questions are shown in time order, oldest first (since 2026-11-16), so next/prev move to
  the neighbouring question on the same day. Before that the priority tier came first, and "next"
  jumped between days. A needs-review dot on the Day timeline opens its question inside that queue
  (the switch card only when the flag has no open question), so the arrows carry on from there.
  Opening the Review tab always starts at the first open question of the day the Day tab is on
  (none that day: the next later one).
* A needs-review dot with no question behind it means a flag was raised after the segment's
  question had already been declined, dismissed or archived: the generator counted that old row as
  "asked" and never looked again (segment 34836, 2026-11-16, flagged by a propagating correction).
  Those rows no longer count, so the next pass either writes a question or declines again and
  clears the dot. Note `questions._triggers()` is not read-only: it declines short-thread doubts
  as it gathers, so calling it by hand changes the live database.
* The review-question queue (`/api/questions/queue`) returns every open question, so anything
  computed per question there is multiplied by the backlog. With ~460 priority questions open
  (2026-11-14) the per-question timelines made it take 25 s and 10 MB, and the Review tab sat on an
  empty card that whole time. The queue now carries no timelines; each card fetches its own from
  `/api/questions/{id}/context` when shown (and warms the next one).
* A segment can be flagged for review with no switch into it (the first segment after a wake from
  sleep, e.g. 34062). Its "needs review" dot has only a question to open; treating the missing
  switch id as a switch drew an empty card.
* With an external monitor every capture moment stores one file per display, and the models
  were shown every k-th ROW of `segment_shots`. The rows alternate laptop, external, so an even k
  meant laptop only: on 2026-11-16 three minutes of commenting on a bill draft (external) were
  judged as "the Sleep Bandit page sitting still" (laptop) and a question asked about a data
  conflict that did not exist. 195 of the 1,114 evaluation runs that involved a second display saw
  only the laptop (an upper bound on the damage: sometimes the laptop was the active one).
  `shots.pick()` now samples moments and takes the active window's display. `eval_runs.shots`
  records what a run was shown, which is how to check a suspicious judgment.
* To re-judge what the wrong-display bug touched: `.venv/bin/python -m focus_monitor
  rejudge-wrong-display` queues every stretch whose current judgment came from a run that showed a
  segment only on the non-active display (81 stretches, ~16.5 h of activity, when first run). The
  running app works through `reeval_queue` one chunk per classifier cycle, newest first, inside the
  daily budget; each re-judged chunk retires the open questions written from the old judgment.
* A thread question saying the model is "0% sure which thread this continues" came from a
  malformed evaluator answer: every candidate antecedent at probability 0.0 (one of them the
  segment itself). All-zero candidates now share the mass evenly instead of reading as doubt.
* About a quarter of "return" judgments linked the segment to the detour it came back from, not
  to the thing returned to (the prompt said "the segment they left when the interruption began").
  The evaluator prompt now says: the last segment of what they were on before the interruption,
  never a segment of the interruption itself. Earlier links were not rewritten.
* The dashboard HTML is read once at startup: after editing it, `restart`. A restart resets which page and
  panel are open, which is why the automatic Stop hook was disabled. Typed text survives a restart: every
  text field saves to localStorage as you type (`draftKey` in the dashboard) and chats reload from the
  server. A new text field needs an id that `draftKey` handles, or a restart can lose what is in it.
  After source edits the usual way to restart is `restart --when-idle` in the background (since 2026-11-19):
  it waits for 2 minutes without keyboard or mouse input, so the dashboard is not reset under your hands.
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
* `note_to_eli` had not been tried on a real model when it was wired in (2026-11-20, at night with the
  API blocked; only `llm_selftest` with a fake client has run). `../shared/README.md` (Gotchas) asks
  for a measurement first: routine remarks should go to the note tool, and an objection or "I want to
  stop" must still go to `contact_eli`. Unmeasured for this app's models and structured output.
* In the evaluator chat a `propose_judgment` / `propose_rule` call made in the same turn as a note is
  not applied from that turn: the model is told it was not run and has to make it again in the
  request that follows. If it does not, the reply has no proposal.
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

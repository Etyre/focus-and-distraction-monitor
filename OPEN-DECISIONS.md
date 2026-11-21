# Open decisions

Questions and design choices turned up by the bug hunts of 2026-11-20, for Eli to take one at a
time. Each says what the app does now, what was measured, and the choices. Numbers are from a
copy of the database covering 2026-08-25 to 2026-11-19 (87 days) unless stated. When one is
decided, move it to the bottom with the date and the answer, and put the rule in the Rules doc
or the README where it belongs.

## Rules about spans

1. **Does a "next task" shift join any running span, or only a task block?**
   Now: it joins whatever span is running if a Toggl entry is running. On 11-19 the longest span
   (84 min) is 22 min of creative work, 4 min on the daily page, then the Anki review under a new
   Toggl entry, counted as one focused-task span.
   Choices: (a) only extend a span that is already a task block: 11-19 becomes a 22-min creative
   span and a 58-min task span; (b) keep as it is.

2. **Does a short pause in the middle of a diversion count toward the 5 minutes that end a span?**
   Now (new today): no. Time away and pauses are two clocks (5 min away, 10 min idle). Three
   spans survive a diversion of 8 to 10 wall-clock minutes of which at most 4.4 were away
   (09-07 23:03, 10-07 16:12, 10-30 19:35).
   Choices: (a) keep; (b) count the pause, ending those spans.

3. **Should a pause restart the "10 unbroken focused minutes" count?**
   Now (new today): yes. A short screen lock or idle-labelled time resets the opening block and
   the unbroken-10 count, like other unfocused time on the thread. Stricter than before.
   Choices: (a) keep; (b) a pause neither counts nor resets.

4. **Is the opening-block rule too strict about very short diversions?**
   Now: any diversion resets the 10 unbroken minutes. "No continuous block of 10 minutes" is the
   largest disqualifier: 86 spans, 1,290 focus min in 88 days (25 spans, 398 min in the last 15
   days). Many miss by under a minute. 7 spans (115 min) would qualify if diversions of 15 s or
   less did not reset the block.
   Choices: (a) keep; (b) diversions of 15 s or less do not reset it.

5. **How should a mixed span be classed?**
   Now: the largest content share decides the whole span. 11-20 03:05-03:30 (task 49%, creative
   42%, no Toggl) is disqualified whole as task without Toggl. Two "meeting" spans have under 50%
   meeting content (09-19 16:35, 45%; 09-24 06:52, 28%).
   Choices: (a) keep; (b) require a majority and treat the rest by its own rule; (c) split.

6. **Chat under a running Toggl entry: focused-task time or never focus?**
   Now: it forms focused-task spans. 9 qualifying spans (about 157 focus min) are mostly Signal or
   Slack, e.g. 10-11 19:20-20:10 (50 min, 34 of it Signal). The Rules doc says both "chatting in
   Signal never counts" and "correspondence under a running Toggl entry is part of the task span".
   Also new today: "under an entry" means an entry ran for at least half the segment.

7. **Reading in Google Docs or Roam always passes the note-taking test.**
   Now: the surface being read is itself a notes surface, so the app cannot tell commenting from
   reading there. 17 qualifying reading spans, 482 min (2 spans, 25 min in the last 15 days).
   Choices: (a) keep; (b) require evidence of typing; (c) ask you about such spans.

8. **Reading inside a live discussion, without notes, is dropped entirely.**
   Now: it is neither a reading span nor meeting time. 11-10 15:35 (18 min) and 21:24 (17 min).
   Choices: (a) keep; (b) count it as meeting time.

9. **Meeting-content segments inside an independent span count as independent focus.**
   29 min in all, none in the last 15 days. Choices: (a) keep; (b) move those minutes to meeting.

10. **Longest focus when a span crosses 4am.**
    Now: each day reports its own piece. 09-24: the span was 77.9 min; the days report 41.3 and
    36.6. Choices: (a) keep; (b) the day it started reports the whole length as its longest.

## Phone

11. **Phone pickups under 30 seconds are never asked about and count in full.**
    A 10-second Signal check at 16:37 on 11-17 ("probably about the meeting... low confidence")
    voids a 12-minute span. Reading those pickups as on-task: 11-17 focus 200 to 213, interruptions
    12 to 10, self-distraction events 35 to 25.
    Choices: (a) keep; (b) ask about the ones that end or break a span; (c) ignore pickups under
    some length.

12. **"Distracted" time: from the drift onward, or the whole stretch?**
    Now: on the Mac a whole stretch away is "distracted" if it contains any self-distraction
    shift, while the timeline colours only the time after the shift (11-18: 17 min differ one way,
    7 the other). For phone-only stretches, per your ruling today, it is distracted only if it
    started with one. The two rules differ.
    Choices: (a) one rule everywhere: distracted from the drift onward; (b) one rule: the whole
    stretch if it started with a drift; (c) keep both.

13. **On-track phone time has no time limit.**
    Now: phone time judged "part of the work" during Mac idle attaches to the last active Mac
    segment, however long ago. On 11-18, workout logging at 21:10 attached to a segment that ended
    about 19:10. No focus number moved (the base was unfocused), but it could.
    Choices: (a) limit it to some minutes after the Mac segment; (b) keep.

14. **A little phone use turns a whole idle period into active time.**
    Now: if phone use leaves less than 10 uncovered idle minutes, the whole idle segment becomes
    active. Two 9.9-minute idle segments on 11-17 became 17 active (interrupted) minutes on 1.2
    minutes of phone use each. Written as intended, but a cliff.

15. **Phone data starts 11-16, so trends step there.**
    83, 142 and 54 minutes are carved out as phone time away on 11-17 to 11-19; earlier days have
    none. Nothing to fix; a caveat for any before/after comparison, unless older phone data can be
    judged.

## Review questions

16. **Should doubts about self-distraction get review questions?**
    Now: the question writer was told intent "is never asked", although self-distraction events are
    a headline number and each is weighed at 10 minutes of stakes. From today an option can carry
    "on purpose" or "a drift", but nothing makes the app ask.

17. **Open questions written before today keep their old options.**
    Answering "it was a deliberate errand" on q2942, q2929 and similar still leaves the
    self-distraction counted. Choices: (a) regenerate the open questions; (b) leave them.

18. **Questions about old days after the history re-judgment.**
    Every re-judged segment the evaluator is unsure about becomes a question, with no age limit
    (you rejected a cap on 11-19). The same period already holds 1,609 archived and 645 dismissed
    questions, so the re-judgment could raise thousands.
    Choices: (a) ask them all; (b) ask only for the last N days and keep older doubts as flags;
    (c) ask old ones only when the stakes are high.

## The history re-judgment

19. **Re-judge only from 2026-10-27?**
    No screenshot files exist before 2026-10-27 13:35. Re-judging earlier days would replace
    judgments made with screenshots by judgments made from titles alone. I have limited the queue
    to days that have files; say if you want the older days re-judged anyway.

20. **Coherence checks will run again over every re-judged day.**
    Each re-judged day gets new span summaries and coherence checks, which can re-judge a stretch
    again (bounded at 2 per stretch per 24 hours, not in total). Before the bound, 19 stretches
    were re-judged 3 to 11 times and a few segments flipped back and forth. Choices: (a) let it
    run; (b) one coherence pass per stretch for history.

## Model calls

21. **When the evaluator contacts you, all new judging stops for a day.**
    Now: the stretch it stopped on is held for 24 hours, and nothing after an unjudged segment is
    judged, so spans are never built across a gap. Choices: (a) keep (you answer, then it resumes);
    (b) hold only that stretch and carry on after it.

22. **Retire the per-switch vision classifier?**
    820 calls in 14 days. Its output now reaches the numbers only as an unexplained tag on log
    lines, a doubt flag (5 of 820) and the label for segments the evaluator never judged (7 of
    820). Choices: (a) retire it; (b) keep and rebuild it on the evaluator's inputs.

23. **Delete the switch chat and the thread resolver?**
    Both are dead code on the pre-redesign scheme (no caller, no stored use since August).

26. **How much of the subscription should the app's judging use?**
    Giving the evaluator everything made each call far larger: about 200,000 tokens (all 325
    rulings, up to 60 screenshots) against a few thousand before, roughly $0.50 to $1.30 a call at
    API prices instead of $0.02 to $0.13. One real call ran fine on the subscription (27 s). At 80
    to 340 evaluator calls a day this draws much more of the weekly Opus allowance your own Claude
    sessions share, and when the limit is hit the app falls back to the API key (real money) for
    everything except the history re-judgment.
    Choices: (a) keep it all; (b) keep the screenshots, show only the relevant rulings in full and
    the rest as one line each; (c) a lower image budget.

27. **The history re-judgment is much larger than when you said "all of it".**
    About 24 days (from 10-27) at roughly 8 segments a call is on the order of 1,000 to 1,500
    calls of that size. It runs only on the subscription, one call a minute at most, and can be
    paused. Decide the pace, or whether to do the most recent days first and look at the result.

## Display

24. **Show pauses on the Day view.**
    A short pause now counts inside a span. The timeline cannot yet tell it from other time.
    Suggestion: draw it hatched.

25. **Which time zone should old days be shown in?**
    Days are now cut at 4am where you were at the time, but the dashboard prints clock times in the
    browser's current zone, so a September day reads 07:00 to 07:00. Choices: (a) show each day in
    the zone it was lived in; (b) keep the current zone.

## From the first day of live running on the new code (2026-11-21)

28. **A call with no typing: still in the call, or away?**
    Until now ten minutes without keyboard or mouse input was "away" even with camera and
    microphone on: 114 minutes of calls in two days were recorded as away, and the meeting spans
    around them cut short. Being fixed so that a call (camera on, or microphone on most of the
    time) is not away. To decide: what to do about past calls recorded as away (a list is being
    produced), and whether microphone alone should count.

29. **Phone open beside the keyboard while you type on the Mac.**
    Now: a phone app segment judged a shift carves its whole length out of the Mac record as time
    away. On 11-20, 58 minutes were counted as away from the Mac during steady Mac typing (Oura
    open 21 min with 1,198 Mac keys typed; Signal 15 min, 889 keys). The evaluator's own notes say
    "the phone was likely sitting unlocked and mostly unwatched" but it can only judge a phone
    segment whole.
    Choices: (a) keep; (b) phone time is away only for the stretch with no Mac input; (c) ask you
    when a long phone segment overlaps Mac input.

30. **Should a phone judgment be redone every time the Mac segments around it are re-judged?**
    Now: yes, and the last answer stands. The Oura segment above was judged five times in 26
    minutes: four said "lying beside the keyboard", the fifth said a shift, and the fifth stuck.
    Choices: (a) a phone judgment, once made, stays until you rule or a phone-specific re-judgment
    runs; (b) keep.

31. **Can a shift judged a self-distraction open a focus span?**
    11-21's only independent focus span (22 min, planning, "fully absorbed") starts with a Claude
    conversation whose switch the evaluator called a self-distraction (flagged unsure).
    Choices: (a) yes, what matters is the focused time that followed (now); (b) no; (c) ask.

32. **Judge new activity in batches?**
    Now: a call goes out as soon as any segment is 15 minutes old: 114 calls for 239 segments in
    5.4 active hours, 60 of them judging one segment, each with about 54 screenshots (context
    images fill the budget whatever the number of segments). $0.30 a segment at API prices, all on
    the subscription. Since the restart the app has used about $200 of API-equivalent usage in 16
    hours ($115 and $136 for the two days, against $30 to $70 before).
    Choices: (a) keep; (b) wait until about 10 minutes of settled activity has gathered (several
    times fewer calls, neighbours judged together, up to 10 minutes more delay); (c) scale the
    context screenshots to the number of segments judged; (b) and (c) together.

33. **Keep the two lower-effort "shadow" calls on every evaluator call?**
    They exist to compare efforts for the Efforts tab. 266 calls, $21 of usage in 16 hours, each
    re-sending the images. Agreement with the high-effort answer: 84% (low), 89% (medium).
    Choices: (a) keep; (b) shadow one call in ten; (c) stop, the question is answered.

34. **Is the piecewise daily audit worth its size?**
    The audit of 11-20: 28 calls, $23.53 of usage (it was $0.39 to $0.50 a day), 22 flags, of
    which the question writer could turn 3 into questions. Choices: (a) keep; (b) audit only
    segments the evaluator was least sure of; (c) one day in N.

35. **The automatic follow-up question after you correct the app is billed to the API key.**
    It replays the evaluation (about 150,000 tokens now) on the key: $9.18 for 13 follow-ups in
    33 minutes on 11-21, seven of which answered only "No question." These were the only real
    charges since the restart. Choices: (a) keep; (b) send it through the subscription;
    (c) send only the block's own lines and images; (d) only when you ask.

36. **Which phone apps get "what was it used for" questions?**
    24 are open, one per app. 11 are about apps that cannot be passive consumption (Phone, Clock,
    Settings, Camera, Toggl, Maps, your own apps); 9 are about uses three to five days old.
    Choices: (a) keep; (b) skip a list of utility apps; (c) only apps the evaluator has judged
    passive at least once; and a limit on how old a use may be.

37. **Pause minutes inside a span count as focus minutes.**
    366 of 15,815 focus minutes are pauses; in 15 spans a quarter or more is pause (09-16 14:04:
    22.6 of 52.7 min). Pauses separated by a few seconds of window activity are not added
    together (one span holds 10.3 near-continuous minutes). Related to items 2 and 3.
    Choices: (a) keep; (b) pauses stay in the span but are not counted as focus minutes;
    (c) add up pauses that are seconds apart.

38. **Time the evaluator labels "idle" for ten minutes ends a span even when the Mac was not idle
    that long.** 18 away stretches exist only because of the label; in 12 the recorded time
    without input is under ten minutes (13.7 min labelled idle, 9.0 without input).
    Choices: (a) keep; (b) the label counts as away only for the time without input.

39. **Segments older than 31 days that were never judged.**
    2,588 segments from 08-25 to 10-17 have no judgment (the backfill covered 31 days). No
    screenshots exist for them. Their days' numbers come from the old per-switch models.
    Choices: (a) leave; (b) judge them from titles alone and mark them low-confidence.

## The review page (from the dashboard test, 2026-11-21)

40. **Does a note follow a changed answer?**
    Now: changing your answer carries the earlier note onto the new ruling. A note written with
    "I can't tell" became the precedent note when an option was then chosen.
    Choices: (a) keep; (b) a changed answer starts with an empty note, the old one offered back.

41. **"Clear my answer" deletes the note with it.** No draft is kept.
    Choices: (a) keep; (b) put the note back in the box as a draft.

42. **A block with an open question and any ruling shows no review dot.**
    With two questions on one block, answering the first hides the dot while the second is open.
    Choices: (a) the dot stays while any question on the block is open; (b) keep.

43. **The Day timeline opens on 9am to midnight.**
    Today's only focus span (05:10) and its dots were outside it until zooming out.
    Choices: (a) open on the stretch that has activity; (b) keep.

44. **Review answers are numbered 1., 2., 3. but the number keys do nothing on a question card.**
    Choices: (a) number keys choose the option; (b) drop the numbers.

## Decided

- 2026-11-21: **sub-threads.** A sub-thread is an attentional thread that buds off from another
  one: a new and distinct object of focus that is still in service of, or at least originated
  from, the first (e.g. in the middle of a task, a blocking bug in the software the task depends
  on, and attention moves to debugging it). If they are focus spans: the sub-thread is recorded as
  a separate focus span if and only if it runs longer than 10 minutes; otherwise it is absorbed
  into the parent span (possibly with a note in the spans table). Eli: "I currently think", so
  treat it as provisional. NOT YET BUILT: needs the Rules doc (his file: propose the wording) and
  the code (a "buds off from" relation on a shift, judged by the evaluator and askable in review
  questions; stats absorbing or splitting by the 10-minute line). Points to settle when building:
  - It overlaps the rule accepted on 2026-11-20 ("if a tool I'm using breaks ... the fixing is
    part of that activity's thread, not a separate task"). That rule treats the fix as the same
    thread; the sub-thread idea treats it as a new thread that is absorbed when short. Reconcile.
  - Today a diversion of 5 minutes or more ends a span. A sub-thread of 5 to 10 minutes would be
    absorbed instead: confirm that it overrides the 5-minute rule.
  - When a sub-thread runs past 10 minutes: does the parent span end where it budded, and does
    returning to the parent start a new span or continue the old one?
  - Does an absorbed sub-thread's time count as the parent's focus time, or as a contained
    diversion that does not break it?

- 2026-11-20: **a task span and Toggl.** One task is one attentional thread. Committed,
  consistent focused engagement on a single thread can be a focus span without a Toggl entry.
  Joining several task threads into one span needs a signal that the switching is held in an
  overall focused intention: a running Toggl entry, working from a todo list he comes back to
  after every item, or working from his task feed. (Raised on the 20:29-22:00 span of 11-20, the
  audio-player agent session, disqualified as "task activity without a Toggl entry".) NOT YET
  IMPLEMENTED: needs the stats gate changed (single thread = one thread root among the span's
  focus segments), a way to see the todo list / task feed (which apps or pages they are: ask),
  and the Rules doc line about Toggl narrowed (Eli's file; propose the wording). Also look at why
  that span is "ended by task change" at 22:00, and the 20:54 note naming the wrong block.

- 2026-11-20: phone time away from the Mac is distracted if it started with a self-distraction
  event, otherwise unfocused. (Rule in the README.)
- 2026-11-20: a span crossing 4am is qualified as a whole; each day keeps its own minutes.
- 2026-11-20: re-judge the history judged at low effort, all of it (see 19 for the limit found).

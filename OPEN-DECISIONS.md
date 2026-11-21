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

28. **Does a single task need a Toggl entry to be a focus span?** (Eli, 2026-11-20, in the span
    chat about 20:29-22:00, the audio-player agent session: "this should actually count as a focus
    span ... Committed, consistent focused engagement on a *single* task can still count as a
    focus span, but if switching tasks rapidly there needs to be some signal that the switching
    is held in an overall focused intention.")
    Now: any task-content span without a Toggl entry is disqualified ("task activity without a
    Toggl entry"). Proposed rule: one task thread needs no Toggl entry; joining several task
    threads into one span needs a signal of an overall intention (a running Toggl entry; anything
    else?). Open points the chat raised: how fine-grained a "task" is (each song, or building the
    playlist?), whether Toggl is the only signal, and rewording the Rules doc line. Also: why that
    span is "ended by task change" at 22:00, and a note at 20:54 that names the wrong block before.

## Display

24. **Show pauses on the Day view.**
    A short pause now counts inside a span. The timeline cannot yet tell it from other time.
    Suggestion: draw it hatched.

25. **Which time zone should old days be shown in?**
    Days are now cut at 4am where you were at the time, but the dashboard prints clock times in the
    browser's current zone, so a September day reads 07:00 to 07:00. Choices: (a) show each day in
    the zone it was lived in; (b) keep the current zone.

## Decided

- 2026-11-20: phone time away from the Mac is distracted if it started with a self-distraction
  event, otherwise unfocused. (Rule in the README.)
- 2026-11-20: a span crossing 4am is qualified as a whole; each day keeps its own minutes.
- 2026-11-20: re-judge the history judged at low effort, all of it (see 19 for the limit found).

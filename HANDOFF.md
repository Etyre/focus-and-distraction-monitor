# Handoff: paused 2026-11-20 23:15, resume 2026-11-21 at 15:00 with Fable

Eli paused the work because the Fable usage limit was reached (it resets 2026-11-21 15:00 Eastern).
Resume with `/model` set to Fable 5.2. Delete this file when the work below is finished.

## Where things stand

- `main` is clean, tested (`./test.sh` PASS, span self-test 38/38, signals self-test) and ready to
  restart. Everything from the 2026-11-20 bug hunt and audit is committed there; the README's
  Gotchas ("Audit, 2026-11-20" and "Bug hunt, 2026-11-20") describe it.
- The only uncommitted file is the Rules doc, which holds Eli's own edits. Do not commit or
  overwrite it.
- First restart onto this code runs one-off migrations: `seg_propagated` (records propagated
  corrections, re-queues 9 recent propagations), `repair_outvoted` (492 old rows get the
  evaluator's own switch back; the old values go to `seg_evals_outvoted`), the late-captures move,
  and the backfill of about 124 never-judged segments. All four were tested on database copies.
- The history re-judgment is built but NOT started (`python -m focus_monitor rejudge-history
  --start`). Wait for Eli's answers to items 19, 26 and 27 in `OPEN-DECISIONS.md` first.

## Unfinished: richer inputs for the other model calls

The audit found that propagation, the question writer, the day chat, the evaluator chat, rule
drafts and span summaries are given far less than exists (caps, truncation, missing screenshots,
old-scheme wording, effort "low" written into the question writer, spot check and chats). The
segment evaluator itself was already reworked (commit b93acca).

- An agent was part way through this when the limit hit. Its untested work is on the branch
  `wip/other-call-sites` (propagation.py, chat.py, summaries.py, new chat_tools.py and
  prompt_parts.py). It had not yet done the question writer, the evaluator chat, rule drafts or
  the effort settings.
- Its full brief: `~/Library/Application Support/FocusMonitor/audit-2026-11-20/brief-other-calls-rework.txt`.
- The audit it implements: `.../audit-2026-11-20/7-other-calls-information.txt`.
- To resume: start an agent with that brief, telling it to begin from the branch (check out
  `wip/other-call-sites` in a worktree or merge it into a working branch, check what is done,
  finish the rest), then test, run `./test.sh`, merge into `main`, commit, and queue the restart.

## All reports from 2026-11-20

`~/Library/Application Support/FocusMonitor/audit-2026-11-20/` (private: outside the repo on
purpose, the reports quote ids, apps and some window titles): seven audits (numbered 1 to 7) and
the reports of the three agents that fixed stats, the collector and the evaluator. Read the
"not done" and "needed in other files" parts of the three fix reports: a few small follow-ups
are listed there (for example `pause` is not yet exported to the Day view, and `seg_models`
reads `toggl_id`, whose meaning changed).

## Decided after the pause, not yet built

- The single-task rule for task spans (see "Decided" in `OPEN-DECISIONS.md`, 2026-11-20). Ask
  which apps or pages are his todo list and his task feed before building the detection.
  When the rule is talked through and built, it must update the 2026-11-20 20:29-22:00 span
  (the audio-player agent session, now disqualified as "task activity without a Toggl entry"):
  check that the app's own call on it becomes a focus span under the new rule. Eli may also have
  recorded a verdict on it by then (span chat or the card): look in `span_reviews` first.

## Decisions waiting for Eli

`OPEN-DECISIONS.md`, one at a time. Item 26 (how much of the subscription the larger evaluator
calls should use) takes effect as soon as the new code runs.

"""Focus & distraction monitor for macOS."""

# Internal keys (also DB column suffixes). "task_change" is displayed as "focus start".
LABELS = ["continuation", "interruption", "distraction", "task_change"]
LABEL_NAMES = {"continuation": "continuation", "interruption": "interruption",
               "distraction": "distraction", "task_change": "focus start"}
LABEL_HELP = {
    "continuation": "Same activity/state continues: reading<->notes, editor<->terminal, tab-to-tab on the "
    "same thread, a related side-question - and also staying distracted (feed -> feed) or staying in a "
    "messaging detour (Slack -> Signal). Nothing changes.",
    "interruption": "Leaving the current activity for a quick unrelated detour: a stray question, reactive "
    "email/message checking, a notification. Not stimulation-seeking.",
    "distraction": "Leaving the current activity to seek stimulation without an objective: feeds, aimless "
    "shopping, dating apps, a blocked URL.",
    "task_change": "Focus start: focused work begins or resumes here - returning to the task after a detour, "
    "deliberately starting a different task, or genuinely settling into absorbed reading/note-taking.",
}

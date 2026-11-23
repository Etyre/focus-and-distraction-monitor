"""Chunked segment evaluator: the model applies judgment over whole stretches of activity.

Every ~15 minutes (lazily, so after-context exists), one Claude call evaluates a chunk of
segments together, with surrounding context and sampled screenshots. For each segment it
judges:
  1. CONTENT - what kind of activity the segment is:
     creative | task | correspondence | reading | planning | other | passive
     ("task" describes the activity's nature regardless of Toggl - Toggl is span-level
     evidence of deliberate focus, not evidence of task-nature.)
  2. SWITCH - what the switch INTO the segment was, attention-wise:
     continuation (no shift of the object of attention) | shift (attention moved to
     something new; stored as 'interruption', see seg_models.shown_switch) |
     return (coming back to the object attention was on before a shift)
  3. FOCUSED - was attention focused during the segment, or unfocused (quality of attention,
     independent of which thread the segment belongs to).
  4. For shifts, two yes/no judgments: SELF_DISTRACTION (sought out stimulation without a
     clear intent) and NEXT_TASK (the next task under the same task-processing intention;
     the app joins those threads into one focused task block).
The older per-shift KIND (self_distraction | focus_start | detour) is no longer asked of the
model (Rules doc redesign, 2026-11-19); it is derived from these fields and still stored, so
history, the local models and the person's past corrections keep working.

The user audits and edits these (seg_reviews); edits are authoritative for state derivation
and are fed back to the evaluator as few-shot corrections, so it learns over time."""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time

from pydantic import BaseModel, Field

from . import db, llm
from .config import (PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, Config,
                     load_config)

log = logging.getLogger("evaluator")

from .seg_models import (CONTENTS, HEADS, KINDS, SHOWN_SWITCHES, SWITCHES,  # single source for the label space
                         shown_switch, stored_switch)

LAZY_S = 15 * 60          # evaluate only segments at least this old (after-context exists)
CHUNK_S = 20 * 60         # evaluate about this much activity per call
BATCH_S = 10 * 60         # ...and wait until this much is ready to judge, unless activity has stopped (run_pending)
CONTACT_HOLD_S = 24 * 3600   # after the evaluator asks for Eli, how long its chunk is left alone
CTX_S = 10 * 60           # how far around a chunk phone pickups and their stored judgments are looked up
# Context shown to the model (2026-11-20; it was 10 minutes each side). Of 513 segments judged a
# shift in 30 days, 178 were on a window last in front earlier that day but more than 10 minutes
# back, so the model could not see what it was coming back to; and in 199 of 235 first-pass calls
# activity already recorded after the window was withheld (median 5 minutes), though "then back
# to the task" and the self-distraction test both turn on what follows.
BEFORE_S = 45 * 60        # log shown before the chunk
ACTIVE_BEFORE_S = 10 * 60 # ... reaching further back, past an away stretch, until this much active time is in view
BEFORE_MAX_S = 18 * 3600  # ... but never further than this
AFTER_FIRST_MAX_S = 2 * 3600   # first pass: everything recorded up to the call, at most this much (a backlog can be days)
AFTER_REJUDGE_S = 30 * 60      # re-judgments: this much after the chunk
RETURN_MAX_S = 10 * 60    # Rules doc: a return is to something left within the past 10 minutes
HOLD_MAX_S = 60 * 60      # Rules doc: the model may hold a segment back for up to an hour
HOLD_RETRY_S = 15 * 60    # after a hold, ask again this much later (more activity will be logged)
# Images per call (2026-11-20; a fixed 8 before, whatever the number of segments: 29% of all
# targets were judged with no screenshot though one existed). Every target gets its captures
# first; a call never carries more targets than this budget can show, it judges fewer segments
# instead (evaluate_chunk sets last_end, and the callers go on from there). 60: a capture is
# 1280 px wide, about 1,400 tokens, so 60 are about 85k tokens and 6 MB of base64, inside the
# API's limits as far as known (100 images and 32 MB per request; above 20 images each must be
# at most 2000 px on a side, which the 1280 px captures are).
IMAGE_BUDGET = 60
PAGE_LOADS_MAX = 150      # page loads listed under one segment (the rest are counted, not listed)
CONTEXT_IMAGES_MIN = 20   # of the budget, kept for context segments when the targets could fill it
SEG_MOMENTS_MAX = 12      # capture moments shown for one target (a moment is up to two images: both displays)
# Output room (2026-11-20; 8,000 before, and one answer reached 8,128 tokens and was cut off).
# Thinking counts against it too. 20,000 is the most the SDK takes without streaming (it refuses
# a non-streaming request whose max_tokens implies more than 10 minutes: above about 21,300).
MAX_TOKENS = 20000


class AnteCand(BaseModel):
    segment_id: int
    p: float = Field(description="Probability this is the right attachment GIVEN the segment continues an existing thread (the candidates' probabilities sum to ~1).")


class SegJudgement(BaseModel):
    segment_id: int
    content: str = Field(description="One of: creative, task, correspondence, reading, planning, other, passive, meeting, idle, transition.")
    focused: bool = Field(description="Quality of attention during THIS segment: true = focused (doing one thing with intent, however small), false = unfocused (half-hearted, scattered, drifting, or consuming). Independent of which thread the segment belongs to and of whether it is a diversion from something else.")
    switch: str = Field(description="The switch INTO this segment: continuation, shift, or return.")
    antecedents: list[AnteCand] = Field(default_factory=list, description="When switch is continuation or return: up to 3 RANKED candidate antecedents - each the id of the nearest earlier segment of a thread this might continue. Usually one candidate carrying ~all the mass; add a runner-up (and rarely a third) only when genuinely torn between attachments. Empty when switch is shift (a new thread).")
    self_distraction: bool = Field(default=False, description="When switch is 'shift', or a 'return' to something that is not their work: true when the person sought out stimulation or distraction WITHOUT a clear intent. False otherwise, and always false for a continuation.")
    next_task: bool = Field(default=False, description="Only when switch is 'shift': true when this is the person moving on to the NEXT TASK under the same overall intention of processing tasks. False for every other shift, and always false for continuation/return.")
    grew_out_of: int | None = Field(default=None, description="Only when switch is 'shift' and the new object is a SUB-THREAD (section 4): the id of the nearest earlier segment of the thread it grew out of. null for every other shift, and always null for continuation/return.")
    uncertain: bool = Field(default=False, description="True only when you are genuinely unsure of the content, focus or switch judgment and a human should check it. Use sparingly - these are flagged for the person to review.")
    hybrid: bool = Field(default=False, description="True when the activity (reading, task, creative, planning...) was being done INSIDE a live discussion with someone - e.g. reading a bill together on a call while taking notes. Keep content = the activity; hybrid marks the live-discussion dimension. False for solo work and for plain meetings (content='meeting').")
    note: str = Field(description="One short sentence of reasoning.")


class PhoneJudgement(BaseModel):
    phone_id: int = Field(description="The number after 'id=P' on the EVALUATE PHONE line.")
    content: str = Field(description="As for a computer segment: creative, task, correspondence, reading, planning, other, passive, meeting, or transition.")
    focused: bool = Field(description="As for a computer segment: quality of attention during this phone segment.")
    switch: str = Field(description="The switch INTO this phone segment: continuation, shift, or return - the same meanings as for a computer segment.")
    continues: str = Field(default="", description="When switch is continuation or return: the id of the log line this attaches to - 'P<number>' for an earlier phone line, or a computer segment's id. Leave empty for a continuation of the line just before it.")
    self_distraction: bool = Field(default=False, description="As for a computer segment (section 4): only on a shift, or a return to something that is not their work.")
    next_task: bool = Field(default=False, description="As for a computer segment (section 4): only on a shift.")
    note: str = Field(description="One short sentence of reasoning.")


class ChunkJudgement(BaseModel):
    segments: list[SegJudgement]
    phone: list[PhoneJudgement] = Field(default_factory=list, description="One entry per line marked EVALUATE PHONE (each is one app on the phone's screen); empty when there are none.")
    hold_from: int | None = Field(default=None, description="Only when the prompt says holding is allowed. The id of the first EVALUATE segment you would rather judge LATER, because what the person does next would change your judgment of it. Judge every segment before it; leave it and every segment after it out of `segments`. null = judge everything now (the normal case).")


def _thread_last_end(conn, ante_id: int, before: float, seg_ends: dict) -> float:
    """When the thread that segment `ante_id` belongs to was last in front before `before`: the
    latest end among the segments of the past two hours whose pointers lead to the same root."""
    rows = conn.execute("""SELECT g.id, COALESCE(g.end, g.start) AS e, COALESCE(r.antecedent_id, v.antecedent_id) AS a
                           FROM segments g LEFT JOIN seg_evals v ON v.segment_id = g.id
                           LEFT JOIN seg_reviews r ON r.segment_id = g.id
                           WHERE g.start >= ? AND g.start < ? AND g.idle = 0""", (before - 7200, before)).fetchall()
    ante = {r["id"]: r["a"] for r in rows}

    def root(i):
        for _ in range(40):
            nxt = ante.get(i)
            if nxt is None or nxt == i:
                return i
            i = nxt
        return i
    want = root(ante_id)
    last = seg_ends.get(ante_id, 0.0)
    for r in rows:
        if r["id"] != ante_id and root(r["id"]) == want:
            last = max(last, min(r["e"], before))
    return last


def kind_of(switch: str | None, focused, self_distraction) -> str | None:
    """The older per-shift kind, derived from the fields the model now gives (see the module
    docstring). Stored alongside them so everything written for the old label keeps working."""
    if switch != "interruption":
        return None
    if self_distraction:
        return "self_distraction"
    return "focus_start" if focused else "detour"


SYSTEM = """You are the segment evaluator for a personal attention monitor. You receive a stretch \
of one person's computer activity - an ordered log of segments (one active window each), with \
the activity before and after it as context, their Toggl time entries for the day, a summary of \
the day so far, and screenshots - and you judge every segment marked EVALUATE. How to read the \
log, the screenshots and the person's own past rulings is explained in sections 6 to 9. For each \
segment marked EVALUATE output:

1. content - what the activity IS:
   - creative: making something (writing an essay/doc, coding, video editing).
   - task: executing discrete tasks (forms, errands, admin, logistics, purchases). Judge the \
activity's nature only - whether a Toggl timer is running is NOT part of this judgment.
   - correspondence: communicating with people in writing - chatting on Signal/Messages/\
WhatsApp/Slack, reading and answering email, replying to comments or DMs. The exchange itself \
is the activity. (A live voice/video call is 'meeting'; a message that is really a piece of \
writing the person is absorbed in composing can be 'creative'.) Opening a chat only to fetch a \
link, file, or reference on the way to a task is NOT correspondence - no exchange is happening - \
it is 'transition' (or 'task'), whatever the app. Correspondence is interactive, \
so on its own it is never part of a focus span - but label the segment's nature regardless: \
email processed between other tasks under a running Toggl entry is still 'correspondence' \
content, and the span logic decides that it belongs to the task block.
   - reading: reading posts/papers/books. (Whether notes are being taken matters for spans, not \
for this label - reading without notes is still 'reading' content, unless it is really passive \
browsing/consumption.)
   - planning: metacognition - journaling, prioritizing, reviewing goals, deciding what to work on.
   - other: real engaged activity fitting none of the above.
   - passive: passive consumption - watching videos, scrolling feeds, browsing without an \
objective, reading comics/entertainment. When in doubt between 'other' and 'passive', lean passive: \
this person's calibration is that the system has been far too optimistic about focus. That lean \
applies to AMBIGUOUS ENGAGEMENT only - a segment long enough that consumption could have happened. \
It does not apply to segments too short for any engagement (see 'transition' below).
     EXCEPTION - a video (or talk, or screencast) watched IN SERVICE OF THE CURRENT TASK is not \
passive: a tutorial on exactly the thing they are stuck on, a recording they need for the piece \
they are writing. Both must be evident from the log: the video is clearly about the task they \
were just on, and they go back to that task afterwards (or it is still playing at the end of \
the log). Give it the content label of the task it serves, switch 'continuation' of that task, \
focused true. A video that is merely on a related topic, or that they drift on from to other \
videos, is passive as usual.
   - meeting: a live conversation - video call (Zoom/Meet/FaceTime), call notes or live transcript \
(e.g. Otter) during one. When the conversation is the ACTIVITY. If instead an activity (reading a \
document, working through a task, writing, planning) is being carried on INSIDE a live discussion - \
reading a bill with a colleague on a call while taking notes, pairing on code, going through a plan \
together - keep content = that activity and set hybrid=true. Evidence: a call/meeting window or a \
Toggl entry naming another person ("with Dave"), notes growing through input-idle stretches (listening), \
a transcript side panel. Hybrid stretches count as meeting time, not independent focus.
   - idle: the window was open but keyboard/mouse input was idle for most of the segment and \
nothing suggests engaged reading or watching - the person was likely away or disengaged. \
Log lines show "(no input for ... of it ...)" when pauses were recorded. A mostly-input-idle segment can NOT be task, \
creative, or planning (those require interaction); it can still be reading or passive if they \
were plausibly reading/watching the screen.
   - transition: rapidly passing through desktops, windows, or tabs on the WAY to a specific \
target - travel, not attention. Typically a chain of very short segments ending somewhere the \
person settles. Transitions are never attention shifts in themselves; their switch is \
'continuation' of whatever the person is heading toward. Also transition: a window that was \
already sitting open on a desktop and got cycled past - a segment of roughly 15 seconds or less \
with no scrolling or typing in it, flanked by segments of the same thread, is a pass-through \
regardless of what the window contains (a Substack post, a feed, a comic). Ten seconds is not \
consumption; do not label it passive or a shift on the strength of the window's name alone. \
Whether a screenshot of the segment is attached has no bearing on this: captures are taken when \
a window comes forward and about once a minute after that, files are sometimes no longer on \
disk, and each call says which segments have none and why. A missing image is missing evidence, \
never a sign of a pass-through; judge such a segment from its duration, its input counts and its \
neighbours.
2. focused - the QUALITY of the person's attention during this segment: focused or unfocused.
   - focused (true): doing one thing with intent, engaged with it. This holds for small things \
and for things off the main work: jumping to a shop to buy one specific item is focused, even \
though it is a diversion from the essay they were writing.
   - unfocused (false): half-hearted, scattered or drifting - bouncing between surfaces with no \
one object, poking at something without engaging, lackadaisical task churn, browsing without an \
objective. Passive consumption is always unfocused.
   This is a SEPARATE question from which thread the segment belongs to (the switch, below) and \
from whether it is a diversion from something else (the app works that out). Judge the segment \
honestly on its own evidence, with the surrounding segments as context: if the segments around \
it are all the same thread of engaged work, this one is more likely focused. Do not mark time \
unfocused only because progress looks slow - the person may be thinking; mark it unfocused when \
the evidence shows attention was not on one thing. For a 'transition' or 'idle' segment give \
the value of the thread it sits inside (the app does not use it).
3. switch - did the switch INTO this segment shift the person's OBJECT OF ATTENTION?
   - continuation: same object (window-hopping within one piece of work counts - text<->notes, \
editor<->terminal, stepping through desktops to find something; also staying within one ongoing \
distraction session).
   - shift: the object of attention changed to something new. This is a judgment about this \
one switch only, and applies at any time of day, whether or not a focus span is running. (The \
person's rules use "attention-interruption event" for something narrower: a shift that happens \
during a focus span. The app works out which shifts are those; you only say whether attention \
moved.) Finishing one discrete task and starting the next (email -> taxes -> errands) IS a \
shift - attention moved to a new object - even inside a block of task processing; see next_task.
When they reach the new object by passing through other windows, those pass-through segments \
are transitions (switch 'continuation' of what they were on); 'shift' goes on the segment where \
they ARRIVE at the new object.
A BRIEF look at a todo list or task feed to choose the next task, followed by doing that \
task, is part of the chosen task's thread: the shift (with next_task, when it is one) goes on \
the todo-list segment, and the task that follows is a continuation of it. The look itself is \
neither a diversion nor a separate thread, and takes the task's content label.
   - return: attention moved back to an object they were on before an earlier shift, within \
about the last 10 minutes. Coming back to something after longer than that is a shift. When \
they get back by passing through other windows, those pass-through segments are transitions \
(switch 'continuation'); 'return' goes on the segment where they ARRIVE back on the old object.
   For continuation and return, ALSO name the antecedent(s): ranked candidates for the nearest \
EARLIER segment (visible in the log; "(context id=N)" lines count) belonging to the SAME thread - the thing this segment \
continues. For a continuation that is normally the immediately preceding non-transition segment; \
for a return, the last segment of what they were on BEFORE the shift away - the thing they \
are coming back to - and never a segment of the time away itself. Thread identity follows the \
chain of antecedents, so the nearest member is enough - never reach further back than needed. \
Probabilities are CONDITIONAL on the segment continuing an existing thread (whether it does at \
all is your switch judgment) and sum to ~1 across candidates. Usually one candidate carries \
essentially all the mass; when you are genuinely torn between two attachments, say so with a \
runner-up (and rarely a third). This pointer is the primary statement of thread structure; \
make it consistent with your switch judgment.
4. Two yes/no judgments and one pointer, for switch='shift' (false / null otherwise; the one exception is noted):
   - self_distraction: the person sought out stimulation or distraction WITHOUT A CLEAR INTENT \
- absent-mindedly opening a feed, a blocked site, a dating or shopping app with no objective. \
It is about WHY they moved, not what is on the screen. NOT a self-distraction: a planned break \
(they stopped the timer and put on a video they meant to watch), watching something someone \
sent them, a deliberate errand or purchase, deliberately handling an email - even when the \
content is passive. Use your own focused/unfocused judgment of the segment as an INPUT here: \
arriving somewhere and doing one thing with intent (focused) is evidence the move was \
deliberate; arriving and drifting, skimming or doing nothing (unfocused) is evidence it was \
stimulation-seeking. It is evidence, not the answer - a planned video break is unfocused and \
deliberate. EXCEPTION - going BACK to a distraction: when they leave what they \
were doing and go again to a feed or other distraction they visited a few minutes ago, the \
switch is a 'return' (to that earlier visit), and self_distraction is judged afresh for this \
visit by the same test: was THIS move stimulation-seeking without a clear intent? Each such \
visit counts separately. A return to their work is never a self-distraction.
   - next_task: this shift is the person moving on to the NEXT TASK under the same overall \
intention of processing tasks - working from a todo list or tracker and returning to it between \
tasks, executing efficiently, one task after another. The app joins tasks linked this way into \
one focused task block, and it applies the rule that a block needs a running Toggl entry; you \
judge the intention. False when the shift is not the next task: a glance at a feed, a stray \
question, drifting to something else. How long a task takes does not matter: one task that turns \
out fiddly and takes 25 slower minutes is still the next task, and so is the one after it.
   - grew_out_of (SUB-THREADS): sometimes a new object of attention BUDS OFF the thread they \
were on. Doing a task, they hit a bug in the tool the task depends on, and attention moves to \
debugging the tool; planning their day with an assistant, they start reworking how the assistant \
itself should behave. That is a real shift - a new and distinct object - and it is also in \
service of, or grew directly out of, what they were doing. Mark it as a shift and give in \
grew_out_of the id of the nearest earlier segment of the thread it grew out of (normally the \
segment just before). The app decides what to do with it: a sub-thread under 10 minutes is \
absorbed into its parent, a longer one stands as a thread of its own. So do NOT stretch \
'continuation' to cover such a move, and do not withhold 'shift' because the move was in service \
of the task; say shift, and say what it grew out of. (The rules document's line about a tool \
that breaks being "part of that activity's thread" is the same case seen from the other side: \
label it this way, and the short ones end up in the thread as that line wants.) null when the new \
object did not grow out of what they were on: an unrelated task, an incoming message, a \
distraction. A sub-thread is never a self-distraction. Coming back from the sub-thread to the \
parent is an ordinary return (or a shift after more than 10 minutes), pointing at the parent.

5. PHONE SEGMENTS. The log may contain PHONE lines. Each is one app on the screen of the \
person's iPhone for a stretch of time: a segment, like a computer segment, and judged BY THE \
SAME RULES (sections 1-4) - content, focused, switch, and on a shift self_distraction and \
next_task. For each line marked EVALUATE PHONE add an entry to `phone`. What is different is \
only what you can see:
   - For most PHONE lines you know the app and the timing, nothing else: not what was in the \
app. Lean on the app, the duration, how it was opened, what was happening on the computer, and \
the person's notes about their apps in the rules below. The screenshots of the Mac's screen \
attached for a pickup are from just before and just after it.
   - Some PHONE lines come with screenshots of the iPhone's own screen, labelled "the iPhone's \
screen during PHONE id=P..."; the line says how many ("[iPhone screenshots attached: N]"). They \
exist only while the phone was on a cable to the Mac or on its Wi-Fi, are taken at most every \
20 seconds and only when the screen changed, and are withheld for the apps the person marked \
private and for anything between two apps. So most lines have none, a short segment can fall between two of them, and a missing \
one says nothing about what the segment was. When a line has them, judge it by what they show, \
as you would a computer segment.
   - "[Mac input during this: ...]" on a PHONE line says whether the Mac's keyboard, mouse or \
trackpad were used while the phone app was on screen, and how long before and after the nearest \
Mac input was. It is counted in 10-second steps and exists only from 2026-11-20. Mac input all \
through a phone segment suggests the phone was lying open beside the keyboard, not in hand; none, \
with a gap on both sides, suggests they were on the phone.
   - "(phone unlocked)" marks the first line of a pickup. Unlocking puts the phone on whatever \
app was open when it was last locked, so a first line of a few seconds with no "opened from" is \
usually them passing through that app on the way to another, or looking and locking again: by \
the usual standard that is a 'transition' (a continuation), not a shift - unless the pattern \
around it shows they went there for it.
   - "(first <app> open today)" marks an app's first use of their day.
   - switch reads the same way as for a computer segment, with the phone lines and the \
computer lines as one sequence. 'continuation' on the first line of a pickup means the phone \
use is part of what they were doing on the computer (a confirmation code for the thing on \
screen, looking something up for the task, a call or message exchange that IS the current \
activity); on a later line it continues the phone line before it. Starting or stopping their \
time tracker (Toggl) or jotting a note is never a diversion, like flipping to their daily page. \
'shift': attention moved to something new - their rule is that looking at the phone is a \
diversion unless what they look at is clearly related to the current thread. 'return': back \
to something from before the last shift. For a continuation or return that does not attach to \
the line just before it, name the line in `continues`. When the computer shows nothing going on \
around the pickup (idle, away), there is usually nothing on the computer to continue. The \
exception: phone use that starts within minutes of the work and plainly belongs to it (a call \
that is the task) is a continuation of the last computer segment before the idle time - name \
that segment in `continues` - and it counts as part of that work even though the computer sat idle.
   - self_distraction, as in section 4. Their words: "Opening my phone and looking at a random \
app without a clear task, is [a self-distraction event]." A deliberate errand, a use that \
serves the current task, answering a message that just arrived ("opened from a notification" \
points that way), or logging something in a tracker is not.
   - Also their words: "just glancing at my lock screen on my phone is an interruption, but \
isn't a self-distraction event." A PHONE line that says they glanced at the lock screen is \
already settled that way by the app and is never marked EVALUATE. The phone log records a \
lock-screen look only rarely, so most of them never appear.
   Phone lines never change how you label the computer segments themselves: the app cuts phone \
time that was a shift out of the computer segment it falls in, and treats the rest of that \
segment as a return. So do not turn a computer segment into a shift or a return because a PHONE \
line sits inside it. PHONE lines that already have a judgment carry the same tags as computer \
lines.

6. SIGNALS IN SQUARE BRACKETS. A computer line may carry notes in [brackets]: things the \
Mac recorded beside the capture. They are evidence, not verdicts, and older lines have none \
(clicks and input counts exist from 2026-11-20, page loads from 2026-11-17).
   - "arrived by": what the person did just before this app came forward. "a click on a \
notification banner" means something arrived and they followed it: an outside prompt, not \
stimulation-seeking (whether it was a shift still depends on what it was about). "a click on the \
Dock" or "the keyboard, no click" means they went there on their own initiative. "a click inside \
<the previous app>" usually means a link or button there opened this one, so it likely serves \
what they were doing. This is inferred from a click's position and timing and is sometimes wrong.
   - "a notification banner appeared N s before": a banner was on screen shortly before the \
switch. The Mac does not say which app sent it, so it may or may not be why they switched.
   - "page reached by": how the browser got to this page. "typed the address", "opened from a \
bookmark" and "searched from the address bar" are deliberate navigation from nowhere - for a feed \
or video site in the middle of work, the mark of a self-started visit. "a link on <site>" or "a \
new tab opened from <site>" means it grew out of that page. "link opened from another app" means \
it was clicked in a message, document or other app. "no new page load" means they switched to a \
tab they already had open, or moved around inside a page.
   - "input": keys, clicks, scroll events and pointer-move events per minute, averaged over the \
segment. Steady typing is writing or chatting; scrolling with few keys is reading or browsing; \
almost nothing is watching, listening, thinking or being away from the keyboard. Only counts are \
recorded, never what was typed. They are counted in 10-second steps, so for a segment shorter \
than about 20 seconds they are partly the neighbouring segments' input, and the line says so.
   - Under a segment longer than two minutes, "input by minute" gives the same counts for each \
clock minute as keys/clicks/scroll events/pointer moves, followed by a (audio playing), m \
(microphone on) or c (camera on) when that was so for at least half of that minute: \
"14:03 212/3/0/40 a" is 212 key presses, 3 clicks, no scrolling, 40 pointer moves, audio on. Use \
it to see when in a long segment the person was typing, reading, or doing nothing.
   - "no input" under a line lists the recorded pauses inside it, with times: stretches of a \
minute or more with no keyboard or mouse input that did not become an AWAY line (that takes ten \
minutes). Recorded for all dates.
   - "page loads during this segment" lists every page Chrome loaded while the segment was in \
front: time, how it was reached, the address, the page title, and the page it came from. The \
segment's own title and url are those of the window when the segment began; when the loads show \
other pages, the person moved on inside that window, and the loads (and the screenshots) say \
where to. No list means no load was recorded, not that nothing happened in the page.
   - "audio playing", "microphone on", "camera on": share of the segment. Microphone or camera \
on for most of a segment means a call or a recording - or, with the microphone alone and for \
short bursts, voice dictation in place of typing (low key counts then do not mean idleness). Audio playing with \
little input usually means a video or a call is actually running, not just open.

7. HOW TO READ A LOG LINE. Each computer line is, in order: its mark ("EVALUATE id=N:" for a \
segment to judge, "(context id=N)" for one shown as context); the time it began, to the second, \
in the time zone the person was in that day; how long it lasted in all (seconds under two \
minutes), also when it began before or ran on after the stretch shown; the app; "site:", \
"title:" and "url:" in full where the Mac recorded them (most apps other than browsers have no \
title). An "AWAY" line is a stretch of ten minutes or more with no keyboard or mouse input (a locked \
screen counts as no input). \
"(toggl entry covering most of it: ...)" names the Toggl entry that ran for the largest part of \
the segment, or says none ran during it; exactly when an entry started or stopped shows in the \
day's Toggl list, which has times to the second.
   - A line with NO "judged:" part has not been judged by anyone yet; nothing on it is a verdict.
   - "judged: <content> <focused|unfocused> <switch: continuation|shift|return> ..." is the \
judgment the app holds for that segment, normally your own from an earlier call. "<continues \
id=N>" is its thread pointer, "<hybrid>" marks activity inside a live discussion, "<flagged \
uncertain>" that it was flagged for review. A 'transition' or 'idle' line has no focus tag. \
Segments judged before 2026-11-19 carry an older switch tag - <switch: shift (detour)>, \
(self_distraction) or (focus_start) - from when the kind of shift was a single label; read \
those as a shift, and no switch tag on such a line as a continuation. "// note:" after it is \
the one-sentence reasoning stored with that judgment, in full. These are earlier judgments, not \
facts: weigh them as you would a colleague's notes. "earlier quick-pass judgment:" marks one \
made at low effort that has not been redone; treat it as a hint at most.
   - "RULED BY THE PERSON: ..." marks a segment the person has ruled on themselves, with the \
fields they set and their note. Those fields are FIXED ANCHORS: they are what happened. Build \
your judgment of the neighbouring segments around them (a segment they ruled the start of a new \
thread is one; a segment they ruled a continuation of id=N is in N's thread). "stands as:" on \
such a line is the judgment in force, their fields over the app's. When a ruled segment is \
itself marked EVALUATE, give the fields they set exactly as they set them and judge the rest.
   - Lines under a segment, indented, belong to it: the input and page-load lists of section \
6; "earlier today:" (when this same window - same app, title and url - was last in front before \
the log shown, for how long, and how that visit was judged); and "screenshot:" when it has none.
   - The header of each call gives the date, the weekday, the time zone and when the call is \
being made. "DAY SO FAR" above the log is the app's own picture of the day, built from \
judgments like yours: the focus spans it has summarized, the threads it has put together from \
thread pointers, and all of the day's Toggl entries. It is background for recognising what a \
segment belongs to (work resumed after an hour is visible there and nowhere else); it is not \
evidence about the segments you are judging.

8. SCREENSHOTS. Before the log come screenshots, in time order, each with a label naming the \
segment it belongs to and the time it was taken. Every EVALUATE segment has at least one where a \
file exists, long ones several spread over their length; context segments have their first and \
last capture as far as the call's image allowance reaches, nearest the EVALUATE stretch first. \
When the person had two displays, both are attached for the same moment: "active display" is \
the one holding the window the segment is about, "other display" is what else was in view (notes \
beside a text being read, a call, a video). The other display is context: activity is judged by \
the active one. The log says, under its line, which EVALUATE segment has no screenshot and why \
("no capture exists": none was ever taken, which happens with segments of a few seconds; "file \
no longer on disk": taken, since deleted). Window titles are sometimes \
stale or wrong; where a screenshot and a title disagree, the screenshot is what was on screen.

9. THE PERSON'S RULINGS. At the end of these instructions are all the rulings the person has \
made on past segments, each with what was on screen, what came before and after, what the app \
had judged, the question they were answering if there was one, and their note. They are your \
best evidence of how this person wants cases judged: follow them where a segment in front of \
you is the same kind of case, and say so in your note. A ruling is about its own situation: a \
"shift" or "continuation" ruling says what that segment was relative to what preceded it, so \
check that the situation matches, not only the app or site. Each call may list a few rulings \
whose site or window title matches its segments; that list is a pointer, not a selection - the \
others apply just as much.

Judge each segment in the context of the whole stretch - what came before AND after it. A flip to \
a notes app mid-video does not make the video session work; a quick message check inside a writing \
session is a shift away and a return, not the end of the work.

Their own rules and definitions (authoritative):
{rules}

About this person (their own words):
{about_me}

{examples}"""


# ---- what the evaluator is shown (2026-11-20) ---------------------------------------------------
# Eli: "We should also make sure that all the model calls are given all the information that
# could possibly be relevant for their judgement and nothing is left out." The helpers below
# build the pieces of a call: the context window, the log lines, the day so far, the person's
# rulings, and the screenshots. Every time a model reads comes from config.local_dt (the zone
# the Mac was in then).

def _t(ts: float, fmt: str = "%H:%M:%S") -> str:
    from .config import local_dt
    return local_dt(ts).strftime(fmt)


def _one(v) -> str:
    return " ".join(str(v or "").split())


def _context_start(conn, t0: float) -> float:
    """Where the log shown before a chunk begins: BEFORE_S earlier, and further back when that
    stretch is mostly an away line, until ACTIVE_BEFORE_S of activity is in view. 284 segments in
    30 days directly followed an absence of 10+ minutes and were judged with a single idle line
    before them (run 13441: "counts as a new start", unable to see what had been left)."""
    lo, now = t0 - BEFORE_S, time.time()
    act = sum(max(0.0, min(r[1], t0) - max(r[0], lo)) for r in conn.execute(
        "SELECT start, COALESCE(end, ?) FROM segments WHERE idle=0 AND start < ? AND COALESCE(end, ?) > ?", (now, t0, now, lo)))
    need = ACTIVE_BEFORE_S - act
    if need <= 0:
        return lo
    for r in conn.execute("SELECT start, COALESCE(end, start) FROM segments WHERE idle=0 AND start < ? AND start >= ?"
                          " ORDER BY start DESC", (lo, t0 - BEFORE_MAX_S)):
        d = min(r[1], lo) - r[0]
        if d <= 0:
            continue
        lo, need = r[0], need - d
        if need <= 0:
            break
    return lo


def _seg_brief(g) -> str:
    """A segments row in one short line, for the rulings list: when, how long, what."""
    end = g["end"] if g["end"] is not None else g["start"]
    if g["idle"]:
        return f"{_t(g['start'])} {_durtext(end - g['start'])} AWAY (no input)"
    return (f"{_t(g['start'])} {_durtext(end - g['start'])} {_one(g['app'])}"
            + (f" | site: {g['domain']}" if g["domain"] else "")
            + (f" | title: “{_one(g['title'])}”" if _one(g["title"]) else "")
            + (f" | url: {g['url']}" if g["url"] else ""))


def _durtext(sec: float) -> str:
    from .summaries import dur_text
    return dur_text(sec)


def _seg_moments(mom: dict, s: dict) -> list[dict]:
    """The capture moments to show for one segment to judge: all of them up to three, then about
    one per two minutes with the first and the last, at most SEG_MOMENTS_MAX."""
    ms = mom.get(s["id"]) or []
    if len(ms) <= 3:
        return list(ms)
    from . import shots as _shots
    dur = (s.get("end_eff") or s.get("clip_end") or 0) - s.get("start", 0)
    return _shots.spread(ms, min(len(ms), SEG_MOMENTS_MAX, max(3, int(dur // 120) + 2)))


def _moment_images(m: dict, what: str, app: str, both: bool = True) -> list[dict]:
    """Label + file for the image(s) of one capture moment: the active display, then the other
    display(s) when `both`. Until 2026-11-20 only the active display was ever sent, though a
    quarter of capture moments have a second file: notes beside a text, a call, a video."""
    a = m["active"]
    out = [{"ts": m["ts"], "path": a["path"],
            "label": f"Screenshot - {what} id={m['segment_id']} (taken {_t(m['ts'])}, {app}) - "
                     + ("active display:" if m.get("active_known", True) else
                        f"display {a['display']} (the active display's file is no longer on disk):")}]
    for o in (m["others"] if both else []):
        out.append({"ts": m["ts"] + 0.001, "path": o["path"],
                    "label": f"Screenshot - {what} id={m['segment_id']} (taken {_t(m['ts'])}, {app}) - "
                             "other display (not the one holding this segment's window):"})
    return out


def _image_content(images: list[dict]) -> tuple[list[dict], list[dict]]:
    """(message content blocks, [{path, label}] for eval_runs.shots) for the chosen images, in
    time order. A file that cannot be read after all is left out (and so not claimed)."""
    from .classifiers.claude_vision import _b64
    content, meta = [], []
    for im in sorted(images, key=lambda x: x["ts"]):
        b = _b64(im["path"])
        if not b:
            continue
        meta.append({"path": im["path"], "label": im["label"]})
        content.append({"type": "text", "text": im["label"]})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
    return content, meta


def _last_seen(conn, s: dict, d0: float, ctx0: float) -> str:
    """For a segment to judge: when the same window (app + title + URL) was last in front
    earlier in its day, before the log shown. A comeback after 12 minutes or two hours looks the
    same as a new start when the earlier visit is out of view (35% of segments judged a shift)."""
    if not (s.get("title") or s.get("url")):
        return ""     # an app with no recorded title: "the same app" says too little
    r = conn.execute("""SELECT g.id, g.start, g.end, e.content, e.switch_label, e.focused, e.self_distraction
                        FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
                        WHERE g.app=? AND g.title=? AND g.url=? AND g.idle=0 AND g.start >= ? AND g.start < ?
                          AND COALESCE(g.end, g.start) <= ? ORDER BY g.start DESC LIMIT 1""",
                     (s["app"], s.get("title") or "", s.get("url") or "", d0, s["start"], ctx0)).fetchone()
    if r is None:
        return ""
    n, tot = conn.execute("""SELECT COUNT(*), COALESCE(SUM(COALESCE(end, start) - start), 0) FROM segments
                             WHERE app=? AND title=? AND url=? AND idle=0 AND start >= ? AND start < ?""",
                          (s["app"], s.get("title") or "", s.get("url") or "", d0, s["start"])).fetchone()
    if r["content"]:
        j = f"judged <{r['content']}>" + ("" if r["focused"] is None else (" <focused>" if r["focused"] else " <unfocused>")) \
            + f" <switch: {shown_switch(r['switch_label']) or 'continuation'}" + (", self-distraction" if r["self_distraction"] else "") + ">"
    else:
        j = "not judged"
    return (f"earlier today: before the log shown here, this same window (app, title and url) was last in front at {_t(r['start'])} for "
            f"{_durtext((r['end'] or r['start']) - r['start'])} (id={r['id']}), {j}; {n} earlier visit(s) today, {_durtext(tot)} in all")


def _toggl_lines(conn, lo: float, hi: float, T0: float | None, T1: float | None, mark: str = "EVALUATE") -> tuple[list[str], dict]:
    """The Toggl entries overlapping [lo, hi), one line each with times to the second, project
    and tags, and which were running during the stretch to judge [T0, T1]. Until 2026-11-20 only
    the entries overlapping the 20-odd minutes shown were listed, description only: of 2,409
    first-pass prompts that said "(none)", 1,894 had entries earlier that day. An entry with no
    stop is printed as still running with the time of the last sync - it used to be printed as
    ending at the time of the call, which reads as stopped. Returns (lines, {entry id: row})."""
    from . import toggl
    rows = toggl.entries_between(conn, lo, hi)
    out = []
    for e in rows:
        stop = e["stop"]
        span = (f"{_t(e['start'])}-{_t(stop)} ({_durtext(stop - e['start'])})" if stop is not None else
                f"{_t(e['start'])}-still running (as of the last sync with Toggl, {_t(e['synced'], '%Y-%m-%d %H:%M')})")
        what = (f"“{_one(e['description'])}”" if e["description"] else "(no description)") \
            + (f" | project: {e['project']}" if e["project"] else "") + (f" | tags: {e['tags']}" if e["tags"] else "")
        note = ""
        if T0 is not None and e["start"] < T1 and (stop is None or stop > T0):
            whole = e["start"] <= T0 and (stop is None or stop >= T1)
            note = f"   <- running during {'all' if whole else 'part'} of the {mark} stretch"
        out.append(f"  {span}  {what}{note}")
    return out, {e["id"]: e for e in rows}


def _day_block(conn, t0: float, T0: float | None, T1: float | None, ctx0: float, ctx1: float, now: float,
               mark: str = "EVALUATE") -> tuple[str, dict]:
    """The "day so far" part of a call: the logical day's focus-span summaries, its judged
    threads, and all its Toggl entries. What a segment belongs to is often visible only here
    (work picked up again after an hour). Returns (text, {toggl entry id: row})."""
    from . import stats
    from .config import day_bounds, logical_date
    d0, d1 = day_bounds(logical_date(t0))
    hi = min(d1, now)
    parts = []
    sums = conn.execute("SELECT * FROM span_summaries WHERE start < ? AND end > ? ORDER BY start", (d1, d0)).fetchall()
    if sums:
        parts.append("Focus-span candidates the app has summarized this day (each written by a model from the judgments "
                     "in force at the time; descriptions, not rulings):\n" + "\n".join(
                         f"  {_t(r['start'])}-{_t(r['end'])} ({_durtext(r['end'] - r['start'])})"
                         + (f"  “{_one(r['title'])}”" if r["title"] else "") + f": {_one(r['summary'])}"
                         + ("" if r["coherent"] else f"  [the app's own check doubted this is one object of attention: {_one(r['issue'])}]")
                         for r in sums))
    else:
        parts.append("Focus-span candidates summarized this day: none yet.")
    # Threads: chains of judged segments linked by thread pointers; a shift (or a first segment) starts one.
    day = [s for s in stats.labelled_segments(conn, d0, hi, phone=False)
           if s["duration"] > 0 and not s["idle"] and not s.get("neutral") and s.get("source") in ("eval", "review")]
    root: dict = {}
    prev = None
    for s in day:
        a = s.get("antecedent_id")
        if s.get("switch") in ("continuation", "return") and a in root:
            root[s["id"]] = root[a]
        elif s.get("switch") == "continuation" and prev is not None:
            root[s["id"]] = root[prev]     # a continuation stored without a pointer continues the line before it
        else:
            root[s["id"]] = s["id"]
        prev = s["id"]
    threads: dict = {}
    for s in day:
        th = threads.setdefault(root[s["id"]], {"start": s["clip_start"], "end": s["clip_end"], "sec": 0.0, "n": 0,
                                                "content": {}, "foc": 0.0, "focn": 0.0, "win": {}})
        th["end"] = max(th["end"], s["clip_end"])
        th["sec"] += s["duration"]
        th["n"] += 1
        th["content"][s.get("content") or "?"] = th["content"].get(s.get("content") or "?", 0.0) + s["duration"]
        if s.get("focused") is not None:
            th["focn"] += s["duration"]
            th["foc"] += s["duration"] if s["focused"] else 0.0
        w = _one(s.get("app")) + (f" | {s['domain']}" if s.get("domain") else "") + (f" | “{_one(s['title'])}”" if _one(s.get("title")) else "")
        th["win"][w] = th["win"].get(w, 0.0) + s["duration"]
    big = [th for th in threads.values() if th["sec"] >= 60]
    small = [th for th in threads.values() if th["sec"] < 60]
    if big:
        lines = []
        for th in sorted(big, key=lambda x: x["start"]):
            cont = ", ".join(f"{c} {_durtext(v)}" for c, v in sorted(th["content"].items(), key=lambda x: -x[1]))
            wins = "; ".join(f"{w} ({_durtext(v)})" for w, v in sorted(th["win"].items(), key=lambda x: -x[1])[:3])
            foc = f", {100 * th['foc'] / th['focn']:.0f}% of it judged focused" if th["focn"] else ""
            lines.append(f"  {_t(th['start'])}-{_t(th['end'])}: {_durtext(th['sec'])} in {th['n']} segment(s); {cont}{foc}; mostly {wins}")
        parts.append("Threads the app has put together this day from the judgments it holds (a thread = segments linked by "
                     "thread pointers; first to last time, time in it, content, main windows; threads of a minute or more):\n"
                     + "\n".join(lines)
                     + (f"\n  plus {len(small)} shorter thread(s), {_durtext(sum(th['sec'] for th in small))} in all" if small else ""))
    else:
        parts.append("Threads judged this day so far: none of a minute or more"
                     + (f" ({len(small)} shorter, {_durtext(sum(th['sec'] for th in small))} in all)." if small else "."))
    tl, entries = _toggl_lines(conn, min(d0, ctx0), max(hi, ctx1), T0, T1, mark)
    parts.append("Toggl entries this day (all of them, as Toggl has them now; times to the second):\n"
                 + ("\n".join(tl) if tl else "  (none)"))
    return "\n\n".join(parts), entries


def _log_lines(conn, segs: list[dict], marks: dict, entries: dict, history: bool = False,
               notes: dict | None = None) -> list[tuple[float, int, str]]:
    """The activity log's computer lines, one entry per segment (with its indented sub-lines):
    [(start, 0, text)]. `marks` {segment id: prefix} names the segments to judge; every other
    line is "(context id=N)". `notes` {segment id: [extra sub-lines]}. One builder for the
    evaluator and the daily audit, so both read the same thing."""
    from . import signals as _signals, summaries
    ids = [s["id"] for s in segs if s["duration"] > 0]
    evrow, rulings = {}, {}
    for k in range(0, len(ids), 500):
        part = ids[k:k + 500]
        qs = ",".join("?" * len(part))
        evrow.update({r["segment_id"]: r for r in conn.execute(
            f"SELECT segment_id, rationale, created FROM seg_evals WHERE segment_id IN ({qs})", part)})
        rulings.update({r["segment_id"]: r for r in conn.execute(f"SELECT * FROM seg_reviews WHERE segment_id IN ({qs})", part)})
    out, prev_app = [], None
    for s in segs:
        if s["duration"] <= 0:
            continue
        idle = bool(s["idle"])   # (not state == "idle": that also covers a real window labelled idle)
        e, rv = evrow.get(s["id"]), rulings.get(s["id"])
        te = entries.get(s.get("toggl_id"))
        s2 = dict(s, toggl_project=(te["project"] if te else ""))
        real = dict(s, clip_start=s["start"], clip_end=s["end_eff"])   # signals over the whole segment, not the part in the window
        # Context lines carry their id too: the antecedent of the first segment to judge is
        # usually a context line, and without an id there the evaluator named a LATER segment
        # instead (195 pointers dropped as invalid, 2026-11-13 to 11-20).
        mark = marks.get(s["id"]) or f"(context id={s['id']}) "
        # what the Mac recorded beside the capture: how the app came forward, how the page
        # was reached, input rates, audio / microphone / camera (signals.py)
        sig = _signals.describe(conn, real, prev_app, full=True) if not idle else []
        prev_app = None if idle else s.get("app")
        # (history re-judgment: a judgment from before the effort fix is an earlier quick pass, not settled fact)
        quick = history and e is not None and e["created"] < LOW_EFFORT_FIXED_AT
        line = (mark + summaries.seg_line(s2, full=True, ruling=rv, quick_pass=quick)
                + (f"  [{'; '.join(sig)}]" if sig else "")
                # the whole note: cut at 80 characters, 73% of them ended mid-word, and in a context
                # line the note is the only description of what was on screen
                + ((("  // the app's note (it may predate their ruling): " if rv is not None else
                     "  // quick-pass note: " if quick else "  // note: ") + _one(e["rationale"]))
                   if e is not None and e["rationale"] and not idle else ""))
        subs = []
        if not idle:
            a, b = s["start"], s["end_eff"]
            if b - a > 120:
                strip = _signals.minute_strip(conn, a, b)
                if strip:
                    subs.append("input by minute (keys/clicks/scroll events/pointer moves): " + strip)
            ps = _signals.pauses(conn, a, b)
            if ps:
                subs.append("no input: " + "; ".join(f"{_t(x)}-{_t(y)} ({_durtext(y - x)})" for x, y in ps))
            if s.get("app") == "Google Chrome":
                loads = _signals.page_load_lines(conn, a, b)
                if loads:
                    more = len(loads) - PAGE_LOADS_MAX
                    subs.append("page loads during this segment:\n" + "\n".join("        " + x for x in loads[:PAGE_LOADS_MAX])
                                + (f"\n        ... and {more} more page load(s) after these, not listed" if more > 0 else ""))
        subs += (notes or {}).get(s["id"], [])
        out.append((s["clip_start"], 0, line + "".join("\n      " + x for x in subs)))
    return out


PHONE_SEG_SHOTS = 3        # iPhone captures shown per phone app segment: first, last, one between
PHONE_PICKUP_SHOTS = 8     # and per pickup


def _phone_images(cfg, p: dict) -> tuple[list[dict], dict]:
    """(images, {seg key: count}) of the iPhone's own screen for one pickup to judge: the vetted
    captures of each app segment (phone.captures), a few per segment."""
    from . import phone as _phone
    from . import shots as _shots
    out, counts = [], {}
    if p.get("lock_screen"):
        return out, counts
    try:
        caps = _phone.captures(cfg, p["start"] - 60, p["end"] + 60)
    except Exception:
        log.exception("phone captures unavailable")
        return out, counts
    used: set = set()
    for g in p["segs"]:
        mine = [k for k in _phone.captures_of(caps, g["start"], g["end"], g["app"]) if k["name"] not in used]
        for k in (_shots.spread(mine, PHONE_SEG_SHOTS) if len(mine) > PHONE_SEG_SHOTS else mine):
            if len(out) >= PHONE_PICKUP_SHOTS:
                break
            used.add(k["name"])
            counts[g["key"]] = counts.get(g["key"], 0) + 1
            out.append({"ts": max(k["ts"], g["start"]) + 0.002, "path": k["name"],
                        "label": f"Screenshot - the iPhone's screen during PHONE id=P{g['key']} "
                                 f"(taken {_t(k['ts'])}, iPhone: {g['app']}):"})
    return out, counts


def _phone_lines(conn, picks_all: list[dict], pick_keys: set, pk_have: dict,
                 pshots: dict | None = None) -> list[tuple[float, int, str]]:
    """PHONE lines, one per app segment of each pickup in view: [(start, 1, text)]. Those of a
    pickup in `pick_keys` are marked EVALUATE PHONE; the rest are context and carry the judgment
    they already have."""
    from . import signals as _signals
    out = []
    for p in picks_all:
        for gi, g in enumerate(p["segs"]):
            had = pk_have.get(g["key"])
            bits = [_t(g["start"]), _durtext(g["end"] - g["start"])]
            if p["lock_screen"]:
                bits.append("glanced at the iPhone's lock screen, did not unlock it")
            else:
                bits.append("iPhone: " + g["app"])
                if g["how"]:
                    bits.append("(opened from " + ("a notification" if g["how"] == "notification" else
                                                   g["how"] if g["how"] == "another app" else "the " + g["how"]) + ")")
                if gi == 0:
                    bits.append("(phone unlocked)")
                if g["first_today"]:
                    bits.append(f"(first {g['app']} open today)")
                if gi == len(p["segs"]) - 1:
                    bits.append("(then left open, untouched)" if p["put_down"] else "(then locked)")
            if (p["key"] not in pick_keys or p["lock_screen"]) and had is not None and had["switch_label"]:
                bits.append(f"<{had['content']}>")
                bits.append("<focused>" if had["focused"] else "<unfocused>")
                if had["switch_label"] == "interruption":
                    bits.append("<switch: shift" + (", self-distraction" if had["self_distraction"] else "")
                                + (", next task in the block" if had["next_task"] else "") + ">")
                elif had["switch_label"] == "return":
                    bits.append("<switch: return>")
            # Whether the Mac was being used meanwhile is the one direct sign of whether the phone
            # was in hand (run 13251: a 20 s Oura look attached to the work on screen, unseen).
            used = "" if p["lock_screen"] else _signals.input_around(conn, g["start"], g["end"])
            if used:
                bits.append(f"[{used}]")
            if pshots and pshots.get(g["key"]):
                bits.append(f"[iPhone screenshots attached: {pshots[g['key']]}]")
            out.append((g["start"], 1, (f"EVALUATE PHONE id=P{g['key']}: " if p["key"] in pick_keys and not p["lock_screen"]
                                         else "(context) PHONE ") + "  ".join(bits)))
    return out


def _call_header(t0: float, T0: float | None, T1: float | None, now: float, what: str = "judge") -> str:
    from .config import day_start_hour, system_zone_name, zone_at
    z = zone_at(t0)
    zn = getattr(zone_at(now), "key", None) or system_zone_name() or "the Mac's system zone"
    zone = getattr(z, "key", None) or system_zone_name() or "the Mac's system zone"
    ago = ""
    if T1 is not None:
        d = now - T1
        ago = (f"; the last segment to {what} ended " +
               (f"{d / 60:.0f} minutes" if d < 2 * 3600 else f"{d / 3600:.0f} hours" if d < 2 * 86400 else f"{d / 86400:.0f} days")
               + " before that")
    return (f"THIS CALL: activity of {_t(t0, '%A %Y-%m-%d')} (the person's days run from {day_start_hour()}:00 to {day_start_hour()}:00; times are in "
            f"{zone}, the zone their Mac was in that day)"
            + (f". Segments to {what}: {_t(T0)} to {_t(T1)}" if T0 is not None else "")
            + f". This call is being made at {_t(now, '%Y-%m-%d %H:%M')} ({zn} time){ago}.")


_ev = None


def _get():
    global _ev
    if _ev is None:
        _ev = ChunkEvaluator(load_config())
    return _ev


class ChunkEvaluator:
    def __init__(self, cfg: Config):
        import anthropic
        from .config import load_api_key
        load_api_key()
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        if not (getattr(self.client, "api_key", None) or getattr(self.client, "auth_token", None)):
            raise RuntimeError("no Anthropic credentials")
        self._examples_stamp = None
        self._system = None
        self._judged_cache: dict = {}   # (segment id, ruling time) -> what the app had judged before that ruling
        self.last_end = 0.0             # where the last evaluate_chunk call stopped (see there)

    # The person's rulings on past segments reach the model in full, in the cached system prompt
    # (_rulings_block). Until 2026-11-19 only the latest 20 of ~300 were shown; then the 8 newest
    # plus up to 20 "similar" ones per call, as one line each (app, site, 60 characters of title,
    # verdict, note). Measured on 2026-11-20: only 115 of 326 rulings had appeared in any prompt;
    # 96 had neither title nor site, so they read "Agent Coding | -> shift"; a Chrome ruling with
    # no site matched every Chrome segment (one was shown in 59 runs, one for a cloud console
    # page); and none said what the segment had continued or left, though a switch ruling is
    # about exactly that. Each call still names the few rulings whose site or title matches its
    # segments (_similar_precedents), as a pointer into the full list.
    SIMILAR_PRECEDENTS = 20
    _PREC_SQL = """SELECT r.*, g.app, g.domain, g.title, g.url, g.start AS g_start, g.end AS g_end
                   FROM seg_reviews r JOIN segments g ON g.id = r.segment_id ORDER BY r.created DESC"""

    @staticmethod
    def _precedent_line(r) -> str:
        """One ruling in one line (the question writer's precedents, and the per-call pointers)."""
        from .summaries import verdict_text
        what = (f"{_one(r['app'])}" + (f" | site: {r['domain']}" if r["domain"] else "")
                + (f" | title: “{_one(r['title'])}”" if _one(r["title"]) else "") + (f" | url: {r['url']}" if r["url"] else ""))
        note = f" ({_one(r['note'])})" if r["note"] else ""
        return f"- segment id={r['segment_id']}, {_t(r['g_start'], '%Y-%m-%d %H:%M')}: {what} -> {verdict_text(r)}{note}"

    @staticmethod
    def _group_key(r):
        """Rulings made in one act (one chat proposal applied to a run of segments, or one bulk
        edit: the same second and note) are one entry: four notes cover 134 of 326 rulings."""
        return (r["chat_message_id"] if r["chat_message_id"] is not None else int(r["created"]), r["note"] or "")

    def _rulings(self, conn) -> list:
        """Every ruling that sets at least one field, newest first. One that sets none says
        nothing (the ruling on 36249 stood in every prompt as "->  (solo, not hybrid)" beside a
        note about a Toggl entry, until hybrid was its only field; a row with no field at all
        would read as an empty verdict)."""
        from .summaries import verdict_text
        # ("solo, not hybrid" alone is that case: the block editor stored hybrid=0 with every save)
        return [r for r in conn.execute(self._PREC_SQL).fetchall() if verdict_text(r) not in ("", "solo, not hybrid")]

    def _app_judged(self, conn, r) -> str:
        """What the app had judged for a ruled segment BEFORE the person ruled - the thing they
        corrected or confirmed. The live seg_evals row when it is older than the ruling; when
        the segment was judged again since (102 of 326), the answer in the transcript of the
        last evaluation run before the ruling (eval_runs.response)."""
        from .summaries import verdict_text
        e = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (r["segment_id"],)).fetchone()
        if e is not None and e["created"] <= r["created"]:
            bits = [f"content={e['content']}", f"switch={shown_switch(e['switch_label']) or 'continuation'}",
                    f"continues id={e['antecedent_id']}" if e["antecedent_id"] else None,
                    None if e["focused"] is None else ("focused" if e["focused"] else "unfocused"),
                    "self-distraction" if e["self_distraction"] else None,
                    "next task in the block" if e["next_task"] else None,
                    f"a sub-thread that grew out of id={e['sub_of']}" if e["sub_of"] else None,
                    "hybrid (inside a live discussion)" if e["hybrid"] else None,
                    f"kind of shift (older label)={e['interruption_kind']}" if e["interruption_kind"] and e["focused"] is None else None,
                    "flagged uncertain" if e["uncertain"] else None]
            return ", ".join(b for b in bits if b) + (f" - “{_one(e['rationale'])}”" if e["rationale"] else "")
        for run in conn.execute("""SELECT response FROM eval_runs WHERE t0 <= ? AND t1 >= ? AND created < ?
                                   AND response IS NOT NULL ORDER BY created DESC LIMIT 6""",
                                (r["g_start"], r["g_start"], r["created"])):
            try:
                js = json.loads(run["response"])
                j = next((x for x in (js.get("segments", []) if isinstance(js, dict) else js)
                          if x.get("segment_id") == r["segment_id"]), None)
            except (ValueError, AttributeError, TypeError):
                j = None
            if j:
                sw = shown_switch(stored_switch(str(j.get("switch") or "").strip().lower())) or j.get("switch")
                kind = j.get("shift_kind") or j.get("interruption_kind")
                bits = [f"content={j.get('content')}", f"switch={sw}",
                        None if "focused" not in j else ("focused" if j["focused"] else "unfocused"),
                        "self-distraction" if j.get("self_distraction") else None,
                        "next task in the block" if j.get("next_task") else None,
                        "hybrid (inside a live discussion)" if j.get("hybrid") else None,
                        f"kind of shift (older label)={kind}" if kind and "focused" not in j else None]
                return ", ".join(b for b in bits if b) + (f" - “{_one(j.get('note'))}”" if j.get("note") else "")
        return "not on record (no judgment was stored for it before the ruling)"

    def _rulings_block(self, conn) -> str:
        """All of the person's rulings, oldest first, for the system prompt. Per ruling: when and
        what the segment was (app, site, whole title, URL), the Toggl entry then, the line before
        and the line after, what it continued (for a continuation / return ruling), what the app
        had judged, the review question and its context when the ruling was an answer to one,
        and the fields they set with their note."""
        from . import toggl
        from .summaries import verdict_text
        rows = self._rulings(conn)
        if not rows:
            return ""
        qs = [q for q in conn.execute("SELECT * FROM review_questions WHERE status='answered' AND answered_at IS NOT NULL")]
        qopts = {}
        for q in qs:
            try:
                o = json.loads(q["options"])[q["answer_index"]] if q["answer_index"] is not None else {}
            except (ValueError, IndexError, TypeError):
                o = {}
            qopts[q["id"]] = o.get("segment_id") if isinstance(o, dict) else None
        groups: dict = {}
        for r in reversed(rows):     # oldest first
            groups.setdefault(self._group_key(r), []).append(r)
        out = [f"THE PERSON'S RULINGS ({len(rows)} segments ruled on, in {len(groups)} entries, oldest first; where two "
               "disagree the later one is their current view). Times are in the zone they were in that day. \"the app had "
               "judged\" is what stood before they ruled: where their ruling differs it is a correction, where it matches "
               "they confirmed it."]
        for members in groups.values():
            members = sorted(members, key=lambda r: r["g_start"])
            ids = {r["segment_id"] for r in members}
            note = _one(members[0]["note"])
            verdicts = {verdict_text(r) for r in members}
            head = (f"Ruling on segment id={members[0]['segment_id']}" if len(members) == 1 else
                    f"One ruling covering {len(members)} segments (ids {', '.join(str(r['segment_id']) for r in members)})")
            lines = [f"{head}, made {_t(members[0]['created'], '%Y-%m-%d %H:%M')}:"]
            for r in members:
                end = r["g_end"] if r["g_end"] is not None else r["g_start"]
                pre = "  " if len(members) == 1 else "    "
                if len(members) > 1:
                    lines.append(f"  - segment id={r['segment_id']}")
                lines.append(f"{pre}the segment: {_t(r['g_start'], '%a %Y-%m-%d')} " + _seg_brief(
                    {"start": r["g_start"], "end": end, "idle": 0, "app": r["app"], "domain": r["domain"], "title": r["title"], "url": r["url"]}))
                te = toggl.entry_at(conn, r["g_start"])
                lines.append(f"{pre}toggl then: " + (toggl.describe(te) if te else "no entry running"))
                b = conn.execute("SELECT * FROM segments WHERE start < ? AND COALESCE(end, start) > start ORDER BY start DESC LIMIT 1",
                                 (r["g_start"],)).fetchone()
                a = conn.execute("SELECT * FROM segments WHERE start > ? AND COALESCE(end, start) > start ORDER BY start LIMIT 1",
                                 (r["g_start"],)).fetchone()
                if b is not None:
                    lines.append(f"{pre}just before: " + (f"segment id={b['id']} of this ruling" if b["id"] in ids else _seg_brief(b)))
                if a is not None:
                    lines.append(f"{pre}just after: " + (f"segment id={a['id']} of this ruling" if a["id"] in ids else _seg_brief(a)))
                if r["antecedent_id"]:
                    g2 = conn.execute("SELECT * FROM segments WHERE id=?", (r["antecedent_id"],)).fetchone()
                    if g2 is not None:
                        lines.append(f"{pre}what they say it continued (id={g2['id']}): " + _seg_brief(g2))
                key = (r["segment_id"], r["created"])
                if key not in self._judged_cache:
                    self._judged_cache[key] = self._app_judged(conn, r)
                lines.append(f"{pre}the app had judged: " + self._judged_cache[key])
                q = next((q for q in qs if abs(q["answered_at"] - r["created"]) < 5
                          and r["segment_id"] in (q["segment_id"], qopts.get(q["id"]))), None)
                if q is not None:
                    lines.append(f"{pre}they were asked: {_one(q['context'])} {_one(q['question'])}")
                    lines.append(f"{pre}they answered: “{_one(q['answer'])}”"
                                 + (f"; their note on the answer: “{_one(q['note'])}”" if _one(q["note"]) and _one(q["note"]) != note else ""))
                if len(verdicts) > 1:
                    lines.append(f"{pre}THEIR RULING for this segment: " + verdict_text(r))
            lines.append("  THEIR RULING" + (" (each segment as listed)" if len(verdicts) > 1 else ": " + next(iter(verdicts)))
                         + (f"; their note: “{note}”" if note else ""))
            out.append("\n".join(lines))
        try:
            ph = conn.execute("""SELECT v.*, p.app, p.start AS p_start, p.end AS p_end, p.content AS e_content, p.switch_label AS e_switch,
                                        p.focused AS e_focused, p.rationale FROM phone_reviews v JOIN phone_evals p ON p.seg_key = v.seg_key
                                 ORDER BY v.created""").fetchall()
        except Exception:   # a database from before the phone tables
            ph = []
        if ph:
            lines = ["Their rulings on PHONE segments (app on the iPhone's screen):"]
            for v in ph:
                bits = [f"switch={shown_switch(v['switch_label'])}" if v["switch_label"] else None,
                        f"content={v['content']}" if v["content"] else None,
                        None if v["focused"] is None else ("focused" if v["focused"] else "unfocused"),
                        None if v["self_distraction"] is None else ("self-distraction" if v["self_distraction"] else "not a self-distraction"),
                        None if v["away"] is None else ("time away from what was on the Mac" if v["away"] else "part of what was on the Mac")]
                lines.append(f"  - {_t(v['p_start'], '%a %Y-%m-%d %H:%M:%S')} {_durtext(v['p_end'] - v['p_start'])} iPhone: {v['app']}; "
                             f"the app had judged: content={v['e_content']}, switch={shown_switch(v['e_switch'])} - “{_one(v['rationale'])}”; "
                             f"THEIR RULING: {', '.join(b for b in bits if b) or '(no field set)'}"
                             + (f"; their note: “{_one(v['note'])}”" if v["note"] else ""))
            out.append("\n".join(lines))
        return "\n\n".join(out)

    def _examples(self, conn) -> str:
        return self._rulings_block(conn)

    @staticmethod
    def _title_words(t: str | None) -> set:
        return {w for w in re.findall(r"[a-z0-9]{3,}", (t or "").lower())}

    def _similar_precedents(self, conn, target: list[dict], skip_recent: bool = True, pointers: bool = False) -> str:
        """The person's past rulings on segments like the ones in `target`. A ruling counts only
        on a REAL match: the same site (an empty site and Chrome's "newtab" are not sites), or
        window titles that share most of their words, or - for an app that never records titles -
        the same app. (Until 2026-11-20 "same app, no site on either side" scored as a match, so
        a Chrome ruling with no site was listed for every Chrome segment.) Rulings made in one
        act are listed once; at most 5 per app+site, the best SIMILAR_PRECEDENTS in all, newer
        first among equals. With `pointers` (the evaluator, which has every ruling in its system
        prompt) the lines point into that list; without (the question writer) they stand alone.
        `skip_recent` is kept for the callers that pass it; it no longer changes anything."""
        rows = self._rulings(conn)
        tids = {s["id"] for s in target}
        real = lambda d: d if d and d != "newtab" else ""
        tg = [(s.get("app") or "", real(s.get("domain") or ""), self._title_words(s.get("title")), s["id"]) for s in target]
        titled: dict = {}

        def has_titles(app):
            if app not in titled:
                titled[app] = conn.execute("SELECT 1 FROM segments WHERE app=? AND title != '' LIMIT 1", (app,)).fetchone() is not None
            return titled[app]
        scored, seen = [], set()
        for r in rows:
            if r["segment_id"] in tids:
                continue
            ra, rd, rw = r["app"] or "", real(r["domain"] or ""), self._title_words(r["title"])
            best, like = 0.0, None
            for ta, td, tw, tid in tg:
                sc = 4.0 if rd and rd == td else 0.0
                shared = rw & tw
                if rw and tw and (rw == tw or (len(shared) >= 2 and len(shared) / len(rw | tw) >= 0.34)):
                    sc += 3.0 * len(shared) / len(rw | tw) + (0.5 if ra == ta else 0.0)
                if not sc and ra == ta and not rd and not td and not rw and not tw and not has_titles(ra):
                    sc = 2.0
                if sc > best:
                    best, like = sc, tid
            dedupe = (self._group_key(r), ra, rd, _one(r["title"]))
            if best > 0 and dedupe not in seen:
                seen.add(dedupe)
                scored.append((best, r, like))
        scored.sort(key=lambda x: (-x[0], -x[1]["created"]))
        out, per = [], {}
        for _, r, like in scored:
            k = (r["app"], r["domain"])
            if per.get(k, 0) >= 5:
                continue
            per[k] = per.get(k, 0) + 1
            out.append(self._precedent_line(r) + (f"   [same site or title as id={like}]" if pointers else ""))
            if len(out) >= self.SIMILAR_PRECEDENTS:
                break
        if not out:
            return ""
        head = ("Rulings of theirs on the same site or window title as a segment in this call (each is in full in your "
                "instructions, under its segment id; this list is a pointer, not a selection - their other rulings apply as much):"
                if pointers else
                "The person's past rulings on segments that resemble ones in this stretch - follow these precedents:")
        return "\n".join([head] + out)

    def _system_blocks(self, conn) -> list[dict]:
        from .classifiers.claude_vision import _rules_doc, _rules_mtime
        # Rebuilt when ANY ruling is added, changed or removed. The stamp used to be the newest
        # ruling's time, so undoing or editing an older one left it in the prompt until the next
        # ruling came in.
        import hashlib
        h = hashlib.md5()
        for r in conn.execute("SELECT * FROM seg_reviews ORDER BY segment_id"):
            h.update(repr(tuple(r)).encode())
        try:
            for r in conn.execute("SELECT * FROM phone_reviews ORDER BY seg_key"):
                h.update(repr(tuple(r)).encode())
        except Exception:
            pass
        stamp = (h.hexdigest(), _rules_mtime())
        if self._system is None or stamp != self._examples_stamp:
            self._system = SYSTEM.format(rules=_rules_doc(),
                                         about_me=self.cfg.about_me.strip() or "(no description given)",
                                         examples=self._examples(conn))
            self._examples_stamp = stamp
        return [{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}]

    def evaluate_chunk(self, conn, t0: float, t1: float, extra: str | None = None,
                       phone_only: bool = False, history: bool = False) -> int:
        """Evaluate all real segments starting in [t0, t1). Local models predict first (and are
        stored for track-record scoring); segments where the locals have EARNED confident
        deferral skip the LLM; the rest go into one LLM call, and the resolved judgment is the
        accuracy-weighted ensemble of all models. Returns how many segments were judged.

        A call never carries more segments than its image budget can show (IMAGE_BUDGET): when
        the stretch holds more, only its first part is judged, and `self.last_end` says where
        this call stopped (t1 when it took everything). Callers that re-judge a fixed stretch
        go on from there; the first pass finds the rest by itself, as unjudged segments.
        `history` marks a call of the history re-judgment (older judgments in view are shown as
        an earlier quick pass)."""
        from . import seg_models, shots as _shots, stats
        now0 = time.time()
        self.last_end = t1
        first_pass = extra is None and not phone_only
        ctx0 = _context_start(conn, t0)
        ctx1 = max(t1, min(now0, t1 + (AFTER_FIRST_MAX_S if first_pass else AFTER_REJUDGE_S)))
        segs = stats.labelled_segments(conn, ctx0, ctx1, phone=False)  # pickups are shown as their own lines
        target = [s for s in segs if t0 <= s["clip_start"] < t1 and not s["idle"]
                  and not s.get("neutral") and s["duration"] > 0]
        if not phone_only:
            # Only segments that ended at least LAZY_S ago. A long segment starting just after a
            # short one used to be judged along with it, on its first ten minutes and sometimes
            # before it had ended (a 33-minute segment judged at minute 15, 2026-11-19); the
            # judgment was final. It waits for the next cycle instead. Re-judgments skipped this
            # filter until 2026-11-20: run 13375, after a ruling, judged segment 36431 twenty
            # seconds before it ended, and 20 such early judgments stood. The first pass takes
            # whatever a re-judgment leaves.
            settled = now0 - LAZY_S
            target = [s for s in target if (s.get("end") or now0) <= settled]
        if phone_only:
            # Phone data that arrived late: judge the pickups against the activity already
            # judged, and leave the computer segments' judgments alone. Re-judging them too was
            # tried (2026-11-19): it cost 6x as much per call and rewrote 23 settled judgments
            # in one stretch for no reason connected to the phone.
            target = []
        # iPhone pickups in view (phone.pickups). Those starting in [t0, t1) that have no
        # judgment yet are judged in this call; on a re-judgment of the segments all of them are.
        from . import phone as _phone
        p0 = max(ctx0, t0 - BEFORE_S)
        picks_all = _phone.pickups(self.cfg, p0, ctx1)
        for p in picks_all:   # lock-screen glances are settled by rule, not by the model
            if p["lock_screen"] and t0 <= p["start"] < t1:
                store_lock_glance(conn, p)
        pk_have = {r["seg_key"]: r for r in conn.execute(
            "SELECT * FROM phone_evals WHERE start >= ? AND start < ?", (p0 - CTX_S - 1800, ctx1 + CTX_S + 1800))}
        # A pickup is judged whole: every app segment in it, when any of them has no judgment.
        picks = [p for p in picks_all if t0 <= p["start"] < t1 and not p["lock_screen"]
                 and ((extra is not None and not phone_only) or any(
                     g["key"] not in pk_have or pk_have[g["key"]]["switch_label"] is None for g in p["segs"]))]
        if not target and not picks:
            return 0
        # -- local models + deferral decision ---------------------------------------------
        sc = seg_models.scores(conn)
        real = [s for s in segs if not s["idle"] and not s.get("neutral") and s["duration"] > 0]
        local: dict = {}   # id -> (per-model preds, combined dist per head)
        deferred: set = set()
        now = time.time()
        for s in target:
            i = real.index(s)
            prev = real[i - 1] if i > 0 else None
            lp = seg_models.local_predict(conn, self.cfg, s, prev)
            for m, p in lp.items():
                seg_models.store_prediction(conn, s["id"], m, p)
            combined = {h: seg_models.combine_head({m: lp[m][h] for m in lp},
                                                   seg_models.weights_for(sc, h, list(lp)))
                        for h in HEADS}
            local[s["id"]] = (lp, combined)
            if extra is None and seg_models.can_defer(sc, combined, self.cfg):
                deferred.add(s["id"])
        for s in target:
            if s["id"] not in deferred:
                continue
            _, comb = local[s["id"]]
            c = max(comb["content"], key=comb["content"].get)
            sw = max(comb["switch"], key=comb["switch"].get)
            k = max(comb["kind"], key=comb["kind"].get) if sw == "interruption" else None
            conn.execute(
                "INSERT OR REPLACE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, probs, created)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (s["id"], c, sw, k, 0, "(deferred to local models - earned track record)",
                 json.dumps(comb), now))
        if deferred:
            from . import propagation
            propagation.reapply(conn, sorted(deferred))
        target = [s for s in target if s["id"] not in deferred]
        if not target and not picks:
            log.info("chunk %s-%s: all %d segments deferred to local models ($0)",
                     dt.datetime.fromtimestamp(t0).strftime("%H:%M"),
                     dt.datetime.fromtimestamp(t1).strftime("%H:%M"), len(deferred))
            conn.execute("INSERT INTO eval_runs(t0, t1, n_segments, cost_usd, created) VALUES (?,?,?,?,?)",
                         (t0, t1, len(deferred), 0.0, now))
            return len(deferred)
        # -- screenshots ------------------------------------------------------------------
        shown = [s for s in segs if s["duration"] > 0 and not s["idle"]]
        mom = _shots.moments(conn, list({s["id"] for s in shown}))
        if extra is not None and target:
            # A RE-judgment with no image of a segment must not replace a judgment that may have
            # been made with one. Files exist only from 2026-10-27 13:35; of the wrong-display
            # re-judgments on 11-17, 30 runs (activity of 10-13 to 10-27) attached no screenshot
            # at all and their 244 judgments replaced earlier ones (run 12807, segment 23931:
            # "With no screenshot from the right display I can't confirm it was publishing work,
            # so I'm leaning passive"). Such a segment keeps its judgment and is shown as context.
            judged_ids = {r[0] for r in conn.execute(
                f"SELECT segment_id FROM seg_evals WHERE segment_id IN ({','.join('?' * len(target))})", [s["id"] for s in target])}
            blind = [s for s in target if not mom.get(s["id"]) and s["id"] in judged_ids]
            if blind:
                log.info("re-judgment %s-%s: %d segment(s) have no screenshot on disk and keep their judgment (ids %s)",
                         dt.datetime.fromtimestamp(t0).strftime("%m-%d %H:%M"), dt.datetime.fromtimestamp(t1).strftime("%H:%M"),
                         len(blind), ", ".join(str(s["id"]) for s in blind[:30]))
                target = [s for s in target if s not in blind]
                if not target:
                    # nothing on the Mac can be re-judged here, so the pickups keep their judgments
                    # too; only one that has none yet is still judged
                    picks = [p for p in picks if any(g["key"] not in pk_have or pk_have[g["key"]]["switch_label"] is None
                                                     for g in p["segs"])]
                if not target and not picks:
                    return len(deferred)
        app_of = {s["id"]: s["app"] for s in segs}
        images: list[dict] = []
        cut = None
        kept: list[dict] = []
        room = IMAGE_BUDGET - CONTEXT_IMAGES_MIN
        for s in target:   # every target's captures first; a target that does not fit is left for the next call
            ims = [im for m in _seg_moments(mom, s) for im in _moment_images(m, "EVALUATE segment", s["app"])]
            if kept and len(images) + len(ims) > room:
                cut = s["clip_start"]
                break
            kept.append(s)
            images += ims
        if cut is not None:
            log.info("chunk %s: %d of %d segments fit the image budget (%d images); the rest from %s wait for the next call",
                     dt.datetime.fromtimestamp(t0).strftime("%m-%d %H:%M"), len(kept), len(target), len(images),
                     dt.datetime.fromtimestamp(cut).strftime("%H:%M:%S"))
            target = kept
            picks = [p for p in picks if p["start"] < cut]
            t1 = self.last_end = cut
        # A pickup: what was on the Mac's screen just before it and just after it. (Phone-only
        # calls used to spread 8 images over the whole Mac segment behind the pickup: in run
        # 13251 all 8 were from 01:28-01:35 and none bore on the 01:18 pickup.)
        kept_p: list[dict] = []
        pshots: dict = {}
        for p in picks:
            ims = []
            # The phone's own screen, when the capture job saw it (2026-11-23). These come first:
            # they are the only direct evidence of what the phone was used for.
            own, _ = _phone_images(self.cfg, p)
            if len(images) + len(own) > IMAGE_BUDGET:
                own = []
            ims += own
            for side, m in (("just before", _shots.nearest(conn, p["start"], True)), ("just after", _shots.nearest(conn, p["end"], False))):
                if m is not None and not any(i["path"] == m["active"]["path"] for i in images + ims):
                    for im in _moment_images(m, f"the Mac's screen {side} the phone pickup of {_t(p['start'])}, segment", app_of.get(m["segment_id"]) or "", both=not target):
                        ims.append(im)
            if len(images) + len(ims) > IMAGE_BUDGET:
                if not target and kept_p:      # a phone-only call: judge fewer pickups, as with segments
                    picks = kept_p
                    t1 = self.last_end = p["start"]
                    break
                ims = own                      # no room for the Mac's before/after: the phone's own screen stays
            kept_p.append(p)
            images += ims
        # Context segments: first and last capture each, nearest the stretch to judge first, as far
        # as the budget reaches. A window title can be wrong (run 13302: context lines titled as a
        # food-delivery page while the screen showed developer documentation), and only an image shows it.
        T0 = min([s["clip_start"] for s in target] + [p["start"] for p in picks])
        T1 = max([s["clip_end"] for s in target] + [p["end"] for p in picks])
        target_ids = {s["id"] for s in target}
        have_img = {im["path"] for im in images}
        ctx_segs = sorted((s for s in shown if s["id"] not in target_ids and mom.get(s["id"])),
                          key=lambda s: min(abs(s["clip_start"] - T1), abs(s["clip_end"] - T0)) if (s["clip_end"] <= T0 or s["clip_start"] >= T1) else 0.0)
        for s in ctx_segs:
            ms = mom[s["id"]]
            ims = [im for m in ([ms[0]] if len(ms) == 1 else [ms[0], ms[-1]])
                   for im in _moment_images(m, "context segment", s["app"], both=False) if im["path"] not in have_img]
            if len(images) + len(ims) > IMAGE_BUDGET:
                break
            images += ims
        content, shot_meta = _image_content(images)
        sent = {m["path"] for m in shot_meta}
        # Counted from what was really attached (a file can be withheld or gone by now), so a
        # PHONE line never claims a picture the model was not given.
        for m in shot_meta:
            pm = re.search(r"the iPhone's screen during PHONE id=P(\d+) ", m["label"])
            if pm:
                pshots[int(pm.group(1))] = pshots.get(int(pm.group(1)), 0) + 1
        with_img = {s["id"] for s in target if any(m["active"]["path"] in sent for m in mom.get(s["id"]) or [])}
        no_img = {s["id"]: _shots.why_none(conn, s["id"]) for s in target if s["id"] not in with_img}
        # -- the log ----------------------------------------------------------------------
        from .config import day_bounds, logical_date
        d0 = day_bounds(logical_date(t0))[0]
        day_text, entries = _day_block(conn, t0, T0, T1, ctx0, ctx1, now0)
        notes: dict = {}
        for s in target:
            seen_before = _last_seen(conn, s, d0, ctx0)
            notes[s["id"]] = ([seen_before] if seen_before else []) + (
                [f"screenshot: none for this segment ({no_img[s['id']]}). That says nothing about what the segment was."]
                if s["id"] in no_img else [])
        pick_keys = {p["key"] for p in picks}
        rows = (_log_lines(conn, segs, {i: f"EVALUATE id={i}: " for i in target_ids}, entries, history=history, notes=notes)
                + _phone_lines(conn, picks_all, pick_keys, pk_have, pshots))
        lines = [x[2] for x in sorted(rows, key=lambda x: (x[0], x[1]))]   # phone lines slotted in by time
        all_ids = [s["id"] for s in segs if s["duration"] > 0]
        # Hold-off (Rules doc): only for first-time judgments of activity under an hour old.
        can_hold = extra is None and bool(target) and time.time() - min(s["clip_end"] for s in target) < HOLD_MAX_S
        # phone-only calls: the rulings that resemble the Mac windows behind the pickups (they got none before)
        like = target or [s for s in shown if any(s["clip_start"] < p["end"] and s["clip_end"] > p["start"] for p in picks)]
        similar = self._similar_precedents(conn, like, pointers=True)
        n_ctx_img = sum(1 for m in shot_meta if "context segment" in m["label"])
        shots_note = (f"Screenshots attached above: {len(shot_meta)}"
                      + (f" ({len(shot_meta) - n_ctx_img} of the segments to judge or around the pickups, {n_ctx_img} of context segments)" if shot_meta else "")
                      + ". " + (("EVALUATE segments with NO screenshot: "
                                 + "; ".join(f"id={i} ({why})" for i, why in no_img.items())
                                 + ". A missing screenshot is missing evidence, not a sign of anything.") if no_img else
                                ("Every EVALUATE segment has at least one." if target else "")))
        prompt_text = (_call_header(t0, T0, T1, now0)
                       + "\n\nDAY SO FAR\n" + day_text
                       + ("\n\n" + similar if similar else "")
                       + "\n\n" + shots_note
                       + "\n\nActivity log (" + _t(ctx0) + " to " + _t(ctx1) + "; the lines marked EVALUATE are the ones to judge):\n" + "\n".join(lines)
                       + (f"\n\nIMPORTANT - {extra}" if extra else "")
                       + ("\n\nJudge every segment marked EVALUATE (all of them, by id)." if target else
                          "\n\nNo segment is marked EVALUATE: leave `segments` empty.")
                       + (" Judge every line marked EVALUATE PHONE, in `phone`." if picks else "")
                       + ("\n\nHolding is allowed for this chunk: if judging the LAST segments well depends on "
                          "what the person does after the activity shown here (for example a new piece of "
                          "work has only just begun and you cannot yet tell what it is), set hold_from to the "
                          "id of the first segment you would rather judge later, and leave it and everything "
                          "after it out. You will be shown them again in about 15 minutes with more activity "
                          "logged. Use this sparingly; most chunks need no hold." if can_hold else ""))
        content.append({"type": "text", "text": prompt_text})
        system_text = self._system_blocks(conn)[0]["text"]
        resp = llm.parse(self.client, site="evaluator", 
            model=self.cfg.claude_model, max_tokens=MAX_TOKENS,
            system=self._system_blocks(conn),
            messages=[{"role": "user", "content": content}],
            output_format=ChunkJudgement,
            # The configured effort. From 2026-08-31 to 2026-11-20 this said "low" outright while
            # config.toml asked for "high" ("accuracy is the only priority"), and the shadow
            # calls below, told the primary was "high", compared low against low.
            output_config={"effort": self.cfg.claude_effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default")
        cost = ((resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
                + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
                + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)
        n = n_held = n_phone = 0
        # for run_reevaluations: did this call produce judgments, a message for Eli, or nothing?
        self.last_answer = ("parsed" if resp.parsed_output is not None else
                            "contact" if resp.stop_reason == "contact_eli" else "none")
        if resp.stop_reason == "contact_eli" and target and extra is None:
            # The model asked for Eli and the task stops there: no retry. Without a hold the
            # next cycle (15 s later) would select the same chunk and send the same prompt again.
            conn.execute("INSERT OR REPLACE INTO eval_holds(segment_id, until, created) VALUES (?,?,?)",
                         (min(target, key=lambda s: s["clip_start"])["id"], time.time() + CONTACT_HOLD_S, time.time()))
            log.warning("chunk %s: the evaluator contacted Eli; this stretch is held for %d h",
                        dt.datetime.fromtimestamp(t0).strftime("%m-%d %H:%M"), CONTACT_HOLD_S // 3600)
        if resp.parsed_output is None and resp.stop_reason != "contact_eli" and extra is None and target:
            # No judgments and no message for Eli (cut off, unparsable). Nothing was stored, so
            # the next cycle sent the same chunk again, every 15 seconds, with the cost unseen
            # by the budget. The call is recorded; the chunk waits ten minutes; after three
            # empty answers its segments get a flagged default so the queue moves on.
            first = min(target, key=lambda s: s["clip_start"])
            with db.tx(conn):
                conn.execute("INSERT INTO eval_runs(t0, t1, n_segments, cost_usd, created, model) VALUES (?,?,?,?,?,?)",
                             (t0, t1, 0, cost, time.time(), self.cfg.claude_model))
                if _attempt(conn, f"eval.empty.{first['id']}") < 3:
                    conn.execute("INSERT OR REPLACE INTO eval_holds(segment_id, until, created) VALUES (?,?,?)",
                                 (first["id"], time.time() + 600, time.time()))
                else:
                    for s in target:
                        conn.execute(
                            "INSERT OR IGNORE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, created)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (s["id"], "other", "continuation", None, 1,
                             "(the evaluator gave no answer for this stretch in three tries; defaulted, please review)", time.time()))
            log.warning("chunk %s: no judgments came back (stop_reason %s)",
                        dt.datetime.fromtimestamp(t0).strftime("%m-%d %H:%M"), resp.stop_reason)
        if resp.parsed_output is not None:
            now = time.time()
            per_seg_cost = cost / max(len(target), 1)
            seg_starts = {s["id"]: s["clip_start"] for s in segs if s["duration"] > 0}
            seg_ends = {s["id"]: s["clip_end"] for s in segs if s["duration"] > 0}
            # Hold-off (Rules doc): the model may ask to judge the tail of the chunk later.
            hold_id = resp.parsed_output.hold_from if can_hold else None
            held_from = seg_starts.get(hold_id) if hold_id in target_ids else None
            with db.tx(conn):
                run_id = conn.execute(
                    "INSERT INTO eval_runs(t0, t1, n_segments, cost_usd, created, system, prompt, shots, response, model)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (t0, t1, 0, cost, now, system_text, prompt_text, json.dumps(shot_meta),
                     resp.parsed_output.model_dump_json(), self.cfg.claude_model)).lastrowid
                for j in resp.parsed_output.segments:
                    if j.segment_id not in target_ids:
                        continue
                    c = j.content.strip().lower()
                    sw = stored_switch(j.switch.strip().lower())  # the model says 'shift'
                    if c not in CONTENTS or sw not in SWITCHES:
                        log.warning("bad judgement for segment %d: %s/%s", j.segment_id, c, sw)
                        continue
                    if held_from is not None and seg_starts.get(j.segment_id, 0) >= held_from:
                        continue  # the model asked to judge this one later
                    foc = bool(j.focused)
                    sd = bool(j.self_distraction) and sw in ("interruption", "return")
                    nxt = bool(j.next_task) and sw == "interruption"
                    k = kind_of(sw, foc, sd)   # the older per-shift kind, derived
                    # Claude's discrete judgment as (pseudo-)probabilities, so it can be scored
                    # and weighted alongside the local models.
                    top_p = 0.55 if j.uncertain else 0.85
                    cp = {"content": seg_models._uniformish(CONTENTS, c, top_p),
                          "switch": seg_models._uniformish(SWITCHES, sw, top_p),
                          "kind": seg_models._uniformish(KINDS, k, top_p) if k else seg_models._uniformish(KINDS)}
                    seg_models.store_prediction(conn, j.segment_id, "claude", cp, per_seg_cost)
                    # Resolved judgment = accuracy-weighted ensemble of every model's output.
                    lp, _ = local.get(j.segment_id, ({}, None))
                    allp = dict(lp); allp["claude"] = cp
                    comb = {h: seg_models.combine_head({m: allp[m][h] for m in allp},
                                                       seg_models.weights_for(sc, h, list(allp)))
                            for h in HEADS}
                    rc = max(comb["content"], key=comb["content"].get)
                    # The stored switch is the evaluator's own. Its other fields (focused,
                    # self-distraction, the thread pointer, the note) are all about that switch,
                    # and when it was unsure the two local models could outvote it: 497 rows by
                    # 2026-11-20 said "continuation" under a note reading "Left X for Y", with
                    # the thread pointer thrown away. A disagreement is flagged for review instead.
                    rsw = sw
                    unc = int(j.uncertain or max(comb["content"].values()) < 0.5
                              or max(comb["switch"].values()) < 0.5
                              or max(comb["switch"], key=comb["switch"].get) != sw)
                    # Antecedents (thread pointers): stored raw from the evaluator, never
                    # ensembled - the local heads cannot represent "which thread". Ranked
                    # candidates (top + runner-up + third) with probabilities conditional on
                    # same-thread; a candidate must name a real, EARLIER segment in window.
                    cands = []
                    for cand in (j.antecedents or [])[:3]:
                        if (cand.segment_id in seg_starts and cand.segment_id != j.segment_id
                                and seg_starts[cand.segment_id] < seg_starts.get(j.segment_id, 0)):
                            cands.append((cand.segment_id, min(max(cand.p, 0.0), 1.0)))
                        else:
                            log.warning("dropping invalid antecedent %s for segment %d",
                                        cand.segment_id, j.segment_id)
                    # Rules doc: coming back to something after more than 10 minutes is a shift,
                    # not a return. The pointer says how long ago the old thread was last touched.
                    # Sorted first: the rule used to read the first candidate LISTED, and three
                    # returns by 2026-11-20 had a different one on top by probability.
                    cands.sort(key=lambda x: -x[1])
                    if rsw == "return" and cands and cands[0][0] in seg_ends:
                        # Measured from the last time the THREAD was touched, not from the segment the
                        # model happened to name: 37414 named a segment 10.0 minutes back though the
                        # thread had been in front 3 minutes before, and became a shift (2026-11-21).
                        # And going back to something by drifting is still a drift: the flag is kept.
                        if seg_starts[j.segment_id] - _thread_last_end(conn, cands[0][0], seg_starts[j.segment_id], seg_ends) > RETURN_MAX_S:
                            rsw = "interruption"
                            nxt = False
                            j.note = (j.note or "") + " [return after more than 10 minutes: stored as a shift]"
                    sub = None
                    if rsw == "interruption":
                        cands = []  # a new thread has no antecedent
                        g_of = j.grew_out_of
                        if (g_of is not None and not sd and g_of in seg_starts and g_of != j.segment_id
                                and seg_starts[g_of] < seg_starts.get(j.segment_id, 0)):
                            sub = g_of
                        elif g_of is not None:
                            log.warning("dropping grew_out_of %s for segment %d", g_of, j.segment_id)
                    else:
                        nxt = False
                        if rsw != "return":   # a return carries its own self-distraction judgment (always stored 0 until 2026-11-20)
                            sd = False
                    rk = kind_of(rsw, foc, sd)
                    cands.sort(key=lambda x: -x[1])
                    # The candidates' probabilities are conditional on same-thread and must sum
                    # to 1. The model sometimes returns one candidate carrying 0.2 (or three
                    # summing to 0.7); taken at face value that reads as doubt about the
                    # attachment and raises a pointless review question. Renormalize.
                    tot = sum(p for _, p in cands)
                    if tot > 0:
                        cands = [(i, p / tot) for i, p in cands]
                    elif cands:
                        # Every candidate at 0.0 is a malformed answer, not doubt: stored as is,
                        # the lone survivor read as "0% sure which thread" and raised a thread
                        # question about an obvious return to a meeting (segment 34907,
                        # 2026-11-16). Share the mass evenly: one candidate becomes certain,
                        # several stay a real tie.
                        cands = [(i, 1.0 / len(cands)) for i, _ in cands]
                    ante = cands[0][0] if cands else None
                    if cands:
                        comb["antecedent"] = {str(i): p for i, p in cands}
                    conn.execute(
                        "INSERT OR REPLACE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, probs, created, run_id, antecedent_id, hybrid, focused, self_distraction, next_task, sub_of)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (j.segment_id, rc, rsw, rk, unc, j.note, json.dumps(comb), now, run_id, ante, int(bool(j.hybrid)),
                         int(foc), int(sd), int(nxt), sub))
                    n += 1
                # Phone app segments: one row each, with the fields of a computer segment. `away`
                # is worked out here: is this time away from what was on the Mac (a shift, or
                # more of a shift), or part of it (a continuation of the Mac thread, a pass-through)?
                # A segment the model left out is stored as a continuation, so it is not queued again.
                pj = {j.phone_id: j for j in (resp.parsed_output.phone or [])}
                # earlier phone lines in view, so a return to an app from the pickup before attaches to it
                away_of: dict = {k: bool(r["away"]) for k, r in pk_have.items() if r["away"] is not None}
                # (start, app, away) of the same, for a return that names no usable line
                phone_seen = sorted((r["start"], r["app"], bool(r["away"])) for r in pk_have.values() if r["away"] is not None)
                for p in picks:
                    prev_away = False
                    for g in p["segs"]:
                        j = pj.get(g["key"])
                        psw = stored_switch(j.switch.strip().lower()) if j is not None else "continuation"
                        if psw not in SWITCHES:
                            psw = "continuation"
                        pc = j.content.strip().lower() if j is not None else "transition"
                        pc = pc if pc in CONTENTS else "other"
                        cont = (j.continues or "").strip() if j is not None else ""
                        m = re.fullmatch(r"[Pp](\d+)", cont)
                        if psw == "interruption":
                            away, cont = True, ""
                        elif m and int(m.group(1)) in away_of and int(m.group(1)) != g["key"]:
                            away = away_of[int(m.group(1))]        # attaches to an earlier phone line
                        elif cont.isdigit() and int(cont) in seg_starts:
                            away = False                           # attaches to a computer segment in view
                        else:
                            # No usable pointer: none, itself (run 13317), or an id that does not
                            # exist ("P1795045XXX", run 13277). A continuation goes on from the
                            # line before it. A return used to be stored as "back on the work"
                            # with its self-distraction flag dropped: run 13327, 74 s of a dating
                            # app, "browsing with no clear purpose", return + self-distraction,
                            # stored away=0 and no flag. It now takes `away` from the nearest
                            # earlier phone line of the same app in view, and is away in any case
                            # when the model called it a self-distraction or passive.
                            cont = ""
                            if psw == "continuation":
                                away = prev_away
                            else:
                                same = [a for st, app, a in phone_seen if app == g["app"] and st < g["start"]]
                                away = same[-1] if same else False
                                if (j is not None and j.self_distraction) or pc == "passive":
                                    away = True
                        if pc == "transition" and psw == "continuation" and not prev_away:
                            away = False                           # a pass-through is not a diversion
                        away_of[g["key"]] = prev_away = away
                        phone_seen.append((g["start"], g["app"], away))
                        psd = bool(j is not None and j.self_distraction and psw in ("interruption", "return") and away)
                        conn.execute(
                            "INSERT OR REPLACE INTO phone_evals(seg_key, pickup_key, start, end, app, switch_label, content,"
                            " focused, self_distraction, next_task, antecedent, away, rationale, created, run_id)"
                            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (g["key"], p["key"], g["start"], g["end"], g["app"], psw, pc,
                             int(bool(j.focused)) if j is not None else None, int(psd),
                             int(bool(j is not None and j.next_task and psw == "interruption")), cont or None, int(away),
                             j.note if j is not None else "(model omitted this phone segment; defaulted)", now, run_id))
                        n_phone += 1
                # A target the model omitted must not stall the queue forever: store an
                # explicit default the user can audit and correct.
                # ...and so must one whose answer was thrown out for a label outside the list: it used
                # to count as judged, nothing was stored, and the same call went out every cycle.
                judged = {r[0] for r in conn.execute(
                    f"SELECT segment_id FROM seg_evals WHERE run_id=? AND segment_id IN ({','.join('?' * len(target))})",
                    [run_id] + [s["id"] for s in target])} if target else set()
                if held_from is not None:
                    first_held = min((s for s in target if s["clip_start"] >= held_from), key=lambda s: s["clip_start"])
                    conn.execute("INSERT OR REPLACE INTO eval_holds(segment_id, until, created) VALUES (?,?,?)",
                                 (first_held["id"], now + HOLD_RETRY_S, now))
                    n_held = sum(1 for s in target if s["clip_start"] >= held_from)
                for s in target:
                    if held_from is not None and s["clip_start"] >= held_from:
                        continue
                    if s["id"] not in judged:
                        conn.execute(
                            # flagged for review: the default is not a judgment (21 such rows stood
                            # as "other / continuation" with no flag by 2026-11-20, e.g. 36227)
                            "INSERT OR IGNORE INTO seg_evals(segment_id, content, switch_label, interruption_kind, uncertain, rationale, created, run_id)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            (s["id"], "other", "continuation", None, 1, "(model omitted this segment; defaulted)", now, run_id))
                # a correction carried over from the person's ruling on a similar segment survives a new judgment
                from . import propagation
                propagation.reapply(conn, [s["id"] for s in target])
                conn.execute("UPDATE eval_runs SET n_segments=? WHERE id=?", (n + len(deferred), run_id))
        log.info("chunk %s-%s: %d/%d segments judged, %d deferred, %d held for later, %d phone segment(s) ($%.3f)",
                 dt.datetime.fromtimestamp(t0).strftime("%m-%d %H:%M"),
                 dt.datetime.fromtimestamp(t1).strftime("%H:%M"), n, len(target), len(deferred), n_held, n_phone, cost)
        # effort comparison: the same prompt at another effort level, compared + kept (efforts.py)
        if resp.parsed_output is not None and extra is None:
            from . import efforts as _eff

            def _call(eff):
                return llm.parse(self.client, site="evaluator_shadow", 
                    model=self.cfg.claude_model, max_tokens=MAX_TOKENS,
                    system=self._system_blocks(conn),
                    messages=[{"role": "user", "content": content}],
                    output_format=ChunkJudgement,
                    output_config={"effort": eff},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")

            def _cmp(a, b):
                pa = {j.segment_id: (j.content, stored_switch(j.switch), bool(j.focused)) for j in a.segments}
                pb = {j.segment_id: (j.content, stored_switch(j.switch), bool(j.focused)) for j in b.segments}
                keys = set(pa) & set(pb)
                if not keys:
                    return 0.0, "no overlapping segments"
                same = sum(1 for k in keys if pa[k] == pb[k])
                diffs = [f"id={k}: {pa[k]} vs {pb[k]}" for k in keys if pa[k] != pb[k]][:6]
                return same / len(keys), ("; ".join(diffs) or "all verdicts identical")

            _eff.run_shadows(conn, self.cfg, self.client, "segment_evaluator", str(run_id),
                             self.cfg.claude_effort, resp, _call, _cmp)
        return n + len(deferred) + n_phone


def refine_incoherent(conn, cfg: Config | None = None, max_spans: int = 1) -> int:
    """Model judgments are not sacred: when the coherence check catches a span containing two
    objects of attention, RE-EVALUATE its segments with that finding in hand and correct them
    (the user's own edits always take precedence in state derivation and are never the target).
    Bounded: a stretch already refined twice in 24h falls back to asking the user instead."""
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    rows = conn.execute("SELECT * FROM span_summaries WHERE coherent=0 AND refined=0"
                        " ORDER BY created DESC LIMIT ?", (max_spans,)).fetchall()
    n = 0
    for r in rows:
        conn.execute("UPDATE span_summaries SET refined=1 WHERE id=?", (r["id"],))
        # Counted in kv, not in span_summaries: re-summarising a span deletes its older rows,
        # so a count kept there stayed at 0 and the bound never fired (two spans were re-judged
        # five times each in 23 minutes on 2026-11-19).
        try:
            past = [x for x in json.loads(db.kv_get(conn, "refine.log") or "[]") if x[2] >= time.time() - 86400]
        except (ValueError, TypeError, IndexError):
            past = []
        prior = sum(1 for a, b, _ in past if a < r["end"] and b > r["start"])
        db.kv_set(conn, "refine.log", json.dumps(past + [[r["start"], r["end"], time.time()]]))
        if prior >= 2:
            log.info("refinement budget exhausted for %s - leaving for the user",
                     dt.datetime.fromtimestamp(r["start"]).strftime("%H:%M"))
            continue
        try:
            # The whole summary, not only its finding: the reviewer's title and description say
            # which object of attention it took the span for, and the bounds say which lines it read.
            extra = (f"a whole-span review looked at the stretch from {_t(r['start'])} to {_t(r['end'])} "
                     f"({_durtext(r['end'] - r['start'])}), which the app had assembled as ONE focus span from your "
                     f"judgments. It described the span as"
                     + (f" “{_one(r['title'])}”:" if r["title"] else ":") + f" “{_one(r['summary'])}” "
                     f"It then found that this is not one object of attention: {_one(r['issue'])} "
                     f"That review read the same log with a few screenshots; it is a second opinion, not a ruling. "
                     f"Re-judge with it in hand. If a NEW piece of work begins mid-stretch, "
                     f"mark the segment where it begins as switch='shift' (with focused=true "
                     f"if the new work is focused).")
            self_n = _rejudge_stretch(conn, r["start"] - 1, r["end"] + 1, extra)
            log.info("coherence-driven re-evaluation of %s-%s: %d segments re-judged",
                     dt.datetime.fromtimestamp(r["start"]).strftime("%H:%M"),
                     dt.datetime.fromtimestamp(r["end"]).strftime("%H:%M"), self_n)
            n += 1
        except Exception:
            log.exception("refinement failed")
    return n


def _rejudge_stretch(conn, a: float, b: float, extra: str) -> int:
    """Re-judge every settled segment starting in [a, b) with the note `extra`, in as many calls
    as the image budget needs (evaluate_chunk judges only as many segments as it can show; run
    13410 put 73 segments in one call with 8 images). Stops at the first call that brings no
    judgments back."""
    ev = _get()
    n, start = 0, a
    for _ in range(40):
        ev.last_answer = "parsed"
        n += ev.evaluate_chunk(conn, start, b, extra=extra)
        if ev.last_answer != "parsed" or ev.last_end >= b or ev.last_end <= start:
            break
        start = ev.last_end
    return n


def derive_kind(conn, cfg: Config | None = None, segment_id: int = 0, note: str | None = None) -> int:
    """The person ruled a segment a SEPARATE THREAD (switch='interruption') without choosing a
    kind - by design: thread identity is theirs, but whether that thread became a sustained
    object (focus_start) or not (detour) is derived from the recorded data. Re-judge the
    stretch with their ruling in hand. Their switch verdict is sacred (seg_reviews outranks
    field-by-field); only the derived kind shows through the review's NULL kind."""
    cfg = cfg or load_config()
    g = conn.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
    if not g:
        return 0
    t0 = g["start"] - 300
    t1 = min((g["end"] or g["start"]) + 2400, time.time() - 60)  # after-context decides the kind
    # The question they were answering, when the ruling came from one: the note alone was passed
    # (11 of 51 calls had one), and "separate thread" without the question says little about why.
    q = conn.execute("""SELECT * FROM review_questions WHERE segment_id=? AND status='answered'
                        ORDER BY answered_at DESC LIMIT 1""", (segment_id,)).fetchone()
    rv = conn.execute("SELECT created FROM seg_reviews WHERE segment_id=?", (segment_id,)).fetchone()
    asked = ""
    if q is not None and rv is not None and q["answered_at"] is not None and abs(q["answered_at"] - rv["created"]) < 60:
        asked = (f" The ruling was their answer to a review question. Context they were shown: “{_one(q['context'])}” "
                 f"Question: “{_one(q['question'])}” Their answer: “{_one(q['answer'])}”"
                 + (f" Their note on it: “{_one(q['note'])}”" if _one(q["note"]) and _one(q["note"]) != _one(note) else ""))
    extra = (f"The person has RULED that segment id={segment_id} began a SEPARATE thread from "
             f"what preceded it (their ruling is authoritative - do not relitigate it"
             + (f"; their note: {note}" if note else "") + ")." + asked + " "
             f"Your job is the rest: with that segment as switch='shift', re-judge the "
             f"affected segments - whether each was focused or unfocused, whether the shift "
             f"was a self-distraction or the next task in a task block, and the downstream "
             f"switches (continuations of the new thread, returns to the old one).")
    try:
        n = _rejudge_stretch(conn, t0, t1, extra)   # settled segments only; the first pass takes the rest
        log.info("kind derivation after thread ruling on segment %d: %d segments re-judged",
                 segment_id, n)
        return n
    except Exception:
        log.exception("kind derivation failed for segment %d", segment_id)
        return 0


CHAT_TOOLS = [
    {"name": "propose_judgment",
     "description": ("Propose the judgment of THIS segment that the person's clarification implies, for "
                     "them to confirm with one click. Call it when they have told you what was really "
                     "going on and no offered option (or your own judgment) captures it, or when they "
                     "correct you. Do NOT call it for a mere question, or when an existing option already "
                     "says the same thing (point them to that option instead). The label must be in "
                     "their own words, as a review-question option would read."),
     "input_schema": {"type": "object", "properties": {
         "label": {"type": "string", "description": "The answer as an option label, in the person's own terms, e.g. 'A window that was just open on my desktop, not looked at'."},
         "content": {"type": ["string", "null"], "enum": list(CONTENTS) + [None], "description": "Content label this asserts, or null to leave the content as judged."},
         "switch_label": {"type": ["string", "null"], "enum": list(SHOWN_SWITCHES) + [None], "description": "continuation | shift (a separate thread began) | return, or null to leave the thread relation as judged."},
         "focused": {"type": ["boolean", "null"], "description": "true = the person says their attention was focused in this segment; false = they say it was unfocused; null = no ruling on that."},
         "antecedent_id": {"type": ["integer", "null"], "description": "For continuation/return: the id of the segment this continues (from the log), else null."},
         "hybrid": {"type": ["boolean", "null"], "description": "true = the activity was done inside a live discussion; false = solo; null = no ruling."},
         # 2026-11-20: "that was me drifting, count it as a self-distraction" could not be recorded
         # from this chat, though self-distraction events are a headline number.
         "self_distraction": {"type": ["boolean", "null"], "description": "For a segment that is (or that this makes) a shift, or a return to a distraction: true = the person says they sought out stimulation or drifted there without a clear intent; false = they say they went there on purpose; null = no ruling on that."},
         "next_task": {"type": ["boolean", "null"], "description": "For a shift only: true = the person says this was the next task under the same task-processing intention; false = they say it was not; null = no ruling."},
         "focus_exception": {"type": ["boolean", "null"], "description": "true ONLY when the person explicitly asked that this segment count as independent focus although its content (correspondence with no Toggl entry) normally cannot. Never set it on your own judgment; null = no ruling."},
         "note": {"type": "string", "description": "One sentence, in the person's words, stored as the precedent future judgments see: what was really going on and why it is labelled this way."}},
         "required": ["label", "note"]}},
    {"name": "propose_rule",
     "description": ("Propose ONE general rule for the person's rules document, for them to approve. Only "
                     "when their clarification reveals a GENERAL lesson the current rules/precedents miss "
                     "(e.g. 'sub-15-second windows with no input are pass-throughs, whatever the page') - "
                     "never a restatement of this one case, and never something the rules already say."),
     "input_schema": {"type": "object", "properties": {
         "rule": {"type": "string", "description": "The rule, one or two sentences, in the second person as the rules doc is written ('If I ...')."}},
         "required": ["rule"]}},
]


FOLLOWUP_ASK = ("(Sent automatically when I answered; I did not type this.) My answer differs from your judgment "
                "of that segment. If my reason is not already clear from my answer or note, and knowing it would "
                "change how you judge similar cases later, ask me ONE short, concrete question that I can answer "
                "in a sentence. If the reason is already clear, reply with exactly: No question.")


def followup_question(conn, cfg: Config, question_id: int) -> dict:
    """After the person answers a review question against the app's judgment, let the evaluator
    - the replayed evaluation run, as in chat_reply - ask them one follow-up question (Eli,
    2026-11-19: "letting the model instance that did the evaluation ask follow up questions
    when I correct its judgements is fine"). An answer with its reason is a far better precedent
    than a bare click. Returns {} when the answer agreed with the app (no model call), else
    {"auto": the automatic prompt, "reply", "message_id", "proposal", "asked": bool}."""
    import json as _json
    q = conn.execute("SELECT * FROM review_questions WHERE id=?", (question_id,)).fetchone()
    if q is None or q["status"] != "answered" or q["answer_index"] is None:
        return {}
    try:
        o = _json.loads(q["options"])[q["answer_index"]]
    except (ValueError, IndexError):
        return {}
    if o.get("phone"):       # nothing was seen of the phone: there is no judgment to ask about
        return {}
    # The judgment as it stood BEFORE the answer. The answer can kick a re-judgment (derive_kind)
    # that finishes first; the live row then already agrees with the answer, and no follow-up was
    # asked although the person had corrected the app.
    from . import prompt_parts as _pp
    ev_row = _pp.judged_before(conn, o.get("segment_id") or q["segment_id"], q["answered_at"] or time.time())
    sd = o.get("self_distraction")
    differs = o.get("sample_verdict") == "disagree" or (ev_row is not None and bool(
        (o.get("content") and o["content"] != ev_row["content"])
        or (o.get("switch_label") and o["switch_label"] != ev_row["switch_label"])
        or (o.get("focused") is not None and ev_row["focused"] is not None and bool(o["focused"]) != bool(ev_row["focused"]))
        or (sd is not None and ev_row["self_distraction"] is not None and bool(sd) != bool(ev_row["self_distraction"]))))
    if not differs:
        return {}
    r = chat_reply(conn, cfg, q["segment_id"], [{"role": "user", "text": FOLLOWUP_ASK}], question_id)
    asked = not (r.get("reply") or "").strip().lower().startswith("no question")
    return dict(r, auto=FOLLOWUP_ASK, asked=asked)


def chat_reply(conn, cfg: Config, segment_id: int, messages: list[dict],
               question_id: int | None = None) -> dict:
    """Talk to "the one who made the decision": reconstitute the EXACT evaluation conversation
    (same system prompt, same user message with the same screenshots, its actual response), then
    continue it with the reviewer's questions - so mistakes can be triangulated and the
    instruction that would have prevented them identified. Falls back to an informed stand-in
    for segments judged by the local models (deferral) or before transcripts were kept."""
    import json as _json
    from .classifiers.claude_vision import _b64
    ev_row = conn.execute("SELECT * FROM seg_evals WHERE segment_id=?", (segment_id,)).fetchone()
    run = None
    if ev_row is not None and ev_row["run_id"]:
        run = conn.execute("SELECT * FROM eval_runs WHERE id=?", (ev_row["run_id"],)).fetchone()
    seg = conn.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
    who = f"segment id={segment_id} ({seg['app']} | {seg['title'] or ''})" if seg else f"segment id={segment_id}"
    ev = _get()
    if run is not None and run["system"] and run["response"]:
        system = [{"type": "text", "text": run["system"], "cache_control": {"type": "ephemeral"}}]
        content: list[dict] = []
        for m in _json.loads(run["shots"] or "[]"):
            b = _b64(m["path"])
            if b:
                content.append({"type": "text", "text": m["label"]})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}})
        content.append({"type": "text", "text": run["prompt"]})
        api_msgs: list[dict] = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": run["response"],
                                               "cache_control": {"type": "ephemeral"}}]}]
        preface = (f"[The person you monitor is now reviewing your judgments above, in particular "
                   f"{who}. Answer their questions about what you saw and why you judged as you did; "
                   f"if you now think you were wrong, say so plainly and say what instruction or "
                   f"information would have led you to the right judgment. You cannot change any "
                   f"record yourself: when they tell you what was really going on, call propose_judgment "
                   f"(and propose_rule for a general lesson) - the proposal appears as a button they "
                   f"click to apply; otherwise nothing you say changes anything.]\n\n")
    else:
        # No transcript: an informed stand-in built from what actually decided this segment.
        preds = conn.execute("SELECT model, p_content, p_switch, p_kind FROM seg_predictions WHERE segment_id=?",
                             (segment_id,)).fetchall()
        ptxt = "\n".join(f"  {p['model']}: content={p['p_content']} switch={p['p_switch']}" for p in preds) or "  (none stored)"
        from . import prompt_parts as _pp     # (this printed the pre-2026-11-19 `kind=` and left out focused / self-distraction)
        vtxt = ("resolved: " + _pp.judged_text(_pp.judged_before(conn, segment_id, time.time()))) if ev_row else "(no evaluation recorded)"
        system = [{"type": "text", "text": ev._system_blocks(conn)[0]["text"], "cache_control": {"type": "ephemeral"}}]
        api_msgs = []
        preface = (f"[No verbatim transcript exists for the evaluation of {who} - it was judged "
                   f"before transcripts were kept, or deferred to the local (non-LLM) models. "
                   f"What is on record:\n{vtxt}\nPer-model probability outputs:\n{ptxt}\n"
                   f"Help the person triangulate whether the judgment was right.]\n\n")
    if question_id is not None:
        # A follow-up on a review question: the person is looking at that question and its
        # options, so the evaluator must know exactly what was asked.
        qrow = conn.execute("SELECT * FROM review_questions WHERE id=?", (question_id,)).fetchone()
        if qrow is not None:
            opts = "\n".join(f"  {n}. {o.get('label')}" for n, o in enumerate(_json.loads(qrow["options"]), 1))
            if qrow["status"] == "none":
                # Every option was rejected: the job is to find the reading that fits and hand
                # it back as a proposal (which becomes a new option when applied).
                preface += (f"[The person was shown this review question about that segment and answered "
                            f"NONE OF THE ABOVE - no option describes what actually happened"
                            + (f"; their note: \"{qrow['note']}\"" if qrow["note"] else "")
                            + f". Work out with them what it really was (ask one concrete thing if you "
                            f"must), then call propose_judgment with the answer in their words so it can "
                            f"be applied. Be brief.\n"
                            f"Context shown: {qrow['context']}\n"
                            f"Question: {qrow['question']}\nRejected options:\n{opts}]\n\n")
            elif qrow["status"] == "answered":
                preface += (f"[The person was shown this review question about that segment and has answered it.\n"
                            f"Context shown: {qrow['context']}\n"
                            f"Question: {qrow['question']}\nOptions:\n{opts}\n"
                            f"Their answer: {qrow['answer']}"
                            + (f"\nTheir note: \"{qrow['note']}\"" if qrow["note"] else "") + "]\n\n")
            else:
                preface += (f"[The person was shown this review question about that segment and is asking "
                            f"a follow-up before answering it. Help them decide: say concretely what you "
                            f"saw (screens, titles, times), what would distinguish the options, and which "
                            f"option you lean toward and how sure you are. Be brief.\n"
                            f"Context shown: {qrow['context']}\n"
                            f"Question: {qrow['question']}\nOptions:\n{opts}]\n\n")
    from . import chat_tools as _ct
    from .config import logical_date
    day = logical_date(seg["start"]) if seg else logical_date()
    extra = _chat_now(conn, cfg, seg, run) if seg else []
    replays = _ct.replay(conn, cfg, _ct.stored_traces(conn, messages), day)
    for i, m in enumerate(messages):
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = (m.get("text") or "").strip()
        if role == "user" and preface:
            # the first thing the person says carries what is new since the run: screenshots of
            # the segment and its neighbours, and the state of the record now
            api_msgs.append({"role": "user", "content": extra + [{"type": "text", "text": preface + text}]} if extra
                            else {"role": "user", "content": preface + text})
            preface = ""
            continue
        if role == "assistant" and m.get("id") in replays:
            api_msgs += replays[m["id"]]
        api_msgs.append({"role": role, "content": text})
    model = run["model"] if run is not None and run["model"] else cfg.claude_model
    system = llm.with_preamble(system)
    tools = CHAT_TOOLS + _ct.READ_TOOLS + llm.CONTACT_TOOLS
    from .classifiers.claude_vision import _cost
    costs, said = [], []   # one entry per request: a note_to_eli call is answered and the model asked again

    def call(msgs):
        # (16,000 and the configured effort: at 1024 the model's thinking could use it all and leave
        # no answer, as the span chat did on 2026-11-20)
        resp = ev.client.messages.create(model=model, max_tokens=16000, system=system, messages=msgs,
                                         tools=tools, output_config={"effort": cfg.claude_effort})
        costs.append(_cost(resp.usage))
        said.append("".join(b.text for b in resp.content if b.type == "text").strip())
        return resp

    def proposed(resp, proposal):
        for b in resp.content:
            if b.type != "tool_use":
                continue
            if b.name == "propose_judgment":
                j = dict(b.input)
                if j.get("content") not in CONTENTS:
                    j["content"] = None
                j["switch_label"] = stored_switch(j.get("switch_label"))
                if j["switch_label"] not in SWITCHES:
                    j["switch_label"] = None
                if j["switch_label"] not in ("continuation", "return"):
                    j["antecedent_id"] = None
                if not (j.get("label") or "").strip():
                    continue
                yn = lambda k: None if j.get(k) is None else bool(j[k])
                proposal["judgment"] = {"label": j["label"].strip(), "content": j["content"],
                                        "switch_label": j["switch_label"], "antecedent_id": j.get("antecedent_id"),
                                        "hybrid": j.get("hybrid"), "focused": j.get("focused"),
                                        # the two shift judgments mean nothing on a continuation (edits.write_seg_review)
                                        "self_distraction": None if j["switch_label"] == "continuation" else yn("self_distraction"),
                                        "next_task": yn("next_task") if j["switch_label"] in (None, "interruption") else None,
                                        "focus_exception": True if j.get("focus_exception") is True else None,
                                        "note": (j.get("note") or "").strip()}
            elif b.name == "propose_rule" and (b.input.get("rule") or "").strip():
                proposal["rule"] = b.input["rule"].strip()

    # A proposal made in the same turn as a note is not taken from that turn: the model is told it
    # was not run and makes it again in the answer that follows, which is the one read below.
    # Read tools (2026-11-20: get_screenshots and the others of chat_tools) are run and the model
    # asked again, up to CHAT_ROUNDS times; a proposal made beside a lookup is kept.
    proposal: dict = {}
    trace: list[dict] = []
    read_chars = 0
    for _ in range(CHAT_ROUNDS):
        api_msgs, resp = llm.with_notes("evaluator_chat", model, system, api_msgs, call(api_msgs), call, tools)
        rec = llm.contacted("evaluator_chat", model, system, api_msgs, resp, tools)
        if rec or resp.stop_reason == "refusal":
            break
        proposed(resp, proposal)
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not any(u.name in _ct.READ_NAMES for u in uses):
            break
        results, calls = [], []
        for u in uses:
            if u.name in _ct.READ_NAMES:
                out = (_ct.run(conn, cfg, u.name, dict(u.input), day) if read_chars <= _ct.TURN_TEXT_MAX else
                       "not run: the lookups in this reply have already returned the most one reply may read.")
                read_chars += _ct.size(out)
            elif u.name == llm.NOTE_TOOL_NAME:
                out = llm.NOTE_RESULT_TEXT
            else:
                out = "Noted: it will be shown to the person as a button when this reply ends. Nothing is changed until they click it."
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
            calls.append(_ct.trace_entry(u.id, u.name, dict(u.input), out))
        trace.append({"said": said[-1] if said else "", "calls": calls})
        api_msgs = api_msgs + [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]},
                               {"role": "user", "content": results}]
    text = (llm.contact_notice(rec) if rec
            else "(Claude declined to answer that.)" if resp.stop_reason == "refusal"
            else "\n\n".join(t for t in said if t))
    if rec:
        proposal = {}
    if not text:
        text = "(proposal below)" if proposal else "(no reply)"
    from .chat import log_exchange
    mid = log_exchange(conn, "evaluator", str(segment_id), messages, text, sum(costs), model,
                       question_id, ({**proposal, _ct.TRACE_KEY: trace} if trace else proposal) or None)
    return {"reply": text, "proposal": proposal or None, "message_id": mid}


CHAT_ROUNDS = 8          # requests in one evaluator-chat reply when it looks things up
CHAT_IMAGES = 30         # screenshots added after the replayed run: the segment under review and its neighbours


def _chat_now(conn, cfg, seg, run) -> list[dict]:
    """What a chat about a segment is given beyond the replayed evaluation run (2026-11-20).

    The replay shows what the evaluator saw THEN. It often had no image of the segment itself
    (12 of 27 chat turns: segment 33153, "What did I say to Atlas when I returned to it?" - "I
    can't tell you - I don't have a screenshot of that Atlas segment"), and nothing newer than
    the run: 16 of 27 turns were on segments re-judged since, and the person's rulings, the
    corrections carried over from them and changes to the Rules doc were all invisible. So:
    every chosen capture moment of the segment and the first and last of the segment before
    and after it, and a "since you judged" text: the log of the stretch as it stands now
    (current labels, rulings with notes, marks left by propagation), the rulings and
    carried-over corrections made since the run, earlier chat about the segment, and the lines
    of the Rules doc that were not in the run's instructions. Content blocks for the first
    user message."""
    from . import prompt_parts as pp, shots as _shots
    from .classifiers.claude_vision import _rules_doc
    from .summaries import verdict_text
    sid = seg["id"]
    end = seg["end"] or seg["start"]
    since = run["created"] if run is not None else None
    lo, hi = (min(run["t0"], seg["start"]) - 300, max(run["t1"], end) + 300) if run is not None else (seg["start"] - 1200, end + 1200)
    lines, rows = pp.log_lines(conn, cfg, lo, hi, marks={sid: f">>> THE SEGMENT UNDER REVIEW id={sid}: "}, dated=True)
    live = [g for g in rows if not g["idle"]]
    k = next((i for i, g in enumerate(live) if g["id"] == sid), None)
    near = [] if k is None else [g for g in (live[k - 1] if k > 0 else None, live[k + 1] if k + 1 < len(live) else None) if g]
    mom = _shots.moments(conn, [sid] + [g["id"] for g in near])
    own = pp.seg_images(mom, seg, "the segment under review,")
    if len(own) > CHAT_IMAGES - 4:
        own = pp.fit(pp.seg_images(mom, seg, "the segment under review,", both=False), CHAT_IMAGES - 4)
    images = list(own)
    for g in near:
        ms = mom.get(g["id"]) or []
        for m_ in ([ms[0]] if len(ms) == 1 else [ms[0], ms[-1]] if ms else []):
            images += _moment_images(m_, "the segment just " + ("before" if g["clip_start"] < seg["start"] else "after") + " it,", g["app"] or "", both=False)
    content, meta = pp.image_blocks(images[:CHAT_IMAGES])
    ids = [g["id"] for g in rows]
    qs = ",".join("?" * len(ids)) or "NULL"
    parts = [f"[What follows was put together now, {_t(time.time(), '%Y-%m-%d %H:%M')}, and was NOT part of your evaluation above. "
             f"Screenshots just attached: {len(meta)} - " + (f"{len(own)} of the segment under review (id={sid})" if own else
             f"none of the segment under review ({_shots.why_none(conn, sid)})")
             + f", the rest of the segment before and the segment after it. The get_screenshots tool shows any other segment or time.]"]
    if since is not None:
        rul = conn.execute(f"SELECT * FROM seg_reviews WHERE segment_id IN ({qs}) AND created > ? ORDER BY created", ids + [since]).fetchall()
        parts.append(f"SINCE YOU JUDGED (your evaluation above was made {_t(since, '%a %Y-%m-%d %H:%M')}):")
        parts.append("The person's rulings on segments of this stretch made since then: " + ("none." if not rul else "\n" + "\n".join(
            f"  id={r['segment_id']}, ruled {_t(r['created'], '%Y-%m-%d %H:%M')}: {verdict_text(r) or '(no field set)'}"
            + (f"; their note: “{_one(r['note'])}”" if _one(r["note"]) else "") for r in rul)))
        cur = conn.execute("SELECT created, run_id FROM seg_evals WHERE segment_id=?", (sid,)).fetchone()
        again = conn.execute("SELECT COUNT(*) FROM eval_runs WHERE t0 <= ? AND t1 >= ? AND created > ? AND response IS NOT NULL",
                             (seg["start"], seg["start"], since)).fetchone()[0]
        if again:
            parts.append(f"{again} later evaluation run(s) have covered the segment under review since the one replayed above.")
        doc = _rules_doc()
        old = run["system"] or ""
        added = [ln for ln in doc.splitlines() if ln.strip() and ln.strip() not in old]
        parts.append("Their rules document has not changed since your instructions above were written." if not added else
                     "Their rules document has changed since. Lines in it now that were not in your instructions above:\n"
                     + "\n".join("  " + ln for ln in added))
    prop = conn.execute(f"SELECT * FROM seg_propagated WHERE segment_id IN ({qs}) ORDER BY created", ids).fetchall()
    if prop:
        parts.append("Corrections carried over to segments of this stretch from the person's rulings on OTHER segments (the app's "
                     "propagation judge decided the ruling applies; they are part of the labels below):\n" + "\n".join(
                         f"  id={r['segment_id']} from their ruling on id={r['from_segment_id']} ({_t(r['created'], '%Y-%m-%d %H:%M')}): "
                         + ("flagged for review: does that ruling apply here?" if r["verdict"] == "uncertain" else "applied: " + (verdict_text(r) or "(no field)"))
                         + f" - “{_one(r['note'])}”" for r in prop))
    parts.append("THE STRETCH AS THE RECORD STANDS NOW (the labels in force, whoever set them; \"RULED BY THE PERSON\" marks their "
                 "own rulings, with their notes):\n" + "\n".join(lines))
    parts.append(pp.LOG_LEGEND)
    earlier = conn.execute("SELECT * FROM review_questions WHERE segment_id=? AND status IN ('answered', 'unsure', 'none') ORDER BY created", (sid,)).fetchall()
    for q in earlier:
        try:
            opts = json.loads(q["options"] or "[]")
        except ValueError:
            opts = []
        parts.append(f"A review question about this segment that the person has dealt with ({q['status']}):\n" + pp.question_text(q, opts, q["answer_index"]))
    return content + [{"type": "text", "text": "\n\n".join(parts)}]


class AuditFlag(BaseModel):
    segment_id: int
    field: str = Field(description="Which judgment looks wrong: content, switch, or focused.")
    suggested: str = Field(description="The label you would give instead.")
    reason: str = Field(description="One short sentence.")


class DayAudit(BaseModel):
    flags: list[AuditFlag] = Field(description="ONLY segments whose judgment looks wrong. Empty list if the day looks right.")


AUDIT_PROMPT = """This is a second look at judgments already made, not a first judgment. Below is one \
piece of a day of this person's activity, in the same log format as an evaluation call (section 7 of \
your instructions), with the judgment each segment holds: "judged: ..." and the note after "// note:". \
Some were made by cheap local models (their note says "deferred to local models").

The lines marked AUDIT are the ones to check; screenshots of each are attached where a file exists. \
Lines marked (context ...) are there to read the AUDIT lines against, as are the PHONE lines and the \
DAY SO FAR block; do not flag them. Lines marked RULED BY THE PERSON are settled and are never to be \
flagged. Nothing in this log is marked EVALUATE and no hold-off applies.

List ONLY the AUDIT segments whose content, switch or focused judgment looks wrong, with what you \
would say instead. These get flagged for the person to review, so precision matters more than recall \
- do not nitpick defensible calls. An empty list is the expected answer for a clean piece.

{header}

DAY SO FAR
{day}

{shots}

Activity log ({a} to {b}):
{log}"""

AUDIT_CTX_S = 15 * 60      # log shown on each side of an audited piece
AUDIT_PIECES_PER_PASS = 3  # pieces audited per classifier cycle; the rest of the day follows in later cycles


def daily_audit(conn, cfg: Config | None = None, max_pieces: int = AUDIT_PIECES_PER_PASS) -> int:
    """Once per day: an LLM sweep over yesterday's resolved judgments; suspected errors are
    flagged (uncertain=1) for the user's review queue - labels are never silently changed.

    The day is audited in pieces, each small enough that every audited segment's screenshots fit
    one call (IMAGE_BUDGET), with the same log lines the evaluator reads. Until 2026-11-20 the
    whole day went into one call with 8 screenshots (a median day has 407 segments and 1,153
    capture moments: over 98% of segments unseen), no idle or phone lines, no signals, no Toggl
    list, notes cut at 80 characters, and flags spent blind on segments the person had already
    ruled on. Progress is kept in kv ("audit.progress.<day>") so a day spread over several
    cycles, or interrupted, goes on where it stopped; audit_runs gets its row when the day is done."""
    from . import shots as _shots, stats
    from .config import logical_date
    cfg = cfg or load_config()
    if not cfg.daily_audit:
        return 0
    day = logical_date() - dt.timedelta(days=1)
    key = day.isoformat()
    if conn.execute("SELECT 1 FROM audit_runs WHERE day=?", (key,)).fetchone():
        return 0
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    if time.time() < float(db.kv_get(conn, f"audit.retry_after.{key}") or 0):
        return 0   # a piece failed or came back empty: ten minutes before the next try
    t0, t1 = stats.day_bounds(day)
    try:
        prog = json.loads(db.kv_get(conn, f"audit.progress.{key}") or "{}")
    except (ValueError, TypeError):
        prog = {}
    prog = {"to": float(prog.get("to", t0)), "flags": int(prog.get("flags", 0)), "cost": float(prog.get("cost", 0.0)),
            "pieces": int(prog.get("pieces", 0))}
    segs = [s for s in stats.labelled_segments(conn, t0, t1, phone=False) if s["duration"] > 0]
    ruled = {r[0] for r in conn.execute("SELECT segment_id FROM seg_reviews")}
    # to audit: judged by the app, and not ruled on by the person (a flag there was thrown away afterwards)
    todo = [s for s in segs if not s["idle"] and not s.get("neutral") and s.get("source") == "eval" and s["id"] not in ruled]
    n = 0
    ev = None
    for _ in range(max_pieces):
        rest = [s for s in todo if s["clip_start"] >= prog["to"]]
        if not rest:
            break
        mom = _shots.moments(conn, [s["id"] for s in rest[:400]])
        piece, images = [], []
        for s in rest:
            ims = [im for m in _seg_moments(mom, s) for im in _moment_images(m, "AUDIT segment", s["app"])]
            if piece and len(images) + len(ims) > IMAGE_BUDGET:
                break
            piece.append(s)
            images += ims
            if len(piece) >= 400:
                break
        nxt = rest[len(piece)]["clip_start"] if len(piece) < len(rest) else t1
        a, b = piece[0]["clip_start"], piece[-1]["clip_end"]
        lo, hi = max(t0, a - AUDIT_CTX_S), min(t1, b + AUDIT_CTX_S)
        content, shot_meta = _image_content(images)
        sent = {m["path"] for m in shot_meta}
        ids = {s["id"] for s in piece}
        no_img = {s["id"]: _shots.why_none(conn, s["id"]) for s in piece
                  if not any(m["active"]["path"] in sent for m in mom.get(s["id"]) or [])}
        day_text, entries = _day_block(conn, a, a, b, lo, hi, time.time(), mark="AUDIT")
        notes = {i: [f"screenshot: none for this segment ({why}). That says nothing about what the segment was."] for i, why in no_img.items()}
        rows = _log_lines(conn, [s for s in segs if s["clip_end"] > lo and s["clip_start"] < hi],
                          {i: f"AUDIT id={i}: " for i in ids}, entries, notes=notes)
        try:
            from . import phone as _phone
            picks = _phone.pickups(cfg, lo, hi)
            pk_have = {r["seg_key"]: r for r in conn.execute("SELECT * FROM phone_evals WHERE start >= ? AND start < ?", (lo - 3600, hi + 3600))}
            rows += _phone_lines(conn, picks, set(), pk_have)
        except Exception:
            log.exception("daily audit: phone lines unavailable")
        shots_note = (f"Screenshots attached above: {len(shot_meta)}. "
                      + ("AUDIT segments with NO screenshot: " + "; ".join(f"id={i} ({why})" for i, why in no_img.items())
                         + ". A missing screenshot is missing evidence, not a sign of anything." if no_img else
                         "Every AUDIT segment has at least one."))
        text = AUDIT_PROMPT.format(header=_call_header(a, a, b, time.time(), what="audit"), day=day_text, shots=shots_note,
                                   a=_t(lo), b=_t(hi), log="\n".join(x[2] for x in sorted(rows, key=lambda x: (x[0], x[1]))))
        content.append({"type": "text", "text": text})
        ev = ev or _get()
        try:
            resp = llm.parse(ev.client, site="audit",
                model=cfg.claude_model, max_tokens=MAX_TOKENS,
                system=ev._system_blocks(conn),
                messages=[{"role": "user", "content": content}],
                output_format=DayAudit,
                output_config={"effort": cfg.claude_effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default")
        except Exception:
            # (an error here used to escape: the same call again 15 s later, with no count, and
            # the coherence pass skipped for that cycle)
            log.exception("daily audit %s piece %s: the call failed - next try in ten minutes", key, _t(a))
            db.kv_set(conn, f"audit.retry_after.{key}", str(time.time() + 600))
            return n
        cost = ((resp.usage.input_tokens or 0) * PRICE_INPUT + (resp.usage.output_tokens or 0) * PRICE_OUTPUT
                + (getattr(resp.usage, "cache_creation_input_tokens", 0) or 0) * PRICE_CACHE_WRITE
                + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0) * PRICE_CACHE_READ)
        prog["cost"] += cost
        if resp.stop_reason == "contact_eli":
            # The model asked for Eli: that stops the task, and the task is this day's audit. (The
            # piecewise loop used to carry on with the next pieces: 24 more calls in a simulation.)
            with db.tx(conn):
                conn.execute("INSERT OR REPLACE INTO audit_runs(day, n_flags, cost_usd, created) VALUES (?,?,?,?)",
                             (key, prog["flags"], prog["cost"], time.time()))
                conn.execute("DELETE FROM kv WHERE key=?", (f"audit.progress.{key}",))
            log.warning("daily audit %s: the model contacted Eli at piece %s; the audit of this day stops there", key, _t(a))
            return n
        if resp.parsed_output is None and _attempt(conn, f"audit.fail.{key}.{int(a)}") < 3:
            log.warning("daily audit %s piece %s: no answer (%s) - next try in ten minutes", key, _t(a), resp.stop_reason)
            db.kv_set(conn, f"audit.progress.{key}", json.dumps(prog))   # the cost is kept; the piece is asked again
            db.kv_set(conn, f"audit.retry_after.{key}", str(time.time() + 600))
            return n
        if resp.parsed_output is not None and prog["pieces"] == 0:
            # Effort shadows for the first piece of the day only: one comparison a day, as before.
            from . import efforts as _eff

            def _acall(eff, content=content):
                return llm.parse(ev.client, site="audit_shadow",
                    model=cfg.claude_model, max_tokens=MAX_TOKENS,
                    system=ev._system_blocks(conn),
                    messages=[{"role": "user", "content": content}],
                    output_format=DayAudit,
                    output_config={"effort": eff},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")

            def _acmp(x, y):
                fa = {(f.segment_id, f.field) for f in x.flags}
                fb = {(f.segment_id, f.field) for f in y.flags}
                union = fa | fb
                if not union:
                    return 1.0, "both flagged nothing"
                same = len(fa & fb) / len(union)
                return same, f"flags primary={sorted(fa)} vs {sorted(fb)}"

            _eff.run_shadows(conn, cfg, ev.client, "daily_audit", key, cfg.claude_effort,
                             resp, _acall, _acmp, prompt_text=text, always=True)   # once a day: always compared
        with db.tx(conn):
            for f in (resp.parsed_output.flags if resp.parsed_output is not None else []):
                if f.segment_id not in ids:
                    continue
                conn.execute("UPDATE seg_evals SET uncertain=1, rationale = COALESCE(rationale,'') || ? WHERE segment_id=?",
                             (f" | daily audit: {f.field} looks like '{f.suggested}' - {f.reason}", f.segment_id))
                n += 1
            prog.update(to=nxt, flags=prog["flags"] + sum(1 for f in (resp.parsed_output.flags if resp.parsed_output is not None else [])
                                                         if f.segment_id in ids), pieces=prog["pieces"] + 1)
            db.kv_set(conn, f"audit.progress.{key}", json.dumps(prog))
        log.info("daily audit %s: piece %d, %s-%s, %d segment(s), %d image(s) ($%.3f)", key, prog["pieces"], _t(a), _t(b),
                 len(piece), len(shot_meta), cost)
    if not [s for s in todo if s["clip_start"] >= prog["to"]]:
        with db.tx(conn):
            conn.execute("INSERT OR REPLACE INTO audit_runs(day, n_flags, cost_usd, created) VALUES (?,?,?,?)",
                         (key, prog["flags"], prog["cost"], time.time()))
            conn.execute("DELETE FROM kv WHERE key=?", (f"audit.progress.{key}",))
        log.info("daily audit %s: done, %d flag(s) in %d piece(s) ($%.3f)", key, prog["flags"], prog["pieces"], prog["cost"])
    return n


def _invalidate_derived(conn, t0: float, t1: float, questions: bool, summaries: bool) -> tuple[int, int]:
    """Drop what was built ON TOP of judgments that just changed, so it gets rebuilt from the new
    ones: open review questions for re-judged segments (answered ones stay - the person's ruling
    stands), and span summaries overlapping the stretch (the summarizer regenerates any span
    without one, and the coherence check re-runs on the new summary)."""
    nq = ns = 0
    if questions:
        nq = conn.execute("""DELETE FROM review_questions WHERE status IN ('open', 'declined') AND segment_id IN
                             (SELECT id FROM segments WHERE start >= ? AND start < ?)""", (t0, t1)).rowcount
    if summaries:
        ns = conn.execute("DELETE FROM span_summaries WHERE start < ? AND end > ?", (t1, t0)).rowcount
    return nq, ns


WRONG_DISPLAY_REASON = "screenshots shown were of the wrong display"
# The history re-judgment (Eli, 2026-11-20: "Yes all of it"). From 2026-08-31 until the restart
# at 03:34 on 2026-11-20 the evaluator ran at "low" effort whatever config.toml said; everything
# judged before that restart is judged again at the configured effort, newest day first.
LOW_EFFORT_REASON = "judged at low effort"
LOW_EFFORT_FIXED_AT = 1795163657.0   # 2026-11-20 03:34:17 Eastern, as an epoch: not a clock time in whatever zone the Mac is in
REEVAL_BACKOFF_S = 300.0         # after a re-evaluation call fails or comes back empty
HISTORY_EVERY_S = 60.0           # at least this long between two history calls (kv rejudge.history.every_s overrides)
PHONE_REASON = "phone data arrived"
PHONE_LOOKBACK_S = 3 * 86400     # the phone pushes its log hours to a day late
PHONE_MERGE_S = 300.0            # pickups this close together are re-judged in one call
_phone_seen = (None, 0.0)        # (phone.through, when last checked)


LOCK_GLANCE_NOTE = "A look at the lock screen without unlocking: an interruption, never a self-distraction (set by rule, not judged)."


def store_lock_glance(conn, p: dict) -> None:
    """Eli, 2026-11-19: "just glancing at my lock screen on my phone is an interruption, but
    isn't a self-distraction event." Decided here, in code: a shift, never a self-distraction."""
    conn.execute("INSERT OR REPLACE INTO phone_evals(seg_key, pickup_key, start, end, app, switch_label, content, focused,"
                 " self_distraction, next_task, away, rationale, created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (p["key"], p["key"], p["start"], p["end"], "Lock screen", "interruption", "other", 1, 0, 0, 1,
                  LOCK_GLANCE_NOTE, time.time()))


def _dur(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.0f}m"


def queue_phone_rejudge(conn, cfg: Config | None = None) -> int:
    """Phone data arrives after the Mac activity it overlaps has been judged (Eli, 2026-11-19:
    "re-judge when the phone data lands"). Queue a re-judgment (reeval_queue) of each stretch
    that holds a pickup with no judgment, so the evaluator sees it again with the pickup in
    view. Only stretches that contain a pickup, and only behind the point the first-pass
    evaluation has reached - pickups ahead of it are judged in the first pass. Each queued
    pickup gets a placeholder row in phone_evals (no label) so it is queued once."""
    global _phone_seen
    from . import phone
    cfg = cfg or load_config()
    thr = phone.through(cfg)
    now = time.time()
    if thr is None or (thr == _phone_seen[0] and now - _phone_seen[1] < 1800):
        return 0
    _phone_seen = (thr, now)
    horizon = now - LAZY_S
    row = conn.execute("""
        SELECT MIN(g.start) FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
        WHERE g.start >= ? AND g.start < ? AND g.idle = 0 AND g.neutral = 0
          AND COALESCE(g.end, g.start) > g.start AND e.segment_id IS NULL""",
        (now - PHONE_LOOKBACK_S, horizon)).fetchone()
    frontier = min(row[0], horizon) if row and row[0] else horizon
    have = {r[0] for r in conn.execute("SELECT seg_key FROM phone_evals WHERE start >= ?",
                                       (now - PHONE_LOOKBACK_S - 3600,))}
    new = [p for p in phone.pickups(cfg, now - PHONE_LOOKBACK_S, frontier)
           if p["end"] < frontier and any(g["key"] not in have for g in p["segs"])]
    windows: list[list[float]] = []
    with db.tx(conn):
        for p in new:
            on_mac = conn.execute("SELECT 1 FROM segments WHERE start < ? AND COALESCE(end, ?) > ? LIMIT 1",
                                  (p["end"], now, p["start"])).fetchone()
            for g in p["segs"]:
                conn.execute("INSERT OR IGNORE INTO phone_evals(seg_key, pickup_key, start, end, app, rationale, created)"
                             " VALUES (?,?,?,?,?,?,?)",
                             (g["key"], p["key"], g["start"], g["end"], g["app"],
                              None if on_mac else "(the Mac was not recording: not judged)", now))
            if not on_mac:
                continue
            if p["lock_screen"]:      # settled by rule: nothing to ask the model
                store_lock_glance(conn, p)
                continue
            a, b = p["start"] - 30, p["end"] + 30
            if windows and a - windows[-1][1] <= PHONE_MERGE_S and b - windows[-1][0] <= CHUNK_S:
                windows[-1][1] = max(windows[-1][1], b)
            else:
                windows.append([a, b])
        for a, b in windows:
            conn.execute("INSERT INTO reeval_queue(t0, t1, reason, created) VALUES (?,?,?,?)",
                         (a, b, PHONE_REASON, now))
    if windows:
        log.info("phone data through %s: %d pickup(s) in %d stretch(es) queued for judgment",
                 dt.datetime.fromtimestamp(thr).strftime("%m-%d %H:%M"), len(new), len(windows))
    return len(windows)


def queue_wrong_display_rejudge(conn) -> int:
    """Queue for re-judgment every stretch whose current judgment came from a run that showed
    the evaluator a segment only on a display that did NOT hold its active window (the
    row-sampling bug fixed 2026-11-16, see shots.py). eval_runs.shots records what each run
    saw. Safe to rerun: a re-judged segment belongs to a new run that saw the right display,
    and a stretch already waiting in the queue is not added twice."""
    import json as _json
    from .shots import _display_of
    seg_of = {r["path"]: r["segment_id"] for r in conn.execute("SELECT segment_id, path FROM segment_shots")}
    active = {r["id"]: {d for d in (_display_of(r["first_screenshot"]), _display_of(r["last_screenshot"])) if d is not None}
              for r in conn.execute("SELECT id, first_screenshot, last_screenshot FROM segments")}
    owner = {r["segment_id"]: r["run_id"] for r in conn.execute("SELECT segment_id, run_id FROM seg_evals")}
    bad = []
    for r in conn.execute("SELECT id, t0, t1, shots FROM eval_runs WHERE shots IS NOT NULL ORDER BY t0"):
        shown: dict[int, set] = {}
        for s in _json.loads(r["shots"] or "[]"):
            if s["path"] in seg_of:
                shown.setdefault(seg_of[s["path"]], set()).add(_display_of(s["path"]))
        if any(active.get(seg) and not (ds & active[seg]) and owner.get(seg) == r["id"] for seg, ds in shown.items()):
            bad.append((r["t0"], r["t1"]))
    windows: list[list[float]] = []
    for t0, t1 in bad:
        if windows and t0 - windows[-1][1] < 60:
            windows[-1][1] = max(windows[-1][1], t1)
        else:
            windows.append([t0, t1])
    n = 0
    for t0, t1 in windows:
        if conn.execute("SELECT 1 FROM reeval_queue WHERE done_at IS NULL AND t0 <= ? AND t1 >= ?", (t0, t1)).fetchone():
            continue
        conn.execute("INSERT INTO reeval_queue(t0, t1, reason, created) VALUES (?,?,?,?)",
                     (t0, t1, WRONG_DISPLAY_REASON, time.time()))
        n += 1
    log.info("queued %d stretch(es) for re-judgment: %s", n, WRONG_DISPLAY_REASON)
    return n


def queue_history_rejudge(conn) -> int:
    """Queue every logical day judged before LOW_EFFORT_FIXED_AT for re-judgment (reeval_queue,
    one row per day), once. Started by hand (python -m focus_monitor rejudge-history --start):
    Eli asked for another bug hunt before the history is judged again. The judgments as they stand are copied to seg_evals_low_effort
    first: a re-judgment replaces the row, and the old answers are data worth keeping."""
    from .config import day_bounds, logical_date
    if db.kv_get(conn, "rejudge.history.queued") is not None:
        return 0
    first = conn.execute("SELECT MIN(g.start) FROM segments g JOIN seg_evals e ON e.segment_id = g.id").fetchone()[0]
    if first is None:
        return 0
    # the first stretch judged after the fix marks where the low-effort judgments end
    end = conn.execute("""SELECT MIN(g.start) FROM segments g JOIN seg_evals e ON e.segment_id = g.id
                          WHERE e.created >= ? AND g.start >= ?""",
                       (LOW_EFFORT_FIXED_AT, LOW_EFFORT_FIXED_AT - 2 * 3600)).fetchone()[0] or LOW_EFFORT_FIXED_AT
    # Only days that have screenshot files. Before the oldest file every re-judgment would be
    # made from window titles alone (10,876 of 33,656 older segments are a bare app name) and
    # would replace 21,953 judgments, many made with the screenshots that existed then. The
    # date comes from the files (2026-10-27 13:35 when this was written); within that first day,
    # segments from before the first file keep their judgments (evaluate_chunk).
    shots_from = _first_screenshot_ts()
    if shots_from is None:
        log.warning("history re-judgment not queued: no screenshot files found")
        return 0
    first = max(first, shots_from)
    n = 0
    with db.tx(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS seg_evals_low_effort AS SELECT * FROM seg_evals WHERE created < ?",
                     (LOW_EFFORT_FIXED_AT,))
        day = logical_date(first)
        while day_bounds(day)[0] < end:
            t0, t1 = day_bounds(day)
            conn.execute("INSERT INTO reeval_queue(t0, t1, reason, created) VALUES (?,?,?,?)",
                         (t0, min(t1, end), LOW_EFFORT_REASON, time.time()))
            n += 1
            day += dt.timedelta(days=1)
        db.kv_set(conn, "rejudge.history.queued", f"{n} days, {dt.datetime.now():%Y-%m-%d %H:%M}")
    log.info("history re-judgment queued: %d days", n)
    return n


def _first_screenshot_ts() -> float | None:
    """When the oldest capture file still on disk was taken (full size or archive), from the
    file names, which begin with the capture time in milliseconds."""
    from .config import ARCHIVE_DIR, SCREENSHOT_DIR
    import os
    best = None
    for d in (SCREENSHOT_DIR, ARCHIVE_DIR):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for nm in names:
            head = nm.split("_", 1)[0]
            if head.isdigit() and len(head) >= 12 and nm.endswith(".jpg"):
                ts = int(head) / 1000.0
                best = ts if best is None or ts < best else best
    return best


def history_status(conn) -> dict:
    """Where the history re-judgment stands (python -m focus_monitor rejudge-history)."""
    rows = conn.execute("SELECT * FROM reeval_queue WHERE reason=? ORDER BY t0", (LOW_EFFORT_REASON,)).fetchall()
    left = [r for r in rows if r["done_at"] is None]
    return {"days": len(rows), "days_left": len(left),
            "working_on": dt.datetime.fromtimestamp(left[-1]["done_to"] or left[-1]["t0"]).strftime("%Y-%m-%d %H:%M") if left else None,
            "paused": db.kv_get(conn, "rejudge.history.paused") == "1",
            "calls": int(db.kv_get(conn, "rejudge.history.calls") or 0),
            "skipped_chunks": sum(int(db.kv_get(conn, f"reeval.skipped.{r['id']}") or 0) for r in rows),
            "waiting": (None if not left else "paused" if db.kv_get(conn, "rejudge.history.paused") == "1"
                        else "the subscription is not available" if not llm.subscription_available(load_config()) else None),
            "last_call": db.kv_get(conn, "rejudge.history.last")}


def _history_may_run(conn) -> bool:
    """History work is paced and never spills onto the API key: it waits while it is paused,
    while the subscription is at its limit, and for HISTORY_EVERY_S after the last call."""
    if db.kv_get(conn, "rejudge.history.paused") == "1":
        return False
    if not llm.subscription_available(load_config()):   # at its limit, or the backend is set to "api"
        return False
    every = float(db.kv_get(conn, "rejudge.history.every_s") or HISTORY_EVERY_S)
    return time.time() - float(db.kv_get(conn, "rejudge.history.last") or 0) >= every


def _attempt(conn, key: str) -> int:
    """Count one more failed attempt at something, and return the total so far."""
    n = int(db.kv_get(conn, key) or 0) + 1
    db.kv_set(conn, key, str(n))
    return n


def run_reevaluations(conn, cfg: Config | None = None, max_chunks: int = 1) -> int:
    """Re-judge stretches whose upstream data changed after they were evaluated (reeval_queue -
    e.g. a Toggl entry the person corrected later). Same evaluator, same context, plus a note
    saying why; the person's own edits still outrank the result. Each re-judged chunk retires
    the open questions written from the old judgments; when the whole stretch is done its span
    summaries are dropped so they are rewritten from the new judgments."""
    cfg = cfg or load_config()
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    done = 0
    if time.time() < float(db.kv_get(conn, "reeval.retry_after") or 0):
        return 0
    for _ in range(max_chunks):
        # Newest first: recent days are the ones still under review. The history re-judgment
        # comes after everything else, and only when it may run (_history_may_run).
        q = conn.execute("SELECT * FROM reeval_queue WHERE done_at IS NULL ORDER BY (reason = ?), t0 DESC LIMIT 1",
                         (LOW_EFFORT_REASON,)).fetchone()
        if not q:
            break
        history = q["reason"] == LOW_EFFORT_REASON
        if history and not _history_may_run(conn):
            break
        call_at = time.time()
        horizon = time.time() - LAZY_S
        start = max(q["t0"], q["done_to"] or q["t0"])
        end = min(q["t1"], horizon)
        if end - start < 30:   # nothing (left) to re-judge
            _, ns = _invalidate_derived(conn, q["t0"], q["t1"], questions=False, summaries=True)
            conn.execute("UPDATE reeval_queue SET done_at=? WHERE id=?", (time.time(), q["id"]))
            continue
        c1 = end if end - start <= CHUNK_S * 1.5 else start + CHUNK_S
        if q["reason"] == PHONE_REASON:
            extra = ("the phone log for this window arrived AFTER the computer activity was judged: the "
                     "PHONE lines are new. Judge each one marked EVALUATE PHONE, in the context of the "
                     "activity around it. No computer segment is up for judgment in this call.")
        elif history:
            extra = ("your earlier judgment of this window was made in a quick pass at low effort. Judge "
                     "every segment marked EVALUATE again, carefully, from the log and the screenshots. "
                     "Everything marked 'earlier quick-pass judgment' and 'quick-pass note', on the lines to "
                     "judge and on context lines alike, is from that quick pass: a hint at most, never a "
                     "reason to keep a judgment the evidence does not support. Lines marked RULED BY THE "
                     "PERSON are not from the quick pass and stand.")
        elif q["reason"] == WRONG_DISPLAY_REASON:
            # (it used to say "The screenshots above are of the display that held the active
            # window" also when none was attached: 30 runs on 2026-11-17)
            extra = ("your earlier judgment of this window was made from screenshots of the WRONG "
                     "display: the person had two monitors and you were shown the one they were not "
                     "working on. The screenshots attached to this call are labelled: 'active display' is "
                     "the one that held the window each segment is about. Re-judge every segment marked "
                     "EVALUATE from them; where your earlier judgment (including the notes after '// "
                     "note:') rested on what the other screen showed, change it.")
        else:
            extra = (f"the {q['reason']} for this window AFTER your earlier judgment of it - the Toggl "
                     f"entries listed above are the corrected ones and are AUTHORITATIVE: if a screenshot "
                     f"shows a Toggl timer with a different name or still running, that is the uncorrected "
                     f"state and must be ignored. Re-judge every segment against the listed entries; where "
                     f"your earlier judgment (including the notes after '// note:') relied on the old "
                     f"entries, change it." + _toggl_change_note(conn, q["id"]))
        phone_q = q["reason"] == PHONE_REASON
        ev = _get()
        ev.last_answer = "parsed"   # (a chunk with nothing to send returns before the call)
        try:
            llm.only_subscription(history)   # a history call that cannot run on the subscription fails; it never goes to the API key
            try:
                n = ev.evaluate_chunk(conn, start, c1, extra=extra, phone_only=phone_q, history=history)
            finally:
                llm.only_subscription(False)
            if ev.last_answer == "parsed" and start < ev.last_end < c1:
                c1 = ev.last_end   # the call judged only as many segments as it could show; go on from there
            if history:
                # Outside the daily budget: 2,500 history calls must not stop today's judging
                # (spend_today leaves these runs out). The pacing clock counts real calls only.
                made = conn.execute("UPDATE eval_runs SET budget_exempt=1 WHERE created >= ? AND t0=? AND t1<=?",
                                    (call_at, start, c1)).rowcount
                if made:
                    db.kv_set(conn, "rejudge.history.last", str(time.time()))
                    db.kv_set(conn, "rejudge.history.calls", str(int(db.kv_get(conn, "rejudge.history.calls") or 0) + 1))
        except Exception:
            log.exception("re-evaluation failed (%s)", dt.datetime.fromtimestamp(start))
            ev.last_answer = "failed"
        skipped = False
        if ev.last_answer != "parsed":
            # No judgments came back, so nothing was re-judged: the old judgments and the
            # questions written from them stand. Wait five minutes before any re-evaluation
            # (an outage used to fail every 15 seconds). An empty answer is counted per chunk:
            # after three, or when the model asked for Eli (that stops the task: no retry),
            # this chunk alone is passed over. The count used to be per queue row, and three
            # failures closed a whole day as "done" with nothing re-judged.
            db.kv_set(conn, "reeval.retry_after", str(time.time() + REEVAL_BACKOFF_S))
            log.warning("re-evaluation of %s got no judgments (%s)", dt.datetime.fromtimestamp(start), ev.last_answer)
            if history and not llm.subscription_available(load_config()):
                break   # the subscription hit its limit during the call: not this chunk's failure
            if ev.last_answer == "failed":
                break   # an error (offline, the SDK down) is retried for as long as it lasts: no chunk is given up for it
            if ev.last_answer != "contact" and _attempt(conn, f"reeval.fail.{q['id']}.{int(start)}") < 3:
                break
            skipped = True
            n = 0
            if start < getattr(ev, "last_end", 0) < c1:
                c1 = ev.last_end   # pass over only the segments that were sent, not the whole 20 minutes
            db.kv_set(conn, f"reeval.skipped.{q['id']}", str(int(db.kv_get(conn, f"reeval.skipped.{q['id']}") or 0) + 1))
        # (a phone re-judgment leaves the segments' judgments, and so their questions, as they were)
        nq = 0
        if not skipped:
            nq, _ = _invalidate_derived(conn, start, c1, questions=not phone_q, summaries=False)
        finished = c1 >= end
        if finished:
            # ensure_summaries rewrites only the last 7 days: an older day's span summaries
            # would be deleted for good, so they are left (written from the earlier judgments).
            recent = q["t1"] >= time.time() - 6 * 86400
            _, ns = _invalidate_derived(conn, q["t0"], q["t1"], questions=False, summaries=recent)
            conn.execute("UPDATE reeval_queue SET done_to=?, done_at=? WHERE id=?", (c1, time.time(), q["id"]))
        else:
            ns = 0
            conn.execute("UPDATE reeval_queue SET done_to=? WHERE id=?", (c1, q["id"]))
        if skipped:
            break
        log.info("re-evaluated %s-%s (%s): %d segments, %d stale question(s) retired%s",
                 dt.datetime.fromtimestamp(start).strftime("%m-%d %H:%M"),
                 dt.datetime.fromtimestamp(c1).strftime("%H:%M"), q["reason"], n, nq,
                 f", {ns} span summary(ies) dropped for rewrite" if finished else "")
        done += 1
    return done


def _toggl_change_note(conn, queue_id: int) -> str:
    """What changed in Toggl, for the note of a "Toggl entries corrected" re-judgment, as far as
    it was recorded (toggl.queue_reevaluation keeps the entry before and after from 2026-11-20;
    for stretches queued earlier nothing is known and nothing is said)."""
    try:
        rows = json.loads(db.kv_get(conn, f"reeval.detail.{queue_id}") or "[]")
    except (ValueError, TypeError):
        rows = []
    def one(e):
        if not e:
            return None
        return (f"{_t(e[0], '%m-%d %H:%M:%S')}-" + (_t(e[1]) if e[1] is not None else "running (no stop)")
                + f" “{_one(e[2])}”" + (f" | project: {e[3]}" if e[3] else "") + (f" | tags: {e[4]}" if e[4] else ""))
    out = []
    for d in rows:
        old, new = one(d.get("old")), one(d.get("new"))
        out.append(f"added after your judgment: {new}" if old is None else
                   f"deleted since: {old}" if new is None else f"was: {old}; is now: {new}")
    return (" What changed, as the app recorded it (the entry as it was when you judged, then as it is now): "
            + " / ".join(dict.fromkeys(out)) + ".") if out else ""


def repair_outvoted(conn) -> int:
    """One-off (2026-11-20): stored judgments whose switch is not the one the evaluator gave,
    because the local models outvoted it (see evaluate_chunk), get the evaluator's own switch
    back, with the fields that follow from it re-derived from its stored answer
    (eval_runs.response). Rows changed since by a propagated correction, or by the 10-minute
    return rule, are left alone. The old values are kept in seg_evals_outvoted."""
    conn.execute("CREATE TABLE IF NOT EXISTS seg_evals_outvoted (segment_id INTEGER PRIMARY KEY, switch_label TEXT,"
                 " interruption_kind TEXT, self_distraction INTEGER, next_task INTEGER, antecedent_id INTEGER,"
                 " uncertain INTEGER, repaired REAL)")
    rows = conn.execute("""SELECT e.*, g.start AS g_start, r.response FROM seg_evals e
                           JOIN segments g ON g.id = e.segment_id JOIN eval_runs r ON r.id = e.run_id
                           WHERE r.response IS NOT NULL AND COALESCE(e.rationale,'') NOT LIKE '%feedback on segment %'
                             AND COALESCE(e.rationale,'') NOT LIKE '%[return after more than 10 minutes%'
                           ORDER BY e.run_id""").fetchall()
    parsed: dict = {}
    n = 0
    for e in rows:
        if e["run_id"] not in parsed:
            try:
                parsed = {e["run_id"]: {x["segment_id"]: x for x in json.loads(e["response"]).get("segments", [])}}
            except (ValueError, TypeError, KeyError, AttributeError):
                parsed = {e["run_id"]: {}}
        j = parsed[e["run_id"]].get(e["segment_id"])
        if j is None:
            continue
        sw = stored_switch(str(j.get("switch") or "").strip().lower())
        if sw not in SWITCHES or sw == e["switch_label"]:
            continue
        ante = None
        if sw != "interruption":
            ante = e["antecedent_id"]
            if ante is None:   # dropped when the row was stored as a shift
                for cand in sorted(j.get("antecedents") or [], key=lambda x: -(x.get("p") or 0)):
                    a = conn.execute("SELECT start, end FROM segments WHERE id=?", (cand.get("segment_id"),)).fetchone()
                    if a is not None and a["start"] < e["g_start"]:
                        ante = cand["segment_id"]
                        break
            if sw == "return" and ante is not None:
                a = conn.execute("SELECT end FROM segments WHERE id=?", (ante,)).fetchone()
                if a is not None and a["end"] is not None and e["g_start"] - a["end"] > RETURN_MAX_S:
                    continue   # the 10-minute rule would have stored this one as a shift anyway
        if "focused" in j:     # judged with the 2026-11-19 fields
            sd = bool(j.get("self_distraction")) and sw in ("interruption", "return")
            nxt = bool(j.get("next_task")) and sw == "interruption"
            kind = kind_of(sw, bool(j.get("focused")), sd)
        else:
            kind = (j.get("interruption_kind") if j.get("interruption_kind") in KINDS else "detour") if sw == "interruption" else None
            sd, nxt = kind == "self_distraction", False
        conn.execute("INSERT OR IGNORE INTO seg_evals_outvoted VALUES (?,?,?,?,?,?,?,?)",
                     (e["segment_id"], e["switch_label"], e["interruption_kind"], e["self_distraction"],
                      e["next_task"], e["antecedent_id"], e["uncertain"], time.time()))
        # (the flag stays as it was: raising it on all 492 rows asked 339 new questions, most about September)
        conn.execute("UPDATE seg_evals SET switch_label=?, interruption_kind=?, self_distraction=?, next_task=?,"
                     " antecedent_id=? WHERE segment_id=?",
                     (sw, kind, int(sd), int(nxt), ante, e["segment_id"]))
        n += 1
    return n


def run_pending(conn, cfg: Config | None = None, max_chunks: int = 1, lookback_s: float | None = None) -> int:
    """Evaluate the oldest unevaluated stretch that is at least LAZY_S old. Called from the
    classifier loop; also usable for backfill with a longer lookback and more chunks."""
    cfg = cfg or load_config()
    if db.kv_get(conn, "migrated.outvoted") is None:
        with db.tx(conn):
            db.kv_set(conn, "migrated.outvoted", str(repair_outvoted(conn)))
    if db.spend_today(conn) >= cfg.daily_budget_usd * cfg.hard_budget_multiple:
        return 0
    default_scope = lookback_s is None
    if lookback_s is None:
        # Default scope: the current logical day (user 2026-09-01) and the one before it. With
        # today alone, whatever was still unjudged at the 4am rollover was never judged: the
        # last minutes before sleep, or a night of failed calls (162 segments in 30 days).
        from .config import day_bounds, logical_date
        lookback_s = time.time() - day_bounds(logical_date() - dt.timedelta(days=1))[0]
    horizon = time.time() - LAZY_S
    done = 0
    for _ in range(max_chunks):
        row = conn.execute("""
            SELECT g.start, g.id FROM segments g
            LEFT JOIN seg_evals e ON e.segment_id = g.id
            WHERE g.start >= ? AND g.start < ? AND g.idle = 0 AND g.neutral = 0
              AND COALESCE(g.end, g.start) > g.start
              AND COALESCE(g.end, g.start) <= ? AND e.segment_id IS NULL
            ORDER BY g.start LIMIT 1""",
            (time.time() - lookback_s, horizon, horizon)).fetchone()
        if not row or row[0] is None:
            break
        # The evaluator asked to judge this segment later (hold-off): wait until then. Everything
        # after it waits too, so spans are never assembled across an unjudged gap.
        hold = conn.execute("SELECT until FROM eval_holds WHERE segment_id=?", (row[1],)).fetchone()
        if hold and hold["until"] > time.time():
            break
        # Wait for a batch. Each call carries the same large context (rules, rulings, screenshots
        # of the surroundings) however little it judges, and this loop runs every cycle: without
        # the wait it judged each segment the moment it turned LAZY_S old, one or two per call
        # (median chunk 1.2 minutes, about 600 calls a day; found 2026-11-23). So: go when BATCH_S
        # of activity is ready, or when nothing has happened on the Mac since the horizon (the
        # person has left, and what is pending is all there will be for now).
        if row[0] > horizon - BATCH_S and conn.execute(
                "SELECT 1 FROM segments WHERE idle = 0 AND neutral = 0 AND COALESCE(end, ?) > ? LIMIT 1",
                (time.time(), horizon)).fetchone():
            break
        t0 = row[0] - 1
        t1 = min(t0 + CHUNK_S, horizon)
        try:
            if _get().evaluate_chunk(conn, t0, t1) == 0:
                break  # nothing judged: don't spin on the same window
        except Exception:
            log.exception("chunk evaluation failed (%s)", dt.datetime.fromtimestamp(t0))
            break
        done += 1
    if default_scope and done == 0 and db.kv_get(conn, "backfill.unjudged") is None:
        # One-off (2026-11-20): segments left unjudged at a 4am rollover before the look-back
        # above existed (162 in the last 30 days). Judged one chunk per quiet cycle, oldest
        # first, until none is left.
        back = 31 * 86400
        done = run_pending(conn, cfg, max_chunks=1, lookback_s=back)
        left = conn.execute("""SELECT COUNT(*) FROM segments g LEFT JOIN seg_evals e ON e.segment_id = g.id
                               WHERE g.start >= ? AND g.start < ? AND g.idle = 0 AND g.neutral = 0
                                 AND COALESCE(g.end, g.start) > g.start AND e.segment_id IS NULL""",
                            (time.time() - back, time.time() - lookback_s)).fetchone()[0]
        if left == 0:   # (an idle-cycle limit here called it finished after 50 minutes offline)
            db.kv_set(conn, "backfill.unjudged", f"finished {dt.datetime.now():%Y-%m-%d %H:%M}, {left} left")
            log.info("backfill of unjudged segments finished (%d left)", left)
    return done

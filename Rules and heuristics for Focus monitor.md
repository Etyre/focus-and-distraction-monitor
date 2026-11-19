[These rules should take priority over other documentation in this folder]

Hand-coded rules may veto, never affirm; preserve the doc's logical form and scope exactly; audit existing code whenever a principle lands.

### Definitions:

* A **segment** is a period of time during which a particular window and tab is active on my computer.
* A **switch** is the transition between segments.
* A **thread** is a series of segments during which the user's attention is directed toward a single project, task, intention, or idea. A thead can be composed of a single segment, but it will usually include multiple segments as the user switches tabs and windows while doing the same task or executing on the same intention.
* A **focus span** is a period of 10 minutes or greater in which my attention is directed toward a single project, task, intention, or idea (plus the following criteria).
  * A focus span always starts with at least 10 consecutive minutes, uninterrupted by diversions.
  * Focus spans are always independent work, not interactive. Chatting in Signal never counts toward a focus span (though drafting a long message that I then send can - that is creative work).
  * That I’m in a focus span does not mean that I’m only using one app, or only staying on one webpage. I will often switch between different apps and pages while still being engaged in the same intention.
  * A focus span ends...
    * when my computer has been idle for ten minutes (in which case, the span is retroactively made to end when the idle period started) or
    * if my attention has been interrupted and I’ve shifted my attention to a different task, project, or object for 5 minutes (in which case, the span is retroactively shortened to the start of the interruption, which may or may not be the start of a new focus span).
  * Attention interruptions that are less than 5 minutes long do not break the focus span.
  * Once a focus span has started, all time on the span's thread counts as focus time, even if it looks like I'm not making progress: I may still be thinking or processing. The model still judges each segment as focused or unfocused. If more than 20% of a span was judged unfocused, or it contains one continuous unfocused stretch of 10 minutes or more, the app asks me whether that time was really unfocused and should break the span. (added 2026-11-19; the app does not ask this yet)
* An **attention-interruption event** or a **diversion** is an instance when my focus shifts from the object of a focus span to something unrelated.
  * Looking at my phone should count as a diversion, unless what I'm looking at is clearly related to my current attentional thread.
  * Whether a segment is focused or unfocused is a separate question from whether it is a diversion. If I'm writing and I quickly jump to Amazon to buy a specific item, that segment is focused, and it is also a diversion from the writing span.
* A **self-distraction event** is a shift in which I seek out stimulation or distraction without a clear intent.
  * For instance:
    * Absent-mindedly opening lesswrong and scrolling the front page (though this requires some judgement, since I will also go to lessWrong to look up quotes or to read particular posts.
    * Opening a new tab to go to a blocked website.
    * Absentmindedly opening bumble.
    * When I open Amazon without any particular objective and browse.
    * Opening my phone to look at a random app, unrelated to what I'm doing.
* A **bounce-away** is a self-distraction event that occurs in a focus span.
* A **passive consumption span** is a period when I'm passively consuming or browsing: scrolling LessWrong, watching YouTube, reading comics, reading blogposts without taking notes.
  * This is important to track mainly because passive consumption time should never count toward a focus span.
* A **correspondence span** is a period when I'm communicating with people in writing: talking on Signal, responding to email, replying to messages or comments.
  * Correspondence is work, and is tracked in its own right, but it is interactive rather than independent, so it never counts toward a focus span on its own, unless I'm deliberately drafting a long message instead of engaged in a back and forth, which can count as a focused span.
  * Correspondence processed inside a focused task block (under a running toggl entry) is part of that focused task span.
  * A live voice or video call is a meeting, not correspondence.
* Focus span subtypes:
  * A **focused task span** is a special kind of focus span, in which my attention is focused on executing on a series of taks without interuption.
    * Focused task spans are the main exception to the rule that a focus span entails having my attention on only one task / project / object / intention. In a focused task span, I'm switching between different tasks, but I'm maintaining an overall intention of executing on tasks without interuption or distraction.
    * Discerning the difference between focused tasks processing or just doing tasks in a more lackadaisical and less focused way (tagged as "unfocused") requires judgement.
    * If executing on a focused task block, I’ll usually be operating from a todo list, or my project tracker feed, referring back to that list or feed between tasks. Alternatively, I might be going back to my daily page for explicit metacognition about which task to do next.
    * Computer activity can only count as Focused Task processing if there's a toggl entry running.
      * It doesn't have to be the *same* continuous toggl entry. It could be that there's one there's one for processing email, followed by another for doing taxes, followed by another for returning amazon packages. Or maybe there's a single toggl entry labled "task processing". But if there's there's not a running toggl entry, it can't be focused task work.
      * That does not imply the converse: just because a toggl entry is running does not make task segments focused. Judgement is required to evalate if a series of task segments are focused, unfocused, or distracted. 
      * Note that the absensece of a toggl entry disquailifes a span from being a *Focused* task span. But individual segments can and should still be tagged in the "task" category, regardless of whether there's a toggl running. Toggl is required as evidence of deliberate focus, not as evidence of task-nature.
    * **How focused task blocks are put together.** Attentional threads are tracked at the level of the individual task. When I finish one task and move to the next, that is a shift, because my attention moved to a new object. On each shift, the model additionally evaluates: is this the next task under the same overall intention of processing tasks? If yes, the shift stays inside the block. The app then joins the threads linked this way into one focused task block, and treats that block as a single focus span. A shift that is not the next task (a glance at Twitter, a random question to Claude) is a diversion from the block: it is an attention-interruption event if it lasts less than 5 minutes, and it ends the block if it lasts 5 minutes or more. The model judges each shift; the app does the joining, and applies the hard rule that a block can only count as a focused task span if a Toggl entry is running. (added 2026-11-19; the app does not work this way yet)
    * Focused task spans will often include email correspondence. Correspondence that happens inside a task block (under a running toggl entry) is part of the focused task span, not a separate correspondence span; the segments are still tagged "correspondence".
  * A **creative work span** is a sub-type of focus span in which I’m creating something: an essay, a video. etc.
    * Creative work spans and focused task spans are mutually exclusive.
  * A **reading span** is a sub-type of focus span in which I'm reading  posts, papers, or books. This counts as a focus span only if I am actively reading: either making comments or taking notes. Most often, when reading I will take notes in parellel in Roam or LogSeq.
    * A reading span can lead directly into a creative span, if the notes I'm writing develop into a writing an essay. In this case, the whole block should be counted as one uninterrupeted focus span, even if part of that focus span is reading and the other part is creative. This is an exception to the general rule that task switching implies the end of a focus span, because I'm *not* switching intentions or changing the focus of my attention. Those are stable, even as the kind of work that I'm doing changes.
      * (However, correctly tagging writing work vs reading work, is unimportant compared to correctly capturing the length of a focus span)
  * A **planning / reflecting span** is a sub-type of focus span in which I'm doing metacognition rather than execution: reviewing my goals, journaling, prioritizing, deciding what to work on, or reflecting on how things are going. Typically this happens on my daily page or in a planning document.
    - Brief flips to my daily page during another span are still just note capture, not a planning span. A planning span is when the planning or reflecting is the work.
  * An **other focused work span** is a residual category: a genuine focus span — sustained attention on one project or object — that doesn't fit any of the named sub-types.
    * This category should be used sparingly. If a span keeps landing here, that's a signal it either deserves its own named sub-type or the standard isn't being applied strictly enough — it must not become a loophole for activity that failed to qualify as one of the named kinds (e.g., task churn with no Toggl entry is unfocused, not "other focused work").
    * Because this category should be used sparingly, the model should maintain the hypothesis that what seems to be other focused work is instead unfocused activity of some kind.
  * A **fully-absorbed focus span** is a length of focus span with 0 attention interruption events.
    * Focused task spans and meeting spans do not count towards fully-absorbed focus spans.

### Model evaluations:

For each segment, the model does several different related evaluations:

* Evaluate the content of the *segment*:
  * 1) What kind of span is this: Lable as Creative, Reading, Planning/Reflecting, Task, Correspondence (chats and email), Meeting, Other, Passive Consumption, or transition.
    2) Is the user focused during this segment, or unfocused?
      * Note that infering whether a segment is focused or not may depend on context from the surrounding segments. If the surrounding segments are all about the same attentional thread, it's more likely that this focus segment is focused.
* Evaluate the nature of the *switch* that lead into the segment: Did the user's attention just shift from one thread to another? Judge this at every switch, based on all the surrounding context, whether or not a focus span is running. Label as one of:
  * a **continuation**: the user is still on the same attentional thread even though the window changed.
    * A **transition** is a subtype of a continuation: When the user is navigating through apps or windows to get to the relevant target, or to get back from an interruption, each of the switches that make up that navigation is a transition.
  * a **shift**: the user's attention moved to something new.
    * Note: A shift is not the same thing as an attention-interruption event, as defined above. An attention-interruption event is a shift that happens during a focus span. The model labels every shift; the app works out which shifts are attention-interruption events.
  * a **return**: the user's attention moved back to something it was on recently (within the past 10 minutes).
* If a switch is labeled as a continuation or as a return, evaluate which earlier segment does this segment continue.
  * This is used to alogirthmically consturct threads.
* If a switch is labeled as a _shift_, then a model additionally evaluates is that shift was a **self-distraction event**?
* If a switch is labeled as a shift, additionally evaluate whether it is the next task under the same task-processing intention (see focused task spans above).

For  all of these juduements the model also offers it's epistemic confidence in that judgement, which are used to construct review questions for user oversight.

The model evaluates these lazily with a 15 minute delay. That allows the model to evaluate the segments together in the context of all the nearby segments. The model can also choose to hold off on evaluating some segments for up to an hour, if it guesses that it will have more relevant information after more segments have been logged.

All of these jugdement  can be audited and edited by the user, and those edits should be used to adjust the model so that it learns over time.

### Review questions

* Each review question is about one thing. There are a few separate dimensions a judgment can be uncertain along: which thread a segment belongs to (the same as before, and which, or a separate one), what its content was (an errand, reading, writing, passive consumption...), and whether it was done inside a live discussion. A question resolves exactly one of these, and its options differ only along that dimension. When more than one is in doubt, they are separate questions, which can sit on the same review card. "Was this browsing part of the coding work?" is a different question from "was this deliberate browsing, or passively perusing without intent?"; the second is a content question, and "passive consumption" is its answer for the drift case. (added 2026-11-12)

### The app tracks:

1. The total focus time per day (the sum of the lengths of all the focus spans that day).
2. The length of the longest focus span of the day.
3. The number of attention-interruption events during focus spans.
4. The number of self-distraction events during focus spans.
5. The length of the longest fully absorbed focus span.

### Some rules and heuristics for evaluating which app transitions are attention-interruption events:

It's the app's job to review the data and make intelligent judgements about  whether the focus of the user's attention has remained stable or has shifted.

* Toggl should be taken as a highly-weighted clue about the focus the user's intention. If a toggl entry labled with a task is likely to cover a focus span in which the users attention was focused on that task. But toggl data is not definitive. A toggl entry could be runnning, and the user might have gotten destracted or shifted focus to something else without having chanaged the time-tracking. Or it might be that the user is continually getting briefly distracted by interruptions, and the minimum amount of interupted time has not been reached. It's the app's job to review the data and make intelligent judgements about what is actually a span of time in which the user's attention is focused on a single object / project / task / intention / goal.
* Flipping to my daily page in roam / logseq, is never an attention-interruption event. I often want to take metacognitive notes on my daily page, and when I have a stray thought while focusing on something else, my protocol is to jot it down on my daily page.
* Briefly filling out my presesnce check form is not an inturpution or a self-distraction event.
* When I’m reading, I’ll typically go back and forth between the window with the text and the window with my notes.
* Taking notes is an indactor of focused work in general. If I'm reading, but not taking notes, that's leans stongly in favor of unfocused time.
* Watching youtube is unfocused time, unless I'm watching a specific youtube video as part of answering a specific question or completing a specific task.
* When switching between apps and windows, I may activate unrelated windows or tabs on the way to the ones that I'm looking for. If I'm writing something on desktop 3, and I need information from a webpage that I know is open in a tab in desktop 7, I may step through desktops 4, 5, 6, and 7, and then in desktop 7, rotate through several windows to find the one that I want, and then click through several open tabs in that window until I find the relevant webpage. If I'm looking for some peice of information that I know is somewhere on one of my desktops, I'll flip through even more windows. This should not be considered an attention-inturpution event.
  * However, some judgement is required to distinguish this case from one in which I have in fact gotten briefly (or not so briefly) distracted and my focus shifted to a different task or object.
  * I should never spend more than 10 seconds on a page if I'm just passing through.
* If, while I’m reading, or writing an essay, I flip to Claude to ask a question that’s related to or branching off from what I’m reading or writing about, that’s an extension of the full absorption span. But if I flip to Claude to follow up on a random question that just occurred to me, that’s an attention interruption (but probably not a self-distraction event).
* Sometimes opening my email is a distraction event, but other times it’s an intentional choice.
* My anki decks are generated from and linked to roam/logseq, and so anki reviews regularly involve clicking a link on an anki card to edit the card in roam/logseq.
* Agent-coding should usually count as a task, and be evaluated by the standards of task spans. However, deep, continuous  engagement with an agent, where the user is carefully reading, thinking about, and responding to a coding agent, or writing up up a detailed spec for a peice of software are both creative work.
  * Identifying which is which is a judgement call
    * Agent coding as a task looks like quickly sending off orders to an agent, often only skimming the replies.
    * Agent coding as creative work is slower. I will fully read each response and ask followup questions, and generally try to understand and solve problems.
  * If the user appears to switch from the creative mode of agent-coding to the task mode on the _same project_, by default treat this as a continuation of the focus span. Alternatively, if the user switches from the task mode to the the creative mode on a project, only the intensive thinking part should count as part of the focus span.
* Agent-coding work will often switching between different software projects, in different terminals (usually within the same Agentic coding UI app). Work on each project should be regarded separately, for computing focus spans.
  * 10 minutes spent agent coding, where I am alternating between one of several different projects, spending less than 5 minutes on each one, should _not_ count as a focus span (at best it might be a focused task span), except insofar as that block of coding meets the standard for a focused task span.
  * However, 10 minutes engaging with the _same_ coding project can count as a focus span, if it's evaluated to be the thoughtful creative mode (or if it meets the standards for a focus task span).
* While an agent is running, switching away to something else (email, a feed, another tab) is never a continuation of the coding work. It is an attention interruption.
  * The exception is a focused task block: if I'm intentionally moving on to a different task while the agent runs, that is part of the focused task span, by the usual standard for those (a toggl entry running, and judgement that I'm deliberately executing on tasks).
  * Whether the interruption is also a self-distraction event follows the usual rule: scrolling a feed is or absentmindedly looking at my inbox is, deliberately handling an email probably is not.


### Lessons from reviews
* During a meeting, judge each on-screen activity on its content: if it serves the meeting, count it as meeting time; if not, treat it as a distraction during the meeting rather than meeting work. (added 2026-10-08 from a review of segment 18151)
* EXCEPTION (added 2026-10-31): If I am drafting replies to posts on Bluesky, X or Filtered Twitter, and the posts I'm replying to arrived hours earlier so that I'm composing on my own schedule rather than in a live back-and-forth, that drafting counts as independent creative work and can form a focus span - along with the lookups (Claude, Grok, LessWrong, linked posts) that serve the argument I'm writing. A genuinely live exchange with rapid turns is still correspondence, not focus. This is a flagged exception, not a general loosening: if it starts producing multi-hour "focus" days built entirely out of social media argument, it's too loose. (added 2026-10-31 from a review of segment 2026-10-31)

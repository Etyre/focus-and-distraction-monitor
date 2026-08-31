[These rules should take priority over other documentation in this folder]

### Definitions: 

* A **segment** is a period of time during which a particular window and tab is active on my computer.
* A **switch** is the transition between seguments.
* A **focus span** is a period of 10 minutes or greater in which my attention is directed toward a single project, task, intention, or idea.  
  * That I’m in a focus span does not mean that I’m only using one app, or only staying on one webpage. I will often switch between different apps and pages while still being engaged in the same intention.  
  * A focus span ends when my computer has been idle for ten minutes (in which case, the span is retroactively made to end when the idle period started) or if my attention has been interrupted and I’ve shifted my attention to a different task, project, or object for 5 minutes (in which case, the span is retroactively shortened to the start of the interruption.   
  * Attention interruptions that are less than 5 minutes long do not break the focus span.   
* An **attention-interruption event**  or a **diversion** is an instance when my focus shifts from the object of a focus span to something unrelated.  
* A **self-distraction event** is a sub-case of an attention-interruption event in which I seek out stimulation or distraction without a clear intent.  
  * For instance:  
    * absent-mindedly opening lesswrong and scrolling the front page (though this requires some judgement, since I will also go to lessWrong to look up quotes or to read particular posts.  
    * Opening a new tab to go to a blocked website.  
    * Absentmindedly opening bumble  
    * When I open Amazon without any particular objective and browse .
* A **passive consumption span** is a period when I'm passively consuming or browsing: scrolling LessWrong, watching YouTube, reading comics, reading blogposts without taking notes.
  * This is important to track mainly because passive consumption time should never count toward a focus span.
* Focus span subtypes:
  * A **focused task span** is a special kind of focus span, in which my attention is focused on executing on a series of taks without interuption.  
    * Focused task spans are the *only* exception to the rule that a focus span entails having my attention on only one task / project / object / intention. In a focused task span, I'm switching between different tasks. But I'm overall focussed on efficiently executing on tasks, without interuption or distraction. 
    * Discerning the difference between focused tasks processing or just doing tasks in a more lackadaisical and less focused way (tagged as "unfocused") requires judgement.  
    * If executing on a focused task block, I’ll usually be operating from a todo list, referring back to that list between tasks. Alternatively, I might be going back to my daily page for explicit metacognition about which task to do next.   
    * Computer activity can only count as Focused Task processing if there's a toggl entry running. 
      * It doesn't have to be the *same* continuous toggl entry. It could be that there's one there's one for processing email, followed by another for doing taxes, followed by another for returning amazon packages. Or maybe there's a single toggl entry labled "task processing". But if there's there's not a running toggl entry, it can't be focused task work.
      * Note that the absensece of a toggl entry disquailifes a span from being a *Focused* task span. But individual segments can and should still be tagged in the "task" category, regardless of whether there's a toggl running. Toggl is required as evidence of deliberate focus, not as evidence of task-nature.
  * A **creative work span** is a sub-type of focus span in which I’m creating something: an essay, a video, some software. etc.  
    * Creative work spans and focused task spans are mutually exclusive.  
  * A **reading span** is a sub-type of focus span in which I'm reading  posts, papers, or books, and taking notes.
    * A reading span can lead directly into a creative span, if the notes I'm writing develop into a writing an essay. In this case, the whole block should be counted as one uninterrupeted focus span, even if part of that focus span is reading and the other part is creative. This is an exception to the general rule that task switching implies the end of a focus span, because I'm *not* switching intentions or changing the focus of my attention. Those are stable, even as the kind of work that I'm doing changes.
      * (However, correctly tagging writing work vs reading work, is unimportant compared to correctly capturing the length of a focus span)
  * A **planning / reflecting span** is a sub-type of focus span in which I'm doing metacognition rather than execution: reviewing my goals, journaling, prioritizing, deciding what to work on, or reflecting on how things are going. Typically this happens on my daily page or in a planning document.
    - Brief flips to my daily page during another span are still just note capture, not a planning span. A planning span is when the planning or reflecting is the work.
  * An **other focused work span** is a residual category: a genuine focus span — sustained attention on one project or object — that doesn't fit any of the named sub-types. 
    * This category should be used sparingly. If a span keeps landing here, that's a signal it either deserves its own named sub-type or the standard isn't being applied strictly enough — it must not become a loophole for activity that failed to qualify as one of the named kinds (e.g., task churn with no Toggl entry is unfocused, not "other focused work").
    * Because this category should be used sparingly, the model should maintain the hypothesis that what seems to be other focused work is instead unfocused activity of some kind.
  * A **fully-absorbed focus span** is a length of focus span with 0 attention interruption events.
    * Focused task spans do not count towards fully-absorbed focus spans.

### Model evaluations: 

For each segment, the model to do several different related evaluations:

* Evaluate the content of the *segment*: 
  * 1) What kind of span is this: Lable as Creative, Task, Reading, Planning/Reflecting, Meeting, Other, Passive Consumption, or transition.
    2) Is the user focused during this segment, or unfocused? 
* Evaluate the nature of the *switch*: Does this switch corespondent to a shift in the user's object of attention? Label as either a continuation, an interuption, transition (meaning, navigating through apps or windows to get to the relevant target or return from ineruption. 
* If a switch is labled as an interuption, then additionally evaluate if it's additionally a self-distraction event (it probably is if the type is "passive consumption", but not if it's something else), or a new focus start. 

All of these should be things that can be audited and edited by the user, and those edits should be used to adjust the model so that it learns over time. 

The model can evaluate these lazily with a 15 minute delay, if that improves performance. It doesn't have to evaluate each segment immediately, it can wait until it has more context about what happened afterwards has come in. 

### The app tracks:

1. The total focus time per day (the sum of the lengths of all the focus spans that day).  
2. The length of the longest focus span of the day.  
3. The number of attention-interruption events during focus spans.  
4. The number of self-distraction events during focus spans.  
5. The length of the longest fully absorbed focus span.

### Some rules and heuristics for evaluating which app transitions are attention-interruption events:

* Flipping to my daily page in roam / logseq, is never an attention-interruption event. I often want to take metacognitive notes on my daily page, and when I have a stray thought while focusing on something else, my protocol is to jot it down on my daily page.  
* Briefly filling out my presesnce check form is not an inturpution or a self-distraction event.
* When I’m reading, I’ll typically go back and forth between the window with the text and the window with my notes. 
* Taking notes is an indactor of focused work in general. If I'm reading, but not taking notes, that's leans stongly in favor of unfocused time.
* Watching youtube is unfocused time, unless I'm watching a specific youtube video as part of answering a specific question or completing a specific task.
* When switching between apps and windows, I may activate unrelated windows or tabs on the way to the ones that I'm looking for. If I'm writing something on desktop 3, and I need information from a webpage that I know is open in a tab in desktop 7, I may step through desktops 4, 5, 6, and 7, and then in desktop 7, rotate through several windows to find the one that I want, and then click through several open tabs in that window until I find the relevant webpage. If I'm looking for some peice of information that I know is somewhere on one of my desktops, I'll flip through even more windows. This should not be considered an attention-inturpution event. However, some judgement is required to distinguish this case from one in which I have in fact gotten briefly (or not so briefly) distracted and my focus shifted to a different task or object.
* If, while I’m reading, or writing an essay, I flip to Claude to ask a question that’s related to or branching off from what I’m reading or writing about, that’s an extension of the full absorption span. But if I flip to Claude to follow up on a random question that just occurred to me, that’s an attention interruption (but probably not a self-distraction event).  
* Sometimes opening my email is a distraction event, but other times it’s an intentional choice.
* If my computer is inactive, I can't be doing a task. 
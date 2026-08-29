I want to make a piece of software that will watch the open / active windows on my screen to track and measure my focus and distraction.

I want it to record

* Self-distraction events: When I go in search of stimulation without a particular objective  
  * eg  
    * Flipping to a blocked URL, which doesn’t lead anywhere, because it’s blocked.  
    * When I flip to lesswrong and scroll looking for something interesting.  
    * When I open Amazon without any particular objective and browse  
    * When I flip to bumble for no particular reason.  
  * It should count how many self-distraction events there are, and also measure how long they are.  
* Attention interruption events: When I quickly switch tasks to do something else  
  * I want to count how many of these there are  
* Full absorption focus spans, when my attention is a single task, topic, thread, or project, without attention interruptions.   
  * How long these are.

This will require some intelligence. 

Just because I’m switching between windows does not mean that I’m switching tasks or the focus of my attention. When I’m reading, I’ll typically go back and forth between the window with the text and the window with my notes for instance. 

And it’s not always obvious what’s a distraction event. If, while I’m reading, or writing an essay, I flip to claude to ask a question that’s related to or branching off from what I’m reading or writing about, that’s an extension of the full absorption span. But if I flip to Claude to follow up on a random question that just occurred to me, that’s an attention interruption (but probably not a self-distraction event).

Sometimes opening my email is a distraction event, but other times it’s an intentional choice.

I might go to lessWrong to distract myself, but then get into reading something and giving that my full absorption (it counts as full absorption if I’m taking notes, and my attentional focus is stable).

So the app will not just record the titles and urls of the webpages and app windows that are active, it will have to actually observe them, taking screenshots to collect information necessary to make a judgement.

The app should train a model to predict which active window switches count as attention interruptions and which count as self-distraction events (specifically it should generate probabilities of “continuation of focus”, “attention interruption event” and “self-distraction event”. 

It should flag which cases are uncertain for human review and learn from the human reviews to improve the model. 

If practical, we can use an ensemble of different models, trying different methods, each of which can learn via whatever mechanisms they implement, the results of which are weighted according to which has had the best performance at predicting the human reviews.

All the data should be logged, so that 

* A) I can few full absorption blocks on a timeline, and   
* B) I have quantitative daily measures of   
  * the longest full absorption block in a day  
  * The total amount of full absorption time in a day  
  * The number of attention interruption events in a day  
  * The number of self-distraction events in a day  
  * The total time spent self distracted in a day

I also want to be able to see graphs of model accuracy over time.  

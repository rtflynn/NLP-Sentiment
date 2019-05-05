# Amazon Review Sentiment Analysis

Sentiment analysis has been on the rise - both because of the availability of new analysis techniques in deep learning, and because there is an incomprehensibly large amount of data being generated everywhere.  Every product review, every tweet, every reddit post, etc, contains subjective information which we'd like to be able to process and understand.  

For example, say you're Netflix.  Then you're very interested in what your customers have to say about your service and your TV show/movie selection, and you might look toward mining facebook posts and tweets and IMDB reviews and so on to gauge public opinion.  If you're a politician, then you're interested (hopefully) in what your constituents are thinking, what they want, what set of values they hold dear, etc, and so you might have a team which analyzes public sentiment in these areas.  If you're an entrepreneur, then you're interested in public opinion as it concerns your niche, your product, and your competition, since this allows you to make more informed decisions moving forward.

In fact, it's difficult to think of industries which would not benefit from having a bit of machine-learnt sentiment analysis in place.  This makes sentiment analysis (and more generally, natural language processing, or NLP) a valuable skill to possess.  Plus, at times it's just plain fun!  

In this project we'll see how to train binary classifiers to the task of reading in an Amazon product review, and outputting whether the review was positive or negative.  Once you've worked through this, you should have the necessary background to go into other NLP tutorials and have a good idea what's going on at all times.  Some interesting popular projects are: design a neural network which outputs original text in the style of a book or author of your choice... design a model which tokenizes sentences in a larger text as happy, sad, angry, etc... design a chat bot...

Finally, a quick note on the style of this writeup:  It's not so much a tutorial as it is a finished project with clear references to the topics involved.  For example, we might briefly discuss (say) word tokenizing, in that we'll describe in a sentence or two what it is and we'll provide python code to carry it out, but interested readers should look elsewhere for detailed explanations of any new concepts.  It has to be this way --- otherwise this readme would be chapters long and not-at-all fun to read!  The intention here is to have a completed project all in one place, to serve as a quick and easy go-to reference/starred project for readers to come back to when they need a reminder on how something works. 

## Topics Covered
We'll be covering the following topics in varied amounts of detail:
  - Finding data sets to play with + putting them in an appropriate format for the various ML packages we'll use
  - NLTK (The Natural Language Toolkit for Python)
  - Word tokenizing techniques
  - Preprocessing training data for NLP
  - "Old-fashioned" NLP techniques  (don't be fooled - these are still used today!)
  - Building and training models using Scikit-Learn
  - RNN (specifically, LSTM)
  - Building and training models using Keras
  - Parameter searching using Talos
 
 
## Preliminary:
I'm using the Amazon Reviews data set which can be found at https://www.kaggle.com/bittlingmayer/amazonreviews .  You'll want to click the download button, unzip the downloaded file, and then unzip the files within the newly unzipped file (7-zip if you get stuck).  You'll want test.ft.txt and train.ft.txt in the same folder as LSTM.py.  You'll notice that these data sets are very large!  In fact, the training set has 400,000 training examples, so while we're playing around with different models it suffices to take some subset of train.ft.txt as our full train/test/validation set.


# Getting Started






This data set consists of labelled amazon product reviews --- with two labels '__label__1' corresponding to a 1- or 2- star review, and '__label__2' corresponding to a 4- or 5- star review.  We'll build a few models to solve this classification problem




```python
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import FreqDist
```

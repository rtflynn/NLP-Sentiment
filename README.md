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
I'm using the Amazon Reviews data set which can be found at https://www.kaggle.com/bittlingmayer/amazonreviews .  You'll want to click the download button, unzip the downloaded file, and then unzip the files within the newly unzipped file (7-zip if you get stuck).  You'll want test.ft.txt and train.ft.txt in the same folder as LSTM.py.  You'll notice that these data sets are very large!  In fact, the testingg set has 400,000 training examples, so while we're playing around with different models it suffices to take some subset of test.ft.txt as our full train/test/validation set.  

All the data in train.ft.txt and test.ft.txt files has already been shuffled.

We're using Python 3 throughout  (Version 3.6.5 to be specific).

# Getting Started: Data Formatting
Open a command prompt, navigate to the folder containing train.ft.txt, and type `python`.  We first want to get a feel for the data:

```python
current_file = open("test.ft.txt", "rb")
x = current_file.read()                     
current_file.close()
```

So now `x` holds the entire contents of test.ft.txt.
```python
len(x)
```
`177376193`      That's a lot!  Well, what does this data look like?


```python
print(x[:100])
```
`b'__label__2 Great CD: My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. When I\'m in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life\'s hidden gems. This is a desert isle CD in my book. Why she never made it big is just beyond me. Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"\n__label__2 One of the best game music soundtracks - for a game I didn\'t really play: Despite the fact that I have only played a small portion of the game, the music I heard (plus the connection to Chrono Trigger which was great as well) led me to purchase the soundtrack, and it remains one of my favorite albums. There is an incredible mix of fun, epic, and emotional songs. Those sad and beautiful tracks I especially like, as there\'s not too many of th'`

OK.  Mostly english text... some excape characters like `\'` and `\n`... a mysterious `b` at the beginning of it all, and `__label__1`, `__label__2` at the beginning of each review.  We're going to want to separate this into individual reviews and clean them all up a little, so at some point we're going to want to do something like `x = x.splitlines()` since each review seems to end with a newline.  Feel free to try this - you'll find that each line now has that mysterious `b` at the beginning!

Well, this is because `x` is being read as a so-called byte-literal instead of as a string.  Let's fix this before splitting:

```python
x = x.decode("utf-8")       # now type(x) tells us that x is a string
x = x.splitlines()
```

Good.  Now `x` is a list of strings.  Let's look into it a bit more:

```python
len(x)
```
`400000`
```python
x[0]
```
`'__label__2 Great CD: My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. When I\'m in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life\'s hidden gems. This is a desert isle CD in my book. Why she never made it big is just beyond me. Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"'`

Looking good.  Each element of `x` consists of a label (__label__2), a title(Great CD), and a review (everything after the colon).  We're going to want to separate the label from the review.  I've carried this out in two different ways:

(1) Create three lists:  labels, titles, and reviews.  My thinking here had to do with the fact that the title (Great CD) of the first review is enough to classify the sentiment, and with the fact that the non RNN approaches I used employed something called bag-of-words, which means that the word 'Great' would be lost in the rest of the review... More on this later.

(2) Create two lists:  labels and reviews.  This is the more straightforward option and it's what we'll do here.

Look through a few elements of `x` if necessary.  You'll notice that the label is always separated from the review by a single space.

```python
labels = []
reviews = []
for i in x:
  separated = i.split(" ",1)
  labels.append(separated[0])
  reviews.append(separated[1])
```

Go ahead and check that things look correct:  `labels` and `reviews` are both python lists of length 400000, `labels[i]` always takes the value `__label__1` or `__label__2`, and `reviews[0]` gives us the familiar review we've already read a few times.  Looking through a few reviews, it's easy to tell that `__label__1` corresponds to bad reviews and `__label__2` corresponds to good reviews.  Let's go ahead and incorporate this so we've got one less thing to think about in the future:

```python
for i in range(len(labels)):
  if labels[i] == '__label__1':
    labels[i] = 'bad'
  elif labels[i] == '__label__2':
    labels[i] = 'good'
  else:
    print("uh oh")

for i in range(5):
  print(labels[i])
```

`good good bad good good`

Great.  We've got `labels` and `reviews` separated.  Now let's start talking about what to do with these sets.

# Bag-of-Words Models
A bag-of-words model is one in which each input (in our case, each review) is first reduced to an unordered set of the words which occur within the input.  For example, the reviews "Great product but the customer service was terrible" and "Terrible product but the customer service was great" have exactly the same bag-of-words reductions (the set {'but', 'customer', 'great', 'product', 'service', 'terrible', 'the', 'was'}), even though the sentiments are very different.  Even though this bag-of-words idea is terribly naive, it works relatively well.  Good reviews will have more positive words in them and bad reviews will have more negative words in them.  Moreover there are some negative words which don't often show up in positive reviews at all, and vice-versa.  

Our first sentiment analyzer will be a bag-of-words model.  We'll need to process our feature set in various important ways:
 - We'll need to 'tokenize' each review down to constituent words.
 - We'll probably want to reduce the possible vocabulary down to a few thousand words, both to decrease training time and to decrease overfitting (some words may appear in only one review and thus be unfairly attributed a positive/negative score).  We'll use the most commonly occuring 3000 words for now.
 - We'll want to make sure each word is lowercase so we don't end up counting the same word as several different words.
 - We'll want to 'stem' or 'lemmatize' our words so that words which are essentially the same ('is', 'are', 'were', 'was', 'be') or ('happy', 'happily') are put into a standard form, again to avoid counting essentially the same word several times.  In general, stemming is a more crude procedure which has the advantage of usually saving disk space, whereas lemmatizing is a more intelligent procedure which better preserves the original meaning of the text.  Both, however, lose some amount of information.
 - We may be interested in tagging the part of speech of each word, or in identifying certain commonly-occurring groupings of parts of speech such as an 'adverb-verb' pair or an 'adjective-noun' pair.  We won't do this in this project, but we point it out as a possibility because NLTK has powerful built-in tools for doing all sorts of tagging of this form.
 - Finally, we'll want to loop through our `reviews` set and for each review output a bag-of-words representation of that review.
 
 Let's get to it.
 
 







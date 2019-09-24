# Amazon Review Sentiment Analysis

Sentiment analysis has been on the rise - both because of the availability of new analysis techniques in deep learning, and because there is an incomprehensibly large amount of data being generated everywhere.  Every product review, every tweet, every reddit post, etc, contains subjective information which we'd like to be able to process and understand.  

For example, say you're Netflix.  Then you're very interested in what your customers have to say about your service and your TV show/movie selection, and you might look toward mining facebook posts and tweets and IMDB reviews and so on to gauge public opinion.  If you're a politician, then you're interested (hopefully) in what your constituents are thinking, what they want, what set of values they hold dear, etc, and so you might have a team which analyzes public sentiment in these areas.  If you're an entrepreneur, then you're interested in public opinion as it concerns your niche, your product, and your competition, since this allows you to make more informed decisions moving forward.

In fact, it's difficult to think of industries which would not benefit from having a bit of machine-learnt sentiment analysis in place.  This makes sentiment analysis (and more generally, natural language processing, or NLP) a valuable skill to possess.  Plus, at times it's just plain fun!  

In this project we'll see how to train binary classifiers to the task of reading in an Amazon product review, and outputting whether the review was positive or negative.  Once you've worked through this, you should have the necessary background to go into other NLP tutorials and have a good idea what's going on at all times.  Some interesting popular projects are: design a neural network which outputs original text in the style of a book or author of your choice... design a model which tokenizes sentences in a larger text as happy, sad, angry, etc... design a chat bot...

Finally, a quick note on the style of this writeup:  It's not so much a tutorial as it is a finished project with clear references to the topics involved.  For example, we might briefly discuss (say) word tokenizing, in that we'll describe in a sentence or two what it is and we'll provide python code to carry it out, but interested readers should look elsewhere for detailed explanations of any new concepts.  It has to be this way --- otherwise this readme would be chapters long and not-at-all fun to read!  The intention here is to have a completed project all in one place, to serve as a quick and easy go-to reference/starred project for readers to come back to when they need a reminder on how something works. 

## Topics Covered
We'll be covering the following topics in varied amounts of detail:
  - NLTK (The Natural Language Toolkit for Python)
  - Word tokenizing techniques
  - Preprocessing training data for NLP
  - "Old-fashioned" NLP techniques  (don't be fooled - these are still used today!)
  - Building and training models using Scikit-Learn
  - RNN (specifically, LSTM)
  - Word embeddings
  - Building and training models using Keras
  - Multithreading using fit_generator
  - Hyperparameter searching using Talos
 
 
## Preliminary:
I'm using the Amazon Reviews data set which can be found at https://www.kaggle.com/bittlingmayer/amazonreviews .  You'll want to click the download button, unzip the downloaded file, and then unzip the files within the newly unzipped file (7-zip if you get stuck).  You'll want test.ft.txt and train.ft.txt in the same folder as LSTM.py.  You'll notice that these data sets are very large!  In fact, the testing set has 400,000 training examples, so while we're playing around with different models it suffices to take some subset of test.ft.txt as our full train/test/validation set.  

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
`177376193                `      That's a lot!  Well, what does this data look like?


```python
print(x[:1000])
```
`b'__label__2 Great CD: My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. When I\'m in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life\'s hidden gems. This is a desert isle CD in my book. Why she never made it big is just beyond me. Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"\n__label__2 One of the best game music soundtracks - for a game I didn\'t really play: Despite the fact that I have only played a small portion of the game, the music I heard (plus the connection to Chrono Trigger which was great as well) led me to purchase the soundtrack, and it remains one of my favorite albums. There is an incredible mix of fun, epic, and emotional songs. Those sad and beautiful tracks I especially like, as there\'s not too many of th'`

OK.  Mostly english text... some escape characters like `\'` and `\n`... a mysterious `b` at the beginning of it all, and `__label__1`, `__label__2` at the beginning of each review.  We're going to want to separate this into individual reviews and clean them all up a little, so at some point we're going to want to do something like `x = x.splitlines()` since each review seems to end with a newline.  Feel free to try this - you'll find that each line now has that mysterious `b` at the beginning!

Well, this is because `x` is being read as a so-called byte-literal instead of as a string.  Let's fix this before splitting:

```python
x = x.decode("utf-8")       # now type(x) tells us that x is a string.  
x = x.splitlines()

##### OPTIONAL  but recommended for quick iteration  #####
##### x = x[:10000]                                  #####
#####                                                #####
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
    labels[i] = 'Bad'
  elif labels[i] == '__label__2':
    labels[i] = 'Good'
  else:
    print("uh oh")

for i in range(5):
  print(labels[i])
```

`good good bad good good`

Great.  We've got `labels` and `reviews` separated.  Now let's start talking about what to do with these sets.

# Bag-of-Words Models
A bag-of-words model is one in which each input (in our case, each review) is first reduced to an unordered set of the words which occur within the input.  For example, the reviews "Great product but the customer service was terrible" and "Terrible product but the customer service was great" have exactly the same bag-of-words reductions ( the set {'but', 'customer', 'great', 'product', 'service', 'terrible', 'the', 'was'} ), even though the sentiments are very different.  Even though this bag-of-words idea is terribly naive, it works relatively well.  Good reviews will have more positive words in them and bad reviews will have more negative words in them.  Moreover there are some negative words which don't often show up in positive reviews at all, and vice-versa.  

By the way, this is the reason that above, when we were splitting our data into labels and reviews, I mentioned the idea of also separating out the titles.  A bag-of-words reduction of a review title is much less likely to lose much information, and important keywords like 'great' and 'terrible' are less likely to be lost in the noise of a giant bag of words.  We will revisit this idea later, and in particular we'll create a model out of two smaller models - one trained on titles and one trained on review texts.  This model will work by first trying to classify the review based on title alone, and if it returns a label with very high confidence, we use that answer; otherwise we look at the review text.  This separation of one feature into two is a form of 'feature engineering' and is a common practice in real-world classification and regression tasks.

Our first sentiment analyzer will be a bag-of-words model.  We'll need to process our feature set in various important ways:
 - We'll need to 'tokenize' each review down to constituent words.
 - We'll probably want to reduce the possible vocabulary down to a few thousand words, both to decrease training time and to decrease overfitting (some words may appear in only one review and thus be unfairly attributed a positive/negative score).  We'll use the most commonly occuring 3000 words for now.
 - We'll want to make sure each word is lowercase so we don't end up counting the same word as several different words.
 - We'll want to 'stem' or 'lemmatize' our words so that words which are essentially the same ('is', 'are', 'were', 'was', 'be') or ('happy', 'happily') are put into a standard form, again to avoid counting essentially the same word several times.  In general, stemming is a more crude procedure which has the advantage of usually saving disk space, whereas lemmatizing is a more intelligent procedure which better preserves the original meaning of the text.  Both, however, lose some amount of information.
 - We may be interested in tagging the part of speech of each word, or in identifying certain commonly-occurring groupings of parts of speech such as an 'adverb-verb' pair or an 'adjective-noun' pair.  We won't do this in this project, but we point it out as a possibility because NLTK has powerful built-in tools for doing all sorts of tagging of this form.
 - Finally, we'll want to loop through our `reviews` set and for each review output a bag-of-words representation of that review.
 
Let's get to it.

## Preparing the features
 
 ```python
 import nltk
 from nltk.tokenize import word_tokenize, RegexpTokenizer
 
 print(word_tokenize(reviews[0]))
 reTokenizer = RegexpTokenizer(r'\w+')
 print(reTokenizer.tokenize(reviews[0]))  
 ```

If you run the above lines, you'll see that word_tokenize separates the review into a list of words, but considers punctuation as words.  This might be desirable for some sorts of models (for example, an attention model may learn to look at what comes before an exclamation mark or a set of ellipses), but for a bag-of-words model it seems like punctuation will simply add noise.  

A RegexpTokenizer can be built to tokenize according to any regular expression you hand it.  The regular expression `r'\w+'` matches any pattern consisting of one or more consecutive letters, so works fine for our purposes.  Note that this matcher will miss certain things like Ph.D. (i.e. it'll tokenize this as two words) and hyphenated words and contractions, but it should still work fine for our task.  We'll use the RegexpTokenizer going forward.

Now let's work on getting everything stemmed and in lowercase.

```python
 from nltk.stem import PorterStemmer
 
 ps = PorterStemmer()
 temp = reTokenizer.tokenize(reviews[0])
 for i in range(10):
  print(ps.stem( temp[i].lower() ))
```

OK... lots of information has been lost by stemming.  Let's try lemmatizing instead:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
temp = reTokenizer.tokenize(reviews[0])
for i in range(10):
  print(lemmatizer.lemmatize( temp[i].lower() ))

```

Better.  Let's use this.  Finally, we need to collect all the words from all the reviews into one list and keep the 3000 most common words as our bag-of-words vocabulary.

```python
all_words = []
for i in range(len(reviews)):
  tokens = reTokenizer.tokenize(reviews[i])
  reviews[i] = []
  for word in tokens:
    word = word.lower
    word = lemmatizer.lemmatize(word)
    all_words.append(word)
    reviews[i].append(word)
```

All our words are in the list `all_words`.  NLTK conveniently provides functionality for extracting the most common words from a list.
    
```python    
from nltk import FreqDist

all_words = FreqDist(all_words)
most_common_words = all_words.most_common(3000)
word_features = []
for w in most_common_words:
  word_features.append(w[0])
```

`most_common_words` is a python list of tuples like `('good', 32655)` if the word 'good' happens to have appeared 32655 times in our data.  This list consists of the 3000 most common words and they're sorted in order of most- to least- frequent.  We collect all the words into our `word_features` list.  We can now iterate through each review in `reviews` and create a vector of 1's and 0's for a given review depending on which words from our chosen 3000 show up in that review.  However we should think ahead a little --- which ML algorithm will we use, and what format does it prefer its data in?

NLTK includes some classifiers off the shelf, so let's keep things simple and use one of those.  We'll use the `nltk.NaiveBayesClassifier()` for now.  Naive Bayes is, as the name suggests, quite a naive method.  It simply correlates individual words with probability distributions for labels, so that (for instance) the word 'good' might correlate to a probability distribution like  90% 'Good', 10% 'Bad' .  It does this in a manner which treats all word occurences as independent from one another.  A consequence is the fact that the word 'not' has an associated label distribution which only has to do with how many times the word 'not' shows up in positive vs. negative reviews.  A naive Bayes classifier will never, ever be able to understand the phrase 'not good' to mean 'bad'.  But if you think about it, this is the perfect classifier to use with a bag-of-words model, since we've already thrown away all the interconnections between the words anyway!  

It's good that we've decided on which classifier to use, because now we know what format it wants its training data in.  Looking at the documentation real quick (or looking directly at the code for nltk.NaiveBayesClassifier()), we see that it wants its training data to be packaged as a python list of ordered pairs (feature_dict, label), where `feature_dict` is a dictionary with (key, value) pairs of the form ('some_word', 0 or 1).

```python
def make_feature_dict(word_list):
  feature_dict = {}
  for w in word_features:
    if w in word_list:
      feature_dict[w] = 1
    else:
      feature_dict[w] = 0
    return feature_dict


nltk_data_set = []
for i in range(len(labels)):
  nltk_data_set.append( (make_feature_dict(reviews[i]), labels[i]) )

train_proportion = 0.9
train_set_size = int(train_proportion * len(labels))

training_set = nltk_data_set[:train_set_size]
testing_set = nltk_data_set[train_set_size:]
```

## Training and Testing the Model
Believe it or not, we're done with the hard part.  All that's left is to instantiate a classifier, train it, and see how it does on the test set after training.  This will take a while if the optional line of code  `x = x[:10000]` was ommitted at the beginning of the project.


```python
print("Beginning classifier training...")
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Finished classifier training... its accuracy on the testing set is: ")
print(nltk.classify.accuracy(classifier, testing_set))

print("The most informative features were: ")
classifier.show_most_informative_features(30)
```

And there you have it.  

Results will vary based on vocabulary size and number of test/train examples.  On my machine the first run gave accuracy 0.811 and most informative feature 'refund' with a 44.6 to 1 proportion of 'Bad' to 'Good'.  The next most informative features were 'publisher', 'waste', 'worst', 'zero', 'pathetic', and 'elevator', 'awful', and 'defective', all of which point toward 'Bad'.  Many of these seem correct, but publisher and elevator don't make much sense.  This is simply what happens with naive Bayes on small data sets.  The first informative feature which points towards 'Good' is 'refreshing'.  

In fact, after training many models on this problem, I've noticed that the set of informative features which point toward 'Bad' vastly outnumbers the set of features which point toward 'Good'.  This introduces its own problems to the classifier, but that's an issue for another day.

We can check this setup's average performance with the following:

```python
import random

train_proportion = 0.9
train_set_size = int(train_proportion * len(labels))


accuracies = 0
for i in range(10):
    random.shuffle(nltk_data_set)

    training_set = nltk_data_set[:train_set_size]
    testing_set = nltk_data_set[train_set_size:]

    print("Beginning classifier training...")
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Finished classifier training... its accuracy on the test set is: ")
    print(nltk.classify.accuracy(classifier, testing_set))
    accuracies += nltk.classify.accuracy(classifier, testing_set)

print("Average accuracy across 10 models: ")
print(accuracies/10)
```

With total train/test size 10,000, we got an average accuracy of .824.  This is pretty good!  But let's not get too excited - the data sets we used were small and may have had too many similar reviews by chance.  To be sure, we repeated the experiment with 160,000 test/train examples and got an average accuracy of .818.


I'd originally planned on discussing how to get nltk and scikit-learn working together --- and may do so on some future update to this project.  For now, however, let's move on to LSTM models.

# Sequential Models
The very naive bag-of-words model worked surprisingly well, achieving over 80% accuracy on the sentiment classification task we gave it.  Let's turn our attention to neural networks and see how they fare.

We'll be using a special kind of neural network architecture known as LSTM.  LSTM stands for 'Long Short-Term Memory', and is an example of a recurrent neural network (RNN), i.e. a network which analyses data over multiple time steps.  There are many amazing LSTM tutorials out there, and this isn't one of them, so let's content ourselves with a review of the very basics:

The beginning of the process is the same as before:  We tokenize our reviews into individual words, count the number of occurences of each word, and make a vocabulary out of the most common 3000 (say) words.  We build our training and test data by going through a review and keeping all words which belong to our vocabulary, but importantly this time we keep them in order (!).  

We start with the following neural network:  First, there's an 'embedding' layer which does something special we'll discuss in a bit.  Next, we feed our words into an LSTM cell.  We actually feed information to an LSTM cell one letter at a time, or one word at a time, or one sentence at a time... one X at a time, where X is decided on before we create the network (the choice of X determines what the 'embedding' layer alluded to above looks like).  

LSTM cells do remarkable things.  Essentially, an LSTM cell is able to remember previous data, and it's able to forget previous data (within a given training or test example).  More than this, an LSTM cell is able to decide which previous data is important to remember and which previous data is OK to forget, and it's able to learn to make this decision better and better (i.e. it makes these decisions based on internal parameters and is able to tweak these parameters to decrease a loss function).  Going back to an earlier example, an LSTM would have absolutely no trouble distinguishing the statements "Great product but the customer service was terrible" and "Terrible product but the customer service was great" from one another.

A nice property of LSTM which we won't use here is that LSTM cells actually learn sequence-to-sequence mappings, meaning they're well-suited to tasks like: given a paragraph of text from a story book, list all characters mentioned in that paragraph and what they did; given a sentence with the last word omitted, predict what the last word is; translate a passage from English to French.

Back to this magical 'embedding' from before:  The idea is similar to lemmatizing.  When we lemmatize the words 'great', 'greater', 'greatest', they all map to the same word, 'great'.  However, the words 'good' and 'best' are not recognized as being close in meaning to 'great' via this procedure.  An embedding layer works as follows:  We take our vocabulary (which has size 3000), and one-hot encode our individual words into it.  Now our words are sitting incredibly sparsely and quite wastefully in a 3000-dimensional space.  We map this space down to (say) a 50-dimensional space via some linear map, and we learn the parameters of this map over time via gradient descent.

Note:  This isn't how Keras does it --- Keras skips the one-hot embedding and instead uses a more-efficient dictionary lookup.

What makes this work is the fact that in 50-dimensional space, there's not enough room for each word to get its own axis.  Therefore a bunch of words need to point in similar directions, and we get more accurate models when similar words get similar direction vectors associated to them.  The directions are initialized randomly and modified over time through backpropogation, so that eventually words like 'good', 'great', and 'best' are all pointing in similar directions to one another.  

This affords some other very, very nice properties --- for example, a good embedding might have "car" as an almost-linear-combination of "fast" and "vehicle", because "car" is more similar to "fast" and "vehicle" than it is to, say, "transparent" or "hamper".  This requires a little thought, but one can imagine that in a high-dimensional space, being close to two vectors means being close to the plane spanned by them.

Similarly, one might take the word embedding for 'king', subtract the word embedding for 'man', add the word embedding for 'woman', and end up awfully close to the word embedding for 'queen'.  This whole topic is actually really cool and very worthy of a few afternoons on Wikipedia and YouTube.

In any case, our model will look like:  Embedding Layer -->  LSTM Cell --> Single Unit (sigmoid) .

## Preparing the Features

We'll need to prepare our features in a different format for Keras.  For one thing, Keras models need numpy arrays to be passed to them.  For another, since our embedding layer will take care of word similarity, we may want to revisit whether we lemmatize or not.  It may well be the case that the particular form of a word changes the meaning of the sentence it's in enough to matter, so we should experiment both with and without lemmatizing our input.

Another thing to think about:  With the bag-of-words model, all our test/train examples automatically had the same length (3000) because each example was just a binary-valued dictionary whose key set was the set of most common words.  But here, our reviews have different lengths and so our examples potentially have different lengths.  This isn't necessarily problematic but things will be easier if we prepare all inputs to be the same length (say 500).  To do this we'll cut longer inputs down to length 500, and we'll pad shorter reviews at the beginning with 0's.  We'll do this with a preprocessing module from keras called 'sequence'.  

Let's start from scratch.  This time we'll put all our imports up front.

```python
from keras.layers import Embedding, Dense, LSTM  # CuDNNLSTM instead of LSTM if you've got tensorflow-gpu
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

vocab_size = 3000
max_length = 500

current_file = open("test.ft.txt", "rb")
x = current_file.load()
current_file.close()

x = x.decode("utf-8")
x = x.splitlines()
x = x[:10000]     ### For quick iteration.  Recall this dataset has 400,000 examples.

labels = []
reviews = []

for i in x:
  separated = i.split(" ",1)
  labels.append(separated[0])
  reviews.append(separated[1])

for i in range(len(labels)):
  labels[i] = int(labels[i] == '__label__2')

reTokenizer = RegexpTokenizer(r'\w+')
all_words = []

for i in range(len(reviews)):
  tokens = reTokenizer.tokenize(reviews[i])
  reviews[i] = []
  for word in tokens:
    word = word.lower()
    all_words.append(word)
    reviews[i].append(word)
    
all_words = FreqDist(all_words)
all_words = all_words.most_common(vocab_size)
for i in range(len(all_words)):
  all_words[i] = all_words[i][0]
```

Everything up to this point should make sense.  Thinking ahead a bit, a Keras embedding layer will take as input a numpy array of positive integers - for example \[52, 11, 641, 330, 1066, 12, 1, ..., 18\] - where the integers are essentially tokens of words.  So, we'll want a dictionary which takes distinct words to distint positive integers.  It might be helpful to have the reverse dictionary handy, both for debugging and for analysis.  

```python
word2int = {all_words[i] : i+1  for i in range(vocab_size)}   # i+1 because we want to start at 1
int2word = {x : y  for  y, x in word2int.items()}
dict_as_list = list(word2int)

def review2intlist(review):
  int_list = []
  for i in review:
    if i in word2int.keys():
      int_list.append(word2int(i))
  return int_list

lstm_input = []
for rev in reviews:
  lstm_input.append(np.asarray(review2intlist(rev), dtype=int))
lstm_input = sequence.pad_sequences(lstm_input, maxlen=max_length)

train_proportion = 0.9
train_size = int(train_proportion * len(labels))

x_train, y_train = lstm_input[:train_size], labels[:train_size]
x_test, y_test = lstm_input[train_size:], labels[train_size:]
```

## Building the Model

Building a Keras model is a pretty straightforward process.  We'll use the Sequential API because it's simpler and we don't need the flexibility of the functional API.  Most of the following code should be easy enough to parse for meaning, even for people who aren't necessarily used to Keras.  There is one subtle point, though: the Embedding layer has `input_dim=vocab_size+1` because we have 3001 input tokens: our 3000 words, and our 'padding symbol' 0.

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size+1, output_dim=64, input_length=max_length))
model.add(LSTM(100))
Model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=3, verbose=2)

loss, accuracy = model.evaluate(x_test, y_test)     # Better: split the data into train/test/validate sets
print(accuracy)
```

This model achieves an accuracy of .848.  This is already better than a bag-of-words model trained on 160,000 data points, and we've only used 10,000 here.  One of the most well-known principles regarding neural networks is that they seem to improve, across the board, with more data.  I've gone ahead and trained models on larger data sets and found the following accuracies (in the following, the number given is the total size of the train/test/validation sets combined, and we've switched out "test.ft.txt" for "train.ft.txt" because the former has 'only' 400,000 examples):

400,000 examples:  accuracy .92525

800,000 examples:  accuracy .9353

1,600,000 examples:  accuracy .9444

2,500,000 examples:  accuracy .9469

After this point the computation takes too much memory to fit in RAM.  I decided to train models with more LSTM hidden units and 2,500,000 examples to the following results:

200 LSTM units:  accuracy .9498

300 LSTM units:  accuracy .9502

400 LSTM units:  accuracy .9493

It looks like there is some upside to increasing the number of units, but we should always be careful not to use too many as we want to avoid overfitting.  There's also the practical issue that if we want to someday deploy this model, fewer units is better since it takes less space to store the neural network and fewer operations to evaluate.  Just to check, I trained a model with 50 hidden LSTM units and got an accuracy of .9454.  So there's definitely a case to be made for using smaller networks.


## Improving the Model

The previous computations would take a pretty long time on a CPU.  I ran all this on a GPU but at any given time the computation only used about 4% of GPU power.  We've also hit the wall on the number of training examples we can store in memory --- train.ft.txt contains 3,600,000 examples and test.ft.txt contains 400,000, so if we use these as our train/test set, we should be able to improve our model a bit more.  Thankfully there's a way to hit two birds with one stone here, and that's what we'll discuss next.

But first, this whole thing is getting rather large.  It will be useful to refactor this code into several .py files to keep it modular and manageable.  I'll do this without listing all the code here - rather, you'll be able to find the refactored version in the folder 'Improved LSTM'.  Once again for anyone who's cloning this, you'll want to add test.ft.txt and train.ft.txt to this folder, or change the paths in the .py files to wherever your test and train data are sitting.

Here's a brief list of the changes:  
 - Move the data preparation to another file
 - Add a file for the building of 'generators'
 - Add a file for a 'shutup' module whose purpose is to squelch all the tensorflow announcements which appear in the terminal
 - Added a separate pair of files to take care of Talos hyperparameter search
 - Kept the LSTM in the main 'LSTM.py' file, and altered how it runs to take advantage of 'generators'

## model.fit_generator()

Keras provides an alternative method to `model.fit(args)` which goes by `model.fit_generator(args)`.  The main difference is that with `model.fit(x_train, y_train, ...)`, we have to provide the training set all at once, and the training set must fit in memory.  fit_generator, on the other hand, takes in a so-called data generator, which is an object of class keras.utils.Sequence that implements a `__getitem__(index)` method whose job it is to return the index'th training batch in a training set.  With this done, we can train our model via `model.fit_generator(generator=myGenerator, workers=4, use_multiprocessing=True, ...)`.  

The generator feeds in the training data as it's needed; in particular we don't need to hold the entire training set in memory at once.  An added bonus comes with the second and third arguments of `fit_generator`:  the `use_multiprocessing` flag allows us to spawn several generators on distinct threads, and `workers` dictates how many threads to use.  This allows us to push more data to our GPU since we can have all our CPU cores doing this instead of just one.

For my implementation of a generator, I decided to simply pre-generate all the training/testing data as we've done before and to save it to disk in numpy files, each containing batch_size training examples.  This is expensive on disk space but allows for very quick iteration through different models, especially because when using a large portion of the data set, data preparation takes a nontrivial amount of time.  

Oddly enough, when using `workers > 1`, each thread tries to generate all the data from scratch, and it does so every epoch!  So if I try to train a model with 8 workers and 10 epochs, it will try to prepare our data 80 times!  This runs significantly slower than using plain-old `model.fit()` AND uses all system memory by the time the second or third thread has started data preparation.

I got around this by having my data preparation method first check a boolean to see whether data prep has been done already, and to `pass` if it has already been completed.  Before even defining the model I prepare the data so that in the first epoch we don't run out of memory.  After defining the model, Keras still tries to create the training set 80 times, but each time stops as soon as it hits the boolean flag.  

A quick warning to anyone cloning and running this repo:  Each distinct choice of `(batch_size, vocab_size, max_length)` causes the data preparation module to create a new directory on disk and populate it with the relevant data.  This can get out of hand quickly so be careful.

## Talos and Hyperparameter Search

The previous improvements allowed for much quicker iteration through different model sizes and architectures, but each training run had to be set up 'by hand' - i.e. we'd have to go in and change `vocab_size` or `num_units` manually and train the model again to see whether it does better or worse than before.  Luckily there are some nice python modules out there which automate this task to an extent.  I decided to go with 'Talos'.  Installation is simple:  `pip install talos` or `conda install talos`.

Since this isn't a tutorial, I won't go through how to use Talos, except to note that it's very easy to pick up.  We only have to do two things:  

(1) Create a python dictionary which contains the various hyperparameter options we want to try out:
```python
params = {'vocab_size': [3000, 6000, 9000], 'max_length': [300, 500, 800], 'num_data_points': [100000],
                  'embedding_size': [64, 128, 256], 'batch_size': [128, 256, 512], 'optimizer': [Adam, Nadam],
                  'loss': [binary_crossentropy, categorical_hinge, mean_squared_error], 'num_units': [50, 100, 200], 
                  'multiple_LSTM_layers': [False, True], 'lr': [0.001, .01, .1, 1, 10]}
```

(2) Slightly alter the code which builds our model.  As an example, we change 

`model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_size, input_length=max_length))` 

to 

`model.add(Embedding(input_dim=params['vocab_size'] + 1, output_dim=params['embedding_size'], input_length=params['max_length']))`.

Finally, call the `Talos.Scan()` method on the newly-written model.  It will scan through all possibilities (or a user-provided proportion of them, chosen randomly) and output training results to a .csv file.  

One downside is that (it seems) Talos doesn't play nice with `fit_generator`, so I had to use `model.fit()` and pass in the entire training set at once.  I ran this overnight on a training set of size 100,000 and found the following about our models:

 - larger values of max_length and vocab_size improve model performance (on the validation set, here and in the following)
 - large embedding sizes actually decrease performance.  This makes some sense: we want the embedding space to be small enough to force similar words to stick together.
 - large batch sizes also decrease performance.  Granted, we only tested batch_sizes of 128, 256, and 512.  If we tested smaller batch sizes as well, we'd likely find a reasonable, medium-sized batch size works best.
 - More LSTM units and additional LSTM layers help performance.  
 - It doesn't seem to make a difference whether we use Adam or Nadam.
 - TO DO:  rerun to see which works best between loss functions.  Accidentally only ran it with binary_crossentropy the first time!  Although mean_squared_error is almost guaranteed to have abysmal performance, categorical_hinge might work nicely.
 

This Talos search was not exhaustive --- it only went through one one-hundredth of all possible combinations of the hyperparameters we fed it, and this took several hours.  But it was still worthwhile.  If we wanted, we could make changes to our `params` dictionary to reflect what we discovered and try to zero-in a bit more closely to an optimal set of hyperparameters.  At this level we can also change the proportion of combinations to try from .01 to .1.  Care must be taken, however, not to 'overfit to the validation set'.  That is, some models will by chance do better on the validation set, even though they're not actually better at the classification task.  For this reason it's good to have a separate test set hidden away somewhere, and it's also not a bad idea to stop the hyperparameter search after one or two iterations of this entire process.
 
# Pulling Everything Together

We've gotten a good set of hyperparameters for our Amazon Review Sentiment Classifier, but this was only trained on 100,000 data points.  It's time to see how it does when trained on the entire training set (using fit_generator and multiple cores + GPU).  

The final architecture we're going with (which we found with help from Talos) is:

```python
vocab_size=9000
embedding_size=100
max_length=500

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_size, input_length=max_length))
model.add(CuDNNLSTM(200, return_sequences=True))
model.add(CuDNNLSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

After training for 3 epochs, the test loss/accuracy are .1170, .9580.

We could probably eke out another half to full percentage point of accuracy by using a pretrained word embedding, using dropout layers, using an attention mechanism, and iterating a bit more on our hyperparameter search.


Questions, comments, suggestions and corrections are very welcome.  Just raise an issue.







import keras
from keras.datasets import imdb
from keras.layers import Embedding, Dense, LSTM
from keras import Sequential
from keras.preprocessing import sequence
import numpy as np
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import FreqDist


vocab_size = 30000                 # Number of words to Embed - i.e. this determines the size of the one-hot layer
max_length = 400                  # Max length of any review (in words).  We'll pad/cut all reviews to this length.
train_set_proportion = 0.95       # Train/Test set proportion
num_data_points = 20000           # How many train/test examples to use (the data set is massive)
embedding_size = 128              # How many dimensions to embed our words in

train_size = int(num_data_points * train_set_proportion)
batch_size = 64
num_epochs = 10



current_file = open("test.ft.txt", "rb")
x = current_file.read()
current_file.close()

x = x.decode("utf-8")
x = x.splitlines()
x = x[:num_data_points]
labels = []
titles = []
texts = []
full_reviews = []

tokenizer = RegexpTokenizer(r'\w+')
# In the following:  Either use tokenizer.tokenize(i), or use sent_tokenize(i).
# The difference is that the RegexpTokenizer throws away punctuation, and might lose some words like "Mr.".
for i in x:
    separated = i.split(" ", 1)
    labels.append(separated[0])
    full_reviews.append(separated[1])

for i in range(len(labels)):
    labels[i] = int(labels[i] == '__label__1')
#for i in full_reviews:
#    separated = i.split(":", 1)
#    titles.append(separated[0])
#    texts.append(separated[1])

all_words = []
for i in full_reviews:
    for w in tokenizer.tokenize(i):
        all_words.append(w)
#for i in titles:
#    for w in tokenizer.tokenize(i):
#        all_words.append(w)
#for i in texts:
#    for w in tokenizer.tokenize(i):
#        all_words.append(w)

all_words = FreqDist(all_words)
all_words = all_words.most_common(vocab_size)

word2int = {all_words[i][0]: i+1 for i in range(vocab_size)}      # i+1 to start at 1, because we're padding with 0's.
int2word = {x: y for y, x in word2int.items()}
dict_as_list = list(word2int)


def review2intlist(rev_text):
    int_list = []
    tokenized = tokenizer.tokenize(rev_text)
    for i in tokenized:
        if i in word2int.keys():
            int_list.append(word2int[i])
    return int_list


X = []
for i in full_reviews:
    X.append(np.asarray(review2intlist(i), dtype=int))
X = sequence.pad_sequences(X, maxlen=max_length)

x_train, y_train = X[:train_size], labels[:train_size]
x_test, y_test = X[train_size:], labels[train_size:]

x_valid = x_train[:10*batch_size]
y_valid = y_train[:10*batch_size]
x_train = x_train[10*batch_size:]
y_train = y_train[10*batch_size:]

print(X[0])
print(x_train[0])
print(y_train[0])

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=max_length))
# The last word has index 30000 which is not in [0:30000], thus the +1 above
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs, verbose=2)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)


#.835
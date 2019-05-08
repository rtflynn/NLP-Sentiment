from keras.layers import Embedding, Dense, LSTM, CuDNNLSTM
from keras import Sequential
from keras.preprocessing import sequence
import numpy as np
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import random

vocab_size = 6000                 # Number of words to Embed - i.e. this determines the size of the one-hot layer
max_length = 500                  # Max length of any review (in words).  We'll pad/cut all reviews to this length.
train_set_proportion = 0.9        # Train/Test set proportion
num_data_points = 10000           # How many train/test examples to use (the data set is massive)
embedding_size = 128              # How many dimensions to embed our words in

train_size = int(num_data_points * train_set_proportion)
batch_size = 1024
num_epochs = 10

current_file = open("train.ft.txt", "rb")
x = current_file.read()
current_file.close()

x = x.decode("utf-8")
x = x.splitlines()
random.shuffle(x)
x = x[:num_data_points]
labels = []
reviews = []

reTokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
for i in x:
    separated = i.split(" ", 1)
    labels.append(separated[0])
    reviews.append(separated[1])

for i in range(len(labels)):
    labels[i] = int(labels[i] == '__label__1')


all_words = []
for i in range(len(reviews)):
    tokens = reTokenizer.tokenize(reviews[i])
    reviews[i] = []
    for word in tokens:
        word = word.lower()
        #word = lemmatizer.lemmatize(word)
        all_words.append(word)
        reviews[i].append(word)


all_words = FreqDist(all_words)
all_words = all_words.most_common(vocab_size)

word2int = {all_words[i][0]: i+1 for i in range(vocab_size)}
int2word = {x: y for y, x in word2int.items()}
dict_as_list = list(word2int)


def review2intlist(rev_text):
    int_list = []
    for word in rev_text:
        if word in word2int.keys():
            int_list.append(word2int[word])
    return int_list


X = []
for i in reviews:
    X.append(np.asarray(review2intlist(i), dtype=int))
X = sequence.pad_sequences(X, maxlen=max_length)

LSTM_inputs = np.zeros(shape=(max_length, num_data_points), dtype=np.float32)
for i in range(len(X)):
    LSTM_inputs[:, i] = X[i]
LSTM_inputs = LSTM_inputs.T

LSTM_outputs = np.zeros(shape=num_data_points)
for i in range(len(labels)):
    LSTM_outputs[i] = labels[i]

x_train, y_train = LSTM_inputs[:train_size], LSTM_outputs[:train_size]
x_test, y_test = LSTM_inputs[train_size:], LSTM_outputs[train_size:]

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=max_length))
# The last word has index 30000 which is not in [0:30000], thus the +1 above
model.add(CuDNNLSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=num_epochs, verbose=2)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(accuracy)


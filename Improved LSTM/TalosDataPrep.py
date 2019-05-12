import random
from nltk.tokenize import RegexpTokenizer
import os
import pickle
from keras.preprocessing import sequence
import numpy as np

dirname, _ = os.path.split(os.path.abspath(__file__))
vocab_directory = os.path.join(dirname, str("top_vocab_words"))


def prepare_data(filepath, num_data_points=40000, vocab_size=4000, max_length=500):
    train_set_proportion = 0.9
    train_size = int(num_data_points * train_set_proportion)

    print("Preparing Data...")
    current_file = open(filepath, "rb")
    x = current_file.read()
    current_file.close()

    x = x.decode("utf-8")
    x = x.splitlines()
    random.shuffle(x)
    x = x[:num_data_points]
    labels = []
    reviews = []

    reTokenizer = RegexpTokenizer(r'\w+')

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
            all_words.append(word)
            reviews[i].append(word)

    vocab_pickle_location = os.path.join(vocab_directory, "all_words.pkl")

    if not os.path.isdir(vocab_directory):
        print("Error: vocab_directory doesn't exist!")
    else:
        all_words = pickle.load(open(vocab_pickle_location, 'rb'))
        all_words = all_words[:vocab_size]

    word2int = {all_words[i][0]: i + 1 for i in range(vocab_size)}

    # int2word = {x: y for y, x in word2int.items()}
    # dict_as_list = list(word2int)

    def review2intlist(rev_text):
        int_list = []
        for word in rev_text:
            if word in word2int.keys():
                int_list.append(word2int[word])
        return int_list

    X = []
    for i in range(len(reviews)):
        X.append(review2intlist(reviews[i]))
    X = sequence.pad_sequences(X, maxlen=max_length)

    LSTM_inputs = np.zeros(shape=(max_length, num_data_points), dtype=np.float32)
    for i in range(num_data_points):
        LSTM_inputs[:, i] = X[i]
    LSTM_inputs = LSTM_inputs.T

    LSTM_outputs = np.zeros(shape=num_data_points)
    for i in range(num_data_points):
        LSTM_outputs[i] = labels[i]

    x_train, y_train = LSTM_inputs[:train_size], LSTM_outputs[:train_size]
    x_test, y_test = LSTM_inputs[train_size:], LSTM_outputs[train_size:]

    half_test_size = int(len(y_test)/2)
    x_valid = x_test[:half_test_size]
    y_valid = y_test[:half_test_size]
    x_test = x_test[half_test_size:]
    y_test = y_test[half_test_size:]

    print("Finished preparing data...")
    return x_train, y_train, x_test, y_test, x_valid, y_valid
import random
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from keras.preprocessing import sequence
import os
import pickle
from saveloadpath import get_save_load_path

dirname, _ = os.path.split(os.path.abspath(__file__))

class DataPrepare:
    def __init__(self, vocab_size=3000, max_length=400, batch_size=1024, num_data_points=100000, name="name"):

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_data_points = num_data_points
        self.vocab_size = vocab_size
        self.num_batches = int(np.floor(self.num_data_points/float(self.batch_size)))
        self.name = name
        self.save_directory = get_save_load_path(name=self.name, batch_size=self.batch_size,
                                                 vocab_size=self.vocab_size, max_length=self.max_length)
        self.vocab_directory = os.path.join(dirname, str("top_vocab_words"))

    def prepare_data(self, filepath):
        if not os.path.isdir(self.save_directory):
            print("Preparing Data...")
            current_file = open(filepath, "rb")
            x = current_file.read()
            current_file.close()

            x = x.decode("utf-8")
            x = x.splitlines()
            random.shuffle(x)
            x = x[:self.num_data_points]
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

            vocab_pickle_location = os.path.join(self.vocab_directory, "all_words.pkl")

            if not os.path.isdir(self.vocab_directory):
                all_words = FreqDist(all_words)
                all_words = all_words.most_common(100000)
                os.mkdir(self.vocab_directory)
                open(vocab_pickle_location, 'x')
                pickle.dump(all_words, open(vocab_pickle_location, 'wb'))
                all_words = all_words[:self.vocab_size]
            else:
                all_words = pickle.load(open(vocab_pickle_location, 'rb'))
                all_words = all_words[:self.vocab_size]



            word2int = {all_words[i][0]: i + 1 for i in range(self.vocab_size)}
            #int2word = {x: y for y, x in word2int.items()}
            #dict_as_list = list(word2int)

            def review2intlist(rev_text):
                int_list = []
                for word in rev_text:
                    if word in word2int.keys():
                        int_list.append(word2int[word])
                return int_list


            X = []
            for i in range(len(reviews)):
                X.append(review2intlist(reviews[i]))
            X = sequence.pad_sequences(X, maxlen=self.max_length)


            for batch_no in range(self.num_batches):
                LSTM_inputs = np.zeros(shape=(self.max_length, self.batch_size), dtype=np.float32)
                for i in range(self.batch_size):
                    LSTM_inputs[:, i] = X[batch_no*self.batch_size + i]
                LSTM_inputs = LSTM_inputs.T

                LSTM_outputs = np.zeros(shape=self.batch_size)
                for i in range(self.batch_size):
                    LSTM_outputs[i] = labels[batch_no*self.batch_size + i]

                save_id_review = str("batch" + str(batch_no) + "reviews.npy")
                save_id_label = str("batch" + str(batch_no) + "labels.npy")
                print("batch_no " + str(batch_no))

                if not os.path.isdir(self.save_directory):
                    os.mkdir(self.save_directory)
                rev_save_location = os.path.join(self.save_directory, save_id_review)
                label_save_location = os.path.join(self.save_directory, save_id_label)

                np.save(rev_save_location, LSTM_inputs)
                np.save(label_save_location, LSTM_outputs)
            print("Finished preparing data...")












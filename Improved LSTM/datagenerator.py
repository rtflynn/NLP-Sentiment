import numpy as np
from keras.utils import Sequence
import os
import random
from saveloadpath import get_save_load_path

dirname, _ = os.path.split(os.path.abspath(__file__))

class MyGenerator(Sequence):
    'Generates data for Keras on the fly'
    def __init__(self, num_data_points=10000, batch_size=64, max_length=400,
                 name="name", shuffle=True, vocab_size=3000):
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_data_points = num_data_points
        self.len = int(np.floor(self.num_data_points/float(self.batch_size)))
        self.name = name
        self.vocab_size = vocab_size
        self.load_directory = get_save_load_path(name=self.name, batch_size=self.batch_size,
                                                 vocab_size=self.vocab_size, max_length=self.max_length)
        self.shuffle = shuffle
        self.batch_order = np.arange(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = self.batch_order[index]
        load_id_review = str("batch" + str(index) + "reviews.npy")
        load_id_label = str("batch" + str(index) + "labels.npy")
        rev_save_location = os.path.join(self.load_directory, load_id_review)
        label_save_location = os.path.join(dirname, self.load_directory, load_id_label)

        X = np.load(rev_save_location)
        Y = np.load(label_save_location)

        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.batch_order)
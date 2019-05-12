from keras.layers import Embedding, Dense, LSTM, CuDNNLSTM
from keras import Sequential
#from datagenerator import MyGenerator
#from dataprep import DataPrepare
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Nadam, Adam
from keras.losses import binary_crossentropy, categorical_hinge, mean_squared_error
#from saveloadpath import get_save_load_path
import keras
import os
import time
import talos
from talos.model import lr_normalizer
#import numpy as np
from TalosDataPrep import prepare_data
#import pandas as pd


# default_params = {'vocab_size': [3000, 6000, 9000], 'max_length': [300, 500, 800], 'num_data_points': [100000],
#                   'embedding_size': [64, 128, 256], 'batch_size': [128, 256, 512], 'optimizer': [Adam, Nadam],
#                   'loss': [binary_crossentropy, categorical_hinge, mean_squared_error], 'num_units': [50, 100, 200],
#                   'multiple_LSTM_layers': [False, True], 'lr': [0.001, .01, .1, 1, 10]}

default_params = {'vocab_size': [6000, 9000], 'max_length': [500, 800], 'num_data_points': [100000],
                  'embedding_size': [80, 100, 120], 'batch_size': [64, 128, 256], 'optimizer': [Adam, Nadam],
                  'loss': [binary_crossentropy, categorical_hinge, mean_squared_error],
                  'num_units': [100, 150, 200, 300], 'multiple_LSTM_layers': [False, True], 'lr': [.1, 1, 5, 10]}

dirname, _ = os.path.split(os.path.abspath(__file__))

# Dummy info just to get the thing going... These will go in the throwaway variable slots
x_train, y_train, x_test, y_test, x_valid, y_valid = \
    prepare_data("test.ft.txt", num_data_points=4000, vocab_size=4000, max_length=500)

# Pre-compute all versions of training set - can do this because there's only a few and they're all small.
train_data = {}
for i in default_params['vocab_size']:
    for j in default_params['max_length']:
        for k in default_params['num_data_points']:
            xt, yt, xv, yv, _, _ = prepare_data("test.ft.txt", num_data_points=k, vocab_size=i, max_length=j)
            train_data[(i, j, k)] = xt, yt, xv, yv


class TimeHistory(keras.callbacks.Callback):
    def __init__(self):
        self.times = []

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def AmazonModel(x_train_throwaway, y_train_throwaway, x_val_throwaway, y_val_throwaway, params):
    # Pull quantities from params here for convenience
    # Create our training data inside the model
    batch_size = params['batch_size']
    max_length = params['max_length']
    vocab_size = params['vocab_size']
    loss = params['loss']
    num_data_points = params['num_data_points']
    num_units = params['num_units']
    embedding_size = params['embedding_size']
    multiple_LSTM_layers = params['multiple_LSTM_layers']

    x_train, y_train, x_valid, y_valid, = train_data[(vocab_size, max_length, num_data_points)]

    earlyStopper = EarlyStopping(patience=3, verbose=0, restore_best_weights=True, monitor="val_acc")

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_size, input_length=max_length))
    model.add(CuDNNLSTM(num_units, return_sequences=multiple_LSTM_layers))
    if multiple_LSTM_layers:
        model.add(CuDNNLSTM(num_units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], batch_size=batch_size,
                        epochs=30, callbacks=[earlyStopper], verbose=2)

    return history, model


h = talos.Scan(x_train, y_train, x_val=x_valid, y_val=y_valid,  model=AmazonModel, params=default_params,
           grid_downsample=.01)

r = talos.Reporting(h)
print(r.high('val_acc'))        # The highest validation accuracy across all models
print(r.best_params())          # The top 10 parameter sets
print(r.correlate('val_loss'))
print(r.correlate('val_acc'))
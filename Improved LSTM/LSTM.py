import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras        # Necessary even if greyed out on PyCharm
sys.stderr = stderr
# The previous is SOLELY to get rid of "using tensorflow as backend" messages
from keras.layers import Embedding, Dense, LSTM, CuDNNLSTM
from keras import Sequential
from datagenerator import MyGenerator
from dataprep import DataPrepare
from shutup import tensorflow_shutup
from keras.callbacks import EarlyStopping, ModelCheckpoint
from saveloadpath import get_save_load_path

# Well... Unfortunately it looks like Talos doesn't play well with fit_generator.
# So, we'll have to use Talos with plain-old fit to find some good hyperparameters, and then try
# training a similar model on a larger data set with fit_generator.
# Roundabout but it should work...
tensorflow_shutup()

################################# Instructions ########################################################################
# Run TalosModel.py to do a hyperparameter search.  Change the 'default_params' dictionary in that file to change
# the scope of the hyperparameter search.

# Otherwise run this file to train an LSTM using fit_generator.  Be sure to change the number of workers and
# the use_multiprocessing flag to suit your machine.  Be very careful not to run this file many times with different
# (vocab_size, max_length, batch_size) values since distinct values force the DataPrepare method to create new
# directories on your machine and fill them with the relevant data.  Each directory can hold 3+ GB of data.
# Also, due to laziness I've implemented my dataprep module such that if you want to use data for the same
# vocab_size, max_length, and batch_size, but *larger* num_data_points, you'll need to delete the corresponding
# folder on your hard drive and rerun the dataprep (or this file) with the larger num_data_points value.
#######################################################################################################################


vocab_size = 8000                 # Number of words to Embed - i.e. this determines the size of the one-hot layer
max_length = 650                  # Max length of any review (in words).  We'll pad/cut all reviews to this length.
num_data_points = 3600000         # How many train/test examples to use (the data set is massive)
embedding_size = 100              # How many dimensions to embed our words in
batch_size = 512
num_epochs = 10


mydatapreparer = DataPrepare(batch_size=batch_size, max_length=max_length,
                             num_data_points=num_data_points, vocab_size=vocab_size, name="train")

mydatapreparer.prepare_data("train.ft.txt")

validdatapreparer = DataPrepare(batch_size=batch_size, max_length=max_length,
                                num_data_points=400000, vocab_size=vocab_size, name="test")

validdatapreparer.prepare_data("test.ft.txt")

myGen = MyGenerator(batch_size=batch_size, max_length=max_length, num_data_points=num_data_points,
                    name="train", vocab_size=vocab_size)
validGen = MyGenerator(batch_size=batch_size, max_length=max_length, num_data_points=400000,
                       name="test", vocab_size=vocab_size)


earlyStopper = EarlyStopping(patience=2, verbose=1, restore_best_weights=True)
savepath = get_save_load_path(name="BestModel", batch_size=batch_size,
                                                 vocab_size=vocab_size, max_length=max_length)
if not os.path.isdir(savepath):
    os.mkdir(savepath)

savepath = os.path.join(savepath, "bestmodel.hdf5")
checkpt = ModelCheckpoint(filepath=savepath, save_best_only=True, verbose=1)
# input vocab_size+1 because of the additional 'padding token' 0.
model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_size, input_length=max_length))
model.add(CuDNNLSTM(200, return_sequences=True))
model.add(CuDNNLSTM(200))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


if __name__ == "__main__":
    model.fit_generator(generator=myGen, epochs=30, verbose=2, use_multiprocessing=True,
                        workers=8, max_queue_size=20, validation_data=validGen, callbacks=[earlyStopper, checkpt])



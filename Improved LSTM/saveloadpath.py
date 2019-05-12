import os

dirname, _ = os.path.split(os.path.abspath(__file__))

def get_save_load_path(name = "name", batch_size = 1024, vocab_size = 3000, max_length = 400):
    myPath = os.path.join(dirname, name + str("batchsize" +
                                              str(batch_size) + "voc" + str(vocab_size) + "len" + str(max_length)))
    return myPath


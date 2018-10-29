import numpy as np 
import sys
from random import randint
from time import time
from sklearn.utils import shuffle
import pickle
import os
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# Utility Functions
def make_train_batch(x_train, y_train, batch_size, max_len, corpus):
    # Create random training batch
    if batch_size < len(x_train):
        rand = randint(0, len(x_train) - batch_size - 1)
        arr = x_train[rand:rand + batch_size]
        labels = y_train[rand:rand + batch_size]
    else: 
        arr = x_train
        labels = y_train
    
    # # Reverse order of encoder strings (https://arxiv.org/pdf/1409.3215.pdf)
    # rev_list = list(arr)
    # for i, ex in enumerate(rev_list):
    #     rev_list[i] = list(reversed(ex))
    
    # Create Lagged
    lag_labs = []
    end_index = corpus.index('<END>')
    filler_index = corpus.index('<FILL>')
    for l in labels:
        find_end = np.argwhere(l == end_index)[0]
        shift_lab = np.roll(l, 1)
        shift_lab[0] = end_index
        if (find_end != (max_len - 1)):
            shift_lab[find_end+1] = filler_index
        lag_labs.append(shift_lab)
    
    # Transpose Lists
    #rev_list = np.asarray(rev_list).T.tolist()
    #arr = np.asarray(arr).T
    #labels = labels.T
    lag_labs = np.asarray(lag_labs)
    
    return arr, labels, lag_labs

def get_test_input(message, corpus, max_len):
    encoder_msg = np.full((max_len), corpus.index('<FILL>'), dtype='int32')
    msg_split = message.lower().split()
    for i, word in enumerate(msg_split):
        if word in corpus:
            encoder_msg[i] = corpus.index(word)
        else: 
            continue
    encoder_msg[i + 1] = corpus.index('<END>')
    encoder_msg = encoder_msg[::-1]
    encoder_msg_list=[]
    for n in encoder_msg:
        encoder_msg_list.append([n])
    return encoder_msg_list

def ids_to_sentences(ids, corpus):
    end_index = corpus.index('<END>')
    filler_index = corpus.index('<FILL>')
    full_str = ""
    resp_list=[]
    for num in ids:
        if (num[0] == end_index or num[0] == filler_index):
            resp_list.append(full_str)
            full_str = ""
        else:
            full_str = full_str + corpus[num[0]] + " "
    if full_str:
        resp_list.append(full_str)
    resp_list = [i for i in resp_list if i]
    return resp_list

def define_models(sentence_length, embed_size, vocab_size, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(sentence_length,), name="Encoder_input")
    encoder_lstm = LSTM(n_units, return_state=True, name='Encoder_lstm') 
    Shared_Embedding = Embedding(input_dim=vocab_size, output_dim=embed_size, name="Embedding") 
    word_embedding_context = Shared_Embedding(encoder_inputs) 
    encoder_outputs, state_h, state_c = encoder_lstm(word_embedding_context) 
    encoder_states = [state_h, state_c]
    
    # define training decoder
    decoder_inputs = Input(shape=(sentence_length,), name="Decoder_input")
    decoder_lstm = LSTM(n_units, return_sequences=False, return_state=True, name="Decoder_lstm") 
    word_embedding_answer = Shared_Embedding(decoder_inputs)
    decoder_outputs, _, _ = decoder_lstm(word_embedding_answer, initial_state=encoder_states) 
    decoder_dense = Dense(sentence_length, activation='softmax', name="Dense_layer") 
    decoder_outputs = decoder_dense(decoder_outputs) 
    
    # define training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
    
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states) 

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,), name="H_state_input") 
    decoder_state_input_c = Input(shape=(n_units,), name="C_state_input") 
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c] 
    decoder_outputs, state_h, state_c = decoder_lstm(word_embedding_answer, initial_state=decoder_states_inputs) 
    decoder_states = [state_h, state_c] 
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    # return all models
    return model, encoder_model, decoder_model

# Training
# Load Corpus
with open("corpus.txt", "rb") as c:
    corpus = pickle.load(c)
    
vocab_size = len(corpus)

# Load data
X_tr = np.load('Seq2SeqXTrain.npy')
y_tr = np.load('Seq2SeqYTrain.npy')

n_examples = len(X_tr)

# Hyperparamters
sentence_length = 15
lstm_units = 128
embedding_dim = 128
n_layers = 3
n_iters = 500000

# Create Training Sets
X1, X2, y = make_train_batch(X_tr, y_tr, n_examples, sentence_length, corpus)

# Create and Compile Models
train_mod, encoder_mod, decoder_mod = define_models(sentence_length, embedding_dim, vocab_size, lstm_units)
train_mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Create Callbacks
checkpoint = ModelCheckpoint("synsypa-pass-1-{epoch:02d}-{val_acc:.2f}.hdf5")
tensorboard = TensorBoard(log_dir = "tensorboard/{}".format(time()))

train_mod.fit([X1, X2], y, epochs=500000, callbacks=[tensorboard, checkpoint], batch_size=32)
import numpy as np 
from collections import Counter
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os

# Functions
def process_corpus(message_filepath):
    with open(message_filepath, 'r') as f:
        lines = f.readlines()
        full_str = ""
        for line in lines:
            full_str += line
        #word_dict = Counter(full_str.split())
        corpus = list(set(full_str.split()))
        # Add entries for Filler and End of Message
        corpus.append('<FILL>')
        corpus.append('<END>')
        return corpus
    
def create_training(conversation, corpus, max_len):
    # Initialize empty vectors
    num_convos = len(conversation)
    x_train = np.zeros((num_convos, max_len), dtype='int32')
    y_train = np.zeros((num_convos, max_len), dtype='int32')
    for i,[key,val] in enumerate(conversation):
        # integer representation of strings
        encoder_msg = np.full((max_len), corpus.index('<FILL>'), dtype='int32')
        decoder_msg = np.full((max_len), corpus.index('<FILL>'), dtype='int32')
        # split strings into words
        key_split = key.split()
        val_split = val.split()
        key_n = len(key_split)
        val_n = len(val_split)
        # reject messages that are empty or too long
        if (key_n >= max_len or val_n >= max_len or
            key_n == 0 or val_n == 0):
            continue
        # integerize encoder string and add to array
        for encode_i, word in enumerate(key_split):
            if word in corpus:
                encoder_msg[encode_i] = corpus.index(word)
            else:
                encoder_msg[encode_i] = 0
        encoder_msg[encode_i + 1] = corpus.index('<END>')
        x_train[i] = encoder_msg
        # integerize decoder string and add to array
        for decode_i, word in enumerate(key_split):
            if word in corpus:
                decoder_msg[decode_i] = corpus.index(word)
            else:
                decoder_msg[decode_i] = 0
        decoder_msg[decode_i + 1] = corpus.index('<END>')
        y_train[i] = decoder_msg
    # remove rows with all 0
    x_train = x_train[~np.all(x_train == 0, axis=1)]
    y_train = y_train[~np.all(y_train == 0, axis=1)]
    n_examples = x_train.shape[0]
    return n_examples, x_train, y_train
    
# Create Corpus
corpus = process_corpus('messages.txt')
with open('corpus.txt', 'wb') as c:
    pickle.dump(corpus, c)
    
# Create Training Matrices
conversations = np.load('conversations.npy')
# Using 15 as max encoding length
n_examples, X_tr, y_tr = create_training(conversations, corpus, 15)
np.save('Seq2SeqXTrain.npy', X_tr)
np.save('Seq2SeqYTrain.npy', y_tr)
import tensorflow as tf 
import numpy as np 
import sys
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os

# Functions
def make_train_batch(x_train, y_train, n_examples, batch_size, max_len, corpus):
    rand = randint(0, n_examples - batch_size - 1)
    arr = x_train[rand:rand + batch_size]
    labels = y_train[rand:rand + batch_size]
    # Reverse order of encoder strings (https://arxiv.org/pdf/1409.3215.pdf)
    rev_list = list(arr)
    for i, ex in enumerate(rev_list):
        rev_list[i] = list(reversed(ex))
    # Lag Labels
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
    rev_list = np.asarray(rev_list).T.tolist()
    labels = labels.T.tolist()
    lag_labs = np.asarray(lag_labs).T.tolist()
    
    return rev_list, labels, lag_labs

def make_sentences(inputs, corpus, encoder=False):
    end_index = corpus.index('<END>')
    filler_index = corpus.index('<FILL>')
    n_strings = len(inputs[0])
    n_lengths = len(inputs)
    str_list = [''] * n_strings
    for s in inputs:
        for i, n in enumerate(s):
            if (n != end_index and n != filler_index):
                if (encoder):
                    str_list[i] = corpus[n] + " " + str_list[i]
                else:
                    str_list[i] = str_list[i] + " " + corpus[n]
    str_list = [string.strip() for string in str_list]
    return str_list 

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

## Training
# Hyperparamters
batch_size = 24
max_encode = 15
max_decode = max_encode
lstm_units = 112
embedding_dim = lstm_units
n_layers = 3
n_iters = 500000
word_vec_dim = 200

# Load Corpus
with open("corpus.txt", "rb") as c:
    corpus = pickle.load(c)
vocab_size = len(corpus)

# Load data
X_tr = np.load('Seq2SeqXTrain.npy')
y_tr = np.load('Seq2SeqYTrain.npy')
n_examples = len(X_tr)

tf.reset_default_graph()

# Create the placeholders
encode_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_encode)]
decode_labels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decode)]
decode_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decode)]
feed_prev = tf.placeholder(tf.bool)

encoder_LSTM = tf.nn.rnn_cell.LSTMCell(lstm_units, state_is_tuple=True, name='basic_lstm_cell')

# Create decoder model
decode_output, decode_final_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encode_inputs, decode_inputs,
                                                                                    encoder_LSTM, 
                                                                                    vocab_size, vocab_size,
                                                                                    embedding_dim,
                                                                                    feed_previous=feed_prev)
decode_pred = tf.argmax(decode_output, 2)

loss_w = [tf.ones_like(l, dtype=tf.float32) for l in decode_labels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decode_output, decode_labels, loss_w, vocab_size)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('models/'))
sess.run(tf.global_variables_initializer())

# Uploading results to Tensorboard
tf.summary.scalar('Loss', loss)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Test strings
encoder_test_str = ["sup", 
                    "hi",
                    "oh shit waddup",
                    "that girl was really cute tho",
                    "dang kazunoko"]

zeros = np.zeros((1), dtype='int32')

# Training
for i in range(n_iters):
    encode_train, decode_target_train, decode_input_train = make_train_batch(X_tr, y_tr, n_examples, batch_size, max_encode, corpus)
    feed_dict = {encode_inputs[t]: encode_train[t] for t in range(max_encode)}
    feed_dict.update({decode_labels[t]: decode_target_train[t] for t in range(max_decode)})
    feed_dict.update({decode_inputs[t]: decode_input_train[t] for t in range(max_decode)})
    feed_dict.update({feed_prev: False})

    current_loss, _, pred = sess.run([loss, optimizer, decode_pred], feed_dict=feed_dict)
    
    if (i % 50 == 0):
        print('Current loss:', current_loss, 'at iteration', i)
        summary = sess.run(merged, feed_dict=feed_dict)
        writer.add_summary(summary, i)
    if (i % 25 == 0 and i != 0):
        num = randint(0,len(encoder_test_str) - 1)
        print(encoder_test_str[num])
        input_vector = get_test_input(encoder_test_str[num], corpus, max_encode);
        feed_dict = {encode_inputs[t]: input_vector[t] for t in range(max_encode)}
        feed_dict.update({decode_labels[t]: zeros for t in range(max_decode)})
        feed_dict.update({decode_inputs[t]: zeros for t in range(max_decode)})
        feed_dict.update({feed_prev: True})
        ids = (sess.run(decode_pred, feed_dict=feed_dict))
        print(ids_to_sentences(ids, corpus))
    if (i % 10000 == 0 and i != 0):
        savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)
        

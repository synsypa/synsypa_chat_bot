import discord
from discord.ext import commands
import asyncio
import tensorflow as tf
import numpy as np
import logging
import sys
import os
import pickle

# Configure logging
log = logging.getLogger(__name__)
fmt = logging.Formatter(u'\u241e'.join(['%(asctime)s',
                                        '%(name)s',
                                        '%(levelname)s',
                                        '%(funcName)s',
                                        '%(message)s']))
streamInstance = logging.StreamHandler(stream=sys.stdout)
streamInstance.setFormatter(fmt)
log.addHandler(streamInstance)
log.setLevel(logging.DEBUG)

def log_msg(data):
    """
    Accepts a list of data elements, removes the  u'\u241e'character
    from each element, and then joins the elements using u'\u241e'.
    
    Messages should be constructed in the format:
        
        {message_type}\u241e{data}

    where {data} should be a \u241e delimited row.
    """
    tmp = [d.replace(u'\u241e', ' ') for d in data]
    return u'\u241e'.join(tmp)

# Configure Bot
description = '''
            A Bot to reply as synsypa using RNN ChatBot
            '''

bot = commands.Bot(command_prefix='%', description=description)

@bot.event
@asyncio.coroutine
def on_ready():
    log.info(log_msg(['login', bot.user.name, bot.user.id]))

## Prep Model
# Utility Functions
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

# Load in data structures
with open("corpus.txt", "rb") as c:
    corpus = pickle.load(c)
vocab_size = len(corpus)

# Load in hyperparamters
batch_size = 24
max_encode = 15
max_decode = max_encode
lstm_units = 112
n_layers = 3

# Create placeholders
encode_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_encode)]
decode_labels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decode)]
decode_inputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(max_decode)]
feed_prev = tf.placeholder(tf.bool)

# Model
encoder_LSTM = tf.nn.rnn_cell.LSTMCell(lstm_units, state_is_tuple=True, name='basic_lstm_cell')
#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
decode_output, decode_final_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encode_inputs, decode_inputs,
                                                                                    encoder_LSTM, 
                                                                                    vocab_size, vocab_size,
                                                                                    lstm_units,
                                                                                    feed_previous=feed_prev)
decode_pred = tf.argmax(decode_output, 2)

# Start session and get graph
sess = tf.Session()
#y, variables = model.getModel(encoderInputs, decoderLabels, decoderInputs, feedPrevious)

# Load in pretrained model
saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('models'))
saver.restore(sess, 'models/pretrained_seq2seq.ckpt-300000')
zero_vector = np.zeros((1), dtype='int32')

def respond(input_str):
    input_vector = get_test_input(input_str, corpus, max_encode)
    feed_dict = {encode_inputs[t]: input_vector[t] for t in range(max_encode)}
    feed_dict.update({decode_labels[t]: zero_vector for t in range(max_decode)})
    feed_dict.update({decode_inputs[t]: zero_vector for t in range(max_decode)})
    feed_dict.update({feed_prev: True})
    ids = (sess.run(decode_pred, feed_dict=feed_dict))
    return ' '.join(ids_to_sentences(ids, corpus))

@bot.command(pass_context=True)  
@asyncio.coroutine
def listen(ctx, *text : str):
    log.info(log_msg(['received_request', 
                      'listen',
                      ctx.message.author.name, 
                      ctx.message.channel.name,
                      ' '.join(text)]))

    output = respond(str(text))

    log.info(log_msg(['formatted_self', output]))

    yield from bot.say(output)

    log.info(log_msg(['sent_message', 'me', ctx.message.channel.name]))
    
if __name__=='__main__':
    if os.environ['DISCORD_SYNSYPA_TOKEN']:
        log.info(log_msg(['token_read']))

    log.info(log_msg(['bot_intialize']))
    bot.run(os.environ['DISCORD_SYNSYPA_TOKEN'])

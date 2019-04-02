import discord
from discord.ext import commands
import asyncio
import numpy as np
import torch
import logging
import sys
import os
import re

import torch.nn as nn

import bot_helpers as bh

# Configure torch
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

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
    tmp = [str(d).replace(u'\u241e', ' ') for d in data]
    return u'\u241e'.join(tmp)

# Configure Bot
description = '''
            A Bot to reply as synsypa using PyTorch Seq2Seq ChatBot 
            '''

bot = commands.Bot(command_prefix='&', description=description)

# Build Vocabulary
convos = np.load('chat_data/clean_conversations.npy')
vocab = bh.createVocab(convos, 'synsypa_vocab')

# Load Model
# Configure models
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 32

# Set checkpoint to load from; set to None if starting from scratch
loadModel = 'models/save/synsypa_model_3/2-2_500/20000_checkpoint.tar'

# If loading on same machine the model was trained on
checkpoint = torch.load(loadModel)
# If loading a model trained on GPU to CPU
#checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
vocab.__dict__ = checkpoint['vocab_dict']

log.info(log_msg(['Building encoder and decoder from', loadModel]))

# Initialize word embeddings
embedding = nn.Embedding(vocab.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = bh.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = bh.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout)
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

log.info(log_msg([loadModel, 'built']))

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = bh.GreedySearchDecoder(encoder, decoder)

# Response Generation
def respond(input_str, encoder, decoder, searcher, vocab):
    # Normalize input
    input_str = bh.cleanString(input_str)
    
    # Remove words not in dict
    input_list = []
    for word in input_str.split():
        if word in vocab.word2index.keys():
            input_list.append(word)
    
    if len(input_list) == 0:
        return "Error: No words found in Vocab"
    
    clean_str = ' '.join(input_list)
    
    # evaluate sentence
    output_words = bh.evaluate(encoder, decoder, searcher, vocab, clean_str)
    output_ended = []
    for word in output_words:
        if word == '<END>':
            break
        if word == '<PAD>':
            continue
        else:
            output_ended.append(word)
    
    output_str = ' '.join(output_ended)
    output_str = re.sub(r"\s([?!.])", r"\1", output_str).strip()
    
    return output_str
    
# Discord interactions
@bot.event
async def on_ready():
    log.info(log_msg(['login', bot.user.name, bot.user.id]))    

@bot.command()  
async def listen(ctx, *text : str):
    log.info(log_msg(['received_request', 
                      'listen',
                      ctx.message.author.name, 
                      ctx.message.channel.name,
                      ' '.join(text)]))
        
    output = respond(str(text), encoder, decoder, searcher, vocab)

    log.info(log_msg(['formatted_msg', output]))
    
    if not output:
        output = 'Error: Empty Response'
    
    await ctx.channel.send(output)
    log.info(log_msg(['sent_message', ctx.message.channel.name]))
    
if __name__=='__main__':
    if os.environ['DISCORD_SYNSYPA_TOKEN']:
        log.info(log_msg(['token_read']))

    log.info(log_msg(['bot_intialize']))
    bot.run(os.environ['DISCORD_SYNSYPA_TOKEN'])

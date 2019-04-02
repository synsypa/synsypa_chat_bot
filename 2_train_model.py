import numpy as np
import torch 
import torch.nn as nn
import os
from torch import optim

import bot_helpers as bot

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Actually Train Model
MAX_LENGTH = 18
convos = np.load('chat_data/clean_conversations.npy')
vocab = bot.createVocab(convos, 'synsypa_vocab')
convo_filtered = bot.filterConvos(convos, MAX_LENGTH)

# Configure models
model_name = 'synsypa_model_3'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
# loadFilename = 'models/save/synsypa_model_2/2-2_500/10000_checkpoint.tar'
loadFilename = None

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    vocab.__dict__ = checkpoint['vocab_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(vocab.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = bot.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = bot.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 20000
print_every = 10
save_every = 5000

save_dir = os.path.join("models", "save")

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
bot.trainIters(model_name, vocab, convo_filtered, 
               encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, teacher_forcing_ratio,
               save_dir, n_iteration, batch_size, hidden_size,
               print_every, save_every, clip, loadFilename)
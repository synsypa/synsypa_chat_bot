from logzero import logger
import pickle
import random

import data_loaders as loader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import torchtext
import torch.nn as nn

# tensorboard writer
writer = SummaryWriter(f'runs/synsypa_bot_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}')

# Load conversations 
convos = pickle.load(open('chat_data/clean_conversations_2020-05-25.pkl', 'rb'))
vocab = loader.create_vocab(convos)

# Initialize
hidden_size = 200
embedding = nn.Embedding(len(vocab), hidden_size)

# Split dataset
random.shuffle(convos)
train_convos = convos[:20000]
test_convos = convos[20000:]

train_dataset = loader.ConvoDataset(train_convos, vocab)
test_dataset = loader.ConvoDataset(test_convos, vocab)

train_loader = DataLoader(train_dataset,
                          collate_fn = loader.pad_collate,
                          batch_size = 20,
                          shuffle= True,
                          num_workers = 0)

test_loader = DataLoader(test_dataset,
                        collate_fn = loader.pad_collate,
                        batch_size = 20,
                        shuffle= True,
                        num_workers = 0)
                        
# Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


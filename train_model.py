from logzero import logger
import pickle
import random
from datetime import datetime

import data_loaders as loader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import torchtext
import torch.nn as nn

from model_definitions import EncoderRNN, LuongAttnDecoderRNN

def save_checkpoint(epoch, encoder, decoder, en_optimizer, de_optimizer,
                    loss, vocab, embedding, chk_path):
    save_obj = {'epoch': epoch,
                'encoder': encoder,
                'decoder': decoder,
                'en_optimizer': en_optimizer,
                'de_optimizer': de_optimizer,
                'loss': loss,
                'vocab': vocab,
                'embedding', embedding}
    
    torch.save(save_obj, chk_path)
    return

if __name__ == '__main__':
    # tensorboard writer
    writer = SummaryWriter(f'runs/synsypa_bot_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}')

    # Load conversations 
    convos = pickle.load(open('chat_data/clean_conversations_2020-05-25.pkl', 'rb'))
    vocab = loader.create_vocab(convos)

    # Split dataset
    random.shuffle(convos)
    train_convos = convos[:20000]
    test_convos = convos[20000:]
    batch_size = 20

    train_dataset = loader.ConvoDataset(train_convos, vocab)
    test_dataset = loader.ConvoDataset(test_convos, vocab)

    train_loader = DataLoader(train_dataset,
                            collate_fn = loader.pad_collate,
                            batch_size = batch_size,
                            shuffle= True,
                            num_workers = 0)

    test_loader = DataLoader(test_dataset,
                            collate_fn = loader.pad_collate,
                            batch_size = batch_size,
                            shuffle= True,
                            num_workers = 0)
                            
    # Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Arch Options 
    hidden_size = 200
    attn_model = 'dot'
    encoder_layers = 2
    decoder_layers = 2
    dropout = 0.1

    # Initialize Optimization Options
    clipping = 50.0
    teacher_forcing = 1.0
    learning_rate = 0.00001
    decoder_ratio = 5.0
    
    model_name = f'synsypa_bot_v2_{date.today()}'

    # Initalize Embedding
    embedding = nn.Embedding(len(vocab), hidden_size)

    # Intialize models
    encoder = EncoderRNN(hidden_size, embedding, encoder_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size,
                                  len(vocab), decoder_layers,
                                  dropout)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Initialize optimizers
    criteron = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    en_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                              lr=learning_rate)
    de_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()),
                              lr=learning_rate*decoder_ratio)

    # Training Loop
    start_time = time.time()
    print_loss = 0
    encoder.train()
    decoder.train()

    while epoch < 100:
        
        for i, data in enumerate(train_loader):
            
            # Zero Grad Optimizers
            en_optimizer.zero_grad()
            de_optimizer.zero_grad()

            # Send Inputs to Device
            input_tensor = data[0].to(device)
            input_length = data[1].to(device)
            target_tensor = data[2].to(device)
            target_length = data[3].to(device)

            # Forward Through Encoder
            encoder_output, encoder_hidden = encoder(input_tensor, input_length)

            # Initial Decoder Input (SOS = 1)
            decoder_input = torch.LongTensor([[1 for _ in range(batch_size)]]).to(device)

            # Set initial decoder hidden state to encoder hidden state
            decoder_hidden = encoder_hidden[:decoder.n_layers]

            # Set Teacher Forcing
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            # Forward through decoder
            if use_teacher_forcing:
                for t in range(max(target_length)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_output
                    )

                    # Set Next Input to Current Targegt
                    decoder_input = target_tensor[t]

                    # Calculate Loss
                    loss = criterion(decoder_output, target_tensor[t])
            else:
                for t in range(max(target_length)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_output
                    )

                    # Set Next Input to Current Decoder Output
                    topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor

                    # Calculate Loss
                    loss = criterion(decoder_output, target_tensor[t])



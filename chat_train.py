from logzero import logger
import pickle
import random
from datetime import datetime, date
import time

import chat_dataset as loader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

from synsypanet import Transformer

# Checkpointing
def save_checkpoint(epoch, model, optimizer,
                    loss, vocab, chk_path):
    save_obj = {'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss,
                'vocab': vocab}
    
    torch.save(save_obj, chk_path)
    return

if __name__ == '__main__':
    # Model name
    model_name = f'synsypa_transformer_{date.today()}'
    
    # tensorboard writer
    writer = SummaryWriter(f'runs/{model_name}')

    # Load conversations 
    convos = pickle.load(open('chat_data/clean_conversations_2020-10-20.pkl', 'rb'))
    voc = loader.create_vocab(convos, 3)

    # Split dataset
    random.shuffle(convos)
    train_convos = convos
    batch_size = 64
    max_seq = 40

    train_dataset = loader.ConvoDataset(train_convos, voc)

    train_loader = DataLoader(train_dataset,
                              collate_fn = loader.pad_collate,
                              batch_size = batch_size,
                              shuffle= True,
                              num_workers = 0)
                            
    # Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Arch Options 
    model_dim = 256
    heads = 8
    n_layers = 4
    dropout = 0.1
    epochs = 200

    # Checkpointing
    checkpoint_path = 'models/'

    # Intialize models
    transformer = Transformer(voc, model_dim, n_layers, heads, dropout)
    transformer = transformer.to(device)

    # Apparently some sort of magic param initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Draw Network
    transformer.eval()
    obs_example = next(iter(train_loader))
    # Send Inputs to Device
    ex_input_tensor = obs_example[0]
    ex_target_tensor = obs_example[1]
    # Offset Targets
    ex_target_input = ex_target_tensor[:, :-1]
    ex_target_output = ex_target_tensor[:, 1:]
    # Construct Masks
    ex_input_mask, _, ex_lookahead_mask = loader.make_masks(ex_input_tensor, ex_target_input)

    writer.add_graph(transformer, (ex_input_tensor,
                                   ex_target_input,
                                   ex_input_mask,
                                   ex_lookahead_mask))
    transformer.train()

    # Initialize optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transformer.parameters()), 
                            lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Training Loop
    start_time = time.time()
    transformer.train()

    for epoch in range(epochs):

        epoch_loss = 0.0
        print_loss = 0.0
        
        for i, data in enumerate(train_loader):
            
            # Zero Grad Optimizers
            optimizer.zero_grad()

            # Send Inputs to Device
            input_tensor = data[0]
            target_tensor = data[1]

            # Offset Targets
            target_input = target_tensor[:, :-1]
            target_output = target_tensor[:, 1:]

            # Construct Masks
            input_mask, _, lookahead_mask = loader.make_masks(input_tensor, target_input)

            # Send Everything to device
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            target_input = target_input.to(device)
            target_output = target_output.to(device)
            input_mask = input_mask.to(device)
            lookahead_mask = lookahead_mask.to(device)

            # Forward Through Encoder
            output = transformer(input_tensor, target_input,
                                input_mask, lookahead_mask)

            # Calculate Loss
            loss = criterion(output.permute(0,2,1), target_output)

            # Step
            loss.backward()
            optimizer.step()

            # Track Loss
            print_loss += loss.item()
            epoch_loss += loss.item()

            if i % 100 == 99:
                logger.info(f"{time.time()- start_time :7.2f} s | "
                            f"Epoch: {epoch + 1}, Batches: {i + 1 :4} | "
                            f"Loss: {print_loss / 100 :.5f}")
                writer.add_scalar('Training Loss', print_loss / 100, epoch * len(train_loader) + i)
                print_loss = 0.0

        logger.info(f"Avg Training Batch Time: {(time.time() - start_time)/i :7.2f} s")  
        
        if epoch % 25 == 0:
            save_checkpoint(epoch, transformer, optimizer,
                            epoch_loss/len(train_loader), voc,
                            f"{checkpoint_path}/{model_name}_epoch{epoch}_loss{epoch_loss/len(train_loader):.2f}")
            logger.info(f"Saved model state at epoch {epoch + 1} with loss {epoch_loss/len(train_loader)}")

    logger.info('Complete')
    save_checkpoint(epoch, transformer, optimizer,
                    epoch_loss/len(train_loader), voc,
                    f"{checkpoint_path}/{model_name}_epoch{epochs}_loss{epoch_loss/len(train_loader):.2f}")
    writer.close()

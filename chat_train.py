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
def save_checkpoint(epoch, transformer, optimizer,
                    loss, vocab, chk_path):
    save_obj = {'epoch': epoch,
                'transformer': transformer,
                'optimizer': optimizer,
                'loss': loss,
                'vocab': vocab}
    
    torch.save(save_obj, chk_path)
    return

if __name__ == '__main__':
    # tensorboard writer
    writer = SummaryWriter(f'runs/synsypa_transform_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}')

    # Load conversations 
    convos = pickle.load(open('chat_data/clean_conversations_2020-10-20.pkl', 'rb'))
    vocab = loader.create_vocab(convos)

    # Split dataset
    random.shuffle(convos)
    test_convos = convos[:5000]
    train_convos = convos[5000:]
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
    model_dim = 512
    heads = 8
    n_layers = 2
    dropout = 0.1
    epochs = 100
    
    # Checkpointing
    checkpoint_path = './models/'
    model_name = f'synsypa_transformer_{date.today()}'

    # Intialize models
    transformer = Transformer(vocab, model_dim, n_layers, heads, dropout)
    transformer = transformer.to(device)

    # Apparently some sort of magic param initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Draw Network
    transformer.eval()
    obs_example = next(iter(train_loader))
    input_mask_ex, _, lookahead_ex = loader.make_masks(obs_example)
    writer.add_graph(transformer, (obs_example[0].to(device),
                                   obs_example[1].to(device),
                                   input_mask_ex,
                                   lookahead_ex))
    transformer.train()

    # Initialize optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transformer.parameters()), 
                            lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Training Loop
    start_time = time.time()
    print_loss = 0
    transformer.train()

    for epoch in range(epochs):
        
        for i, data in enumerate(train_loader):
            
            # Zero Grad Optimizers
            optimizer.zero_grad()

            # Send Inputs to Device
            input_tensor = data[0].to(device)
            target_tensor = data[1].to(device)

            # Construct Masks
            input_mask, _, lookahead_mask = loader.make_masks(data)
            input_mask = input_mask.to(device)
            lookahead_mask = lookahead_mask.to(device)

            # Forward Through Encoder
            output = transformer(input_tensor, target_tensor,
                                 input_mask, lookahead_mask)

            # Calculate Loss
            target_tensor_out = data[2].to(device)
            loss = criterion(output.permute(0,2,1), target_tensor_out)

            # Step
            loss.backward()
            optimizer.step()

            # Track Loss
            print_loss = loss.item()

            if i % 100 == 99:
                logger.info(f"{time.time()- start_time :7.2f} s | "
                            f"Epoch: {epoch + 1}, Batches: {i + 1 :4} | "
                            f"Loss: {print_loss / 100 :.5f}")
                writer.add_scalar('Training Loss', print_loss / 100, epoch * len(train_loader) + i)
                print_loss = 0.0

        logger.info(f"Avg Training Batch Time: {(time.time() - start_time)/i :7.2f} s")  

        # Check Eval Example
        with torch.no_grad():
            transformer.eval()
            total_test_loss = 0.0
            effective_batches = 0

            for i, data in enumerate(test_loader):
                if i % 100 == 0:
                    logger.debug(f"Epoch-Batch {epoch + 1}:{i} in test-set evaluation.")
                effective_batches += 1

                # Send Inputs to Device
                test_input_tensor = data[0].to(device)
                test_target_tensor = data[1].to(device)

                # Construct Masks
                test_input_mask, _, test_lookahead_mask = loader.make_masks(data)
                test_input_mask = test_input_mask.to(device)
                test_lookahead_mask = test_lookahead_mask.to(device)

                # Forward Through Encoder
                test_output = transformer(test_input_tensor, test_target_tensor,
                                    test_input_mask, test_lookahead_mask)

                # Calculate Loss
                test_target_tensor_out = data[2].to(device)
                test_loss = criterion(test_output.permute(0,2,1), test_target_tensor_out)
                total_test_loss += test_loss

                # Output Sample Strings
                if i % 100 == 9:
                    test_input_str = loader.tensor_to_str(test_input_tensor[0], vocab)
                    logger.info(f"Sample Input: {test_input_str}")

                    test_target_str = loader.tensor_to_str(test_target_tensor[0], vocab)
                    logger.info(f"Sample Target: {test_target_str}")
                    
                    softmax_output = F.softmax(test_output[0], dim=-1)
                    _, test_output_tensor = softmax_output.data.topk(1)
                    test_output_str = loader.tensor_to_str(test_output_tensor.squeeze(), vocab)
                    logger.info(f"Sample Output: {test_output_str}")

        transformer.train()
        logger.info(f"Epoch {epoch + 1}, total test loss: {total_test_loss/effective_batches:.4f}")  


        # Save Checkpoint
        if epoch % 10 == 0:
            save_checkpoint(epoch, transformer, optimizer,
                            print_loss/10, vocab,
                            f"{checkpoint_path}/{model_name}_epoch{epoch}_loss{print_loss/10}")
            logger.info(f"Saved model state at epoch {epoch + 1} with loss {print_loss/10}")

writer.close()

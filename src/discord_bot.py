import discord
from discord.ext import commands
import asyncio
import torch
import os
from logzero import logger

import src.chat_parse as p
from src.synsypanet import Transformer

# Logging Helper
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

# Configure torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Checkpoint
model_dim = 256
heads = 8
n_layers = 4
dropout = 0.1

checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'bin', 'models')
checkpoint = torch.load(os.path.join(checkpoint_path, 'synsypa_transformer_2020-10-29_epoch200_loss0.18'),
                        map_location=device)
vocab_chk = checkpoint['vocab']

net = Transformer(vocab_chk, model_dim, n_layers, heads, dropout)
net.load_state_dict(checkpoint['model_state'])
net.eval()

    
# Discord interactions
# Configure Bot
description = '''
            A Bot to reply as synsypa using PyTorch Seq2Seq ChatBot 
            '''

bot = commands.Bot(command_prefix=commands.when_mentioned_or('&'), description=description)


@bot.event
async def on_ready():
    logger.info(log_msg(['login', bot.user.name, bot.user.id]))    

@bot.command()  
async def hey(ctx, *text : str):
    logger.info(log_msg(['received_request', 
                        'listen',
                        ctx.message.author.name, 
                        ctx.message.channel.name,
                        ' '.join(text)]))
        
    parsed_input, output = p.gen_reply(str(text), net, vocab_chk)

    logger.info(log_msg(['parsed_input', parsed_input]))
    
    if not output:
        output = 'Error: Empty Response'
    
    logger.info(log_msg(['formatted_msg', output]))

    await ctx.channel.send(output)
    logger.info(log_msg(['sent_message', ctx.message.channel.name]))
    
if __name__=='__main__':
    if os.environ['DISCORD_SYNSYPA_TOKEN']:
        logger.info(log_msg(['token_read']))

    logger.info(log_msg(['bot_intialize']))
    bot.run(os.environ['DISCORD_SYNSYPA_TOKEN'])

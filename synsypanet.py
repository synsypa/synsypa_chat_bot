import math
from copy import deepcopy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, dropout=0.1, max_len=128):
        
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.model_dim = model_dim

        pos_encode = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float) * 
                            (-math.log(10000.0) / model_dim))

        pos_encode[:, 0::2] = torch.sin(position * denom)
        pos_encode[:, 1::2] = torch.cos(position * denom)

        pos_encode = pos_encode.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pos_encode)

    def forward(self, x):
        # expand input
        x = x * math.sqrt(self.model_dim)
        # add positional constant
        x = x + Variable(self.pe[:x.size(0), :], requires_grad=False)
        return self.dropout(x)
        

class MultiheadAttention(nn.Module):

    def __init__(self, n_heads, model_dim, dropout=0.1):
        
        super(MultiheadAttention, self).__init__()

        self.model_dim = model_dim
        self.d_k = model_dim // n_heads
        self.h = n_heads

        self.q_lin = nn.Linear(model_dim, model_dim)
        self.k_lin = nn.Linear(model_dim, model_dim)
        self.v_lin = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

        self.out_lin = nn.Linear(model_dim, model_dim)

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)

        # Apply linear layer, reshape to 
        # batch_size * seq_length * heads * (model_dim / heads)
        q = self.q_lin(q).view(batch_size, -1, self.h, self.d_k)
        k = self.k_lin(k).view(batch_size, -1, self.h, self.d_k)
        v = self.v_lin(v).view(batch_size, -1, self.h, self.d_k)

        # transpose batch_size * heads * seq_length * d_k
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention score
        scores = self._attention(q, k, v, mask)

        # concatenate heads
        concat = (
            scores
            .transpose(1,2)
            .contiguous()
            .view(batch_size, -1, self.model_dim)
        )

        output = self.out_lin(concat)

        return output

    def _attention(self, q, k, v, mask):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)
        
        output = torch.matmul(scores, v)

        return output

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim=2048, dropout=0.1):

        super(FeedForward, self).__init__() 
        
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class BatchNorm(nn.Module):
    def __init__(self, model_dim):

        super(BatchNorm, self).__init__()

        self.bn = nn.BatchNorm1d(model_dim)

    def forward(self, x):
        
        x = x.permute(0,2,1)
        x = self.bn(x)
        x = x.permute(0,2,1)

        return x

def stack_layers(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    '''
    1 Multihead attention layer
    1 Feed Forward
    '''
    def __init__(self, model_dim, n_heads, dropout=0.1):

        super(EncoderLayer, self).__init__()

        self.bn1 = BatchNorm(model_dim)
        self.attn = MultiheadAttention(n_heads, model_dim, dropout)
        self.do1 = nn.Dropout(dropout)

        self.bn2 = BatchNorm(model_dim)
        self.ff = FeedForward(model_dim, dropout=dropout)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        
        x_norm = self.bn1(x)
        x = x + self.do1(self.attn(x_norm, x_norm, x_norm, mask))
        x_norm = self.bn2(x)
        x = x + self.do2(self.ff(x_norm))

        return x


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, model_dim, N, n_heads, dropout):
        
        super(Encoder, self).__init__()

        self.N = N
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.pos = PositionalEncoding(model_dim)
        self.layers = stack_layers(EncoderLayer(model_dim, n_heads, dropout), N)
        self.bn = BatchNorm(model_dim)

    def forward(self, source, mask):
        
        x = self.embed(source)
        x = self.pos(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.bn(x)

        return x

class DecoderLayer(nn.Module):
    '''
    2 Multihead Attention layers
    1 Feed Forward
    '''
    def __init__(self, model_dim, n_heads, dropout=0.1):

        super(DecoderLayer, self).__init__()

        self.bn1 = BatchNorm(model_dim)
        self.attn1 = MultiheadAttention(n_heads, model_dim, dropout)
        self.do1 = nn.Dropout(dropout)

        self.bn2 = BatchNorm(model_dim)
        self.attn2 = MultiheadAttention(n_heads, model_dim, dropout)
        self.do2 = nn.Dropout(dropout)

        self.bn3 = BatchNorm(model_dim)
        self.ff = FeedForward(model_dim, dropout=dropout)
        self.do3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, input_mask, lookahead_mask):

        x_norm = self.bn1(x)
        x = x + self.do1(self.attn1(x_norm, x_norm, x_norm, lookahead_mask))
        x_norm = self.bn2(x)
        x = x + self.do2(self.attn2(x_norm, e_outputs, e_outputs, input_mask))
        x_norm = self.bn3(x)
        x = x + self.do3(self.ff(x_norm))

        return x

class Decoder(nn.Module):

    def __init__(self, vocab_size, model_dim, N, n_heads, dropout):

        super(Decoder, self).__init__()

        self.N = N
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.pos = PositionalEncoding(model_dim)
        self.layers = stack_layers(DecoderLayer(model_dim, n_heads, dropout), N)
        self.bn = BatchNorm(model_dim)

    def forward(self, target, e_outputs, input_mask, lookahead_mask):
        
        x = self.embed(target)
        x = self.pos(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, input_mask, lookahead_mask)
        x = self.bn(x)

        return x

class Transformer(nn.Module):

    def __init__(self, vocab, model_dim, N, n_heads, dropout):

        super(Transformer, self).__init__()
        vocab_size = len(vocab)
        self.encoder = Encoder(vocab_size, model_dim, N, n_heads, dropout)
        self.decoder = Decoder(vocab_size, model_dim, N, n_heads, dropout)
        self.fc1 = nn.Linear(model_dim, vocab_size)

    def forward(self, call, response, input_mask, lookahead_mask):
        
        e_outputs = self.encoder(call, input_mask)
        d_output = self.decoder(response, e_outputs, input_mask, lookahead_mask)
        output = self.fc1(d_output)

        return output


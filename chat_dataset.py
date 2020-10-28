import pickle

from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext import vocab

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def create_vocab(convos, max_freq=5):
    # flatten convos
    flat = [text for pair in convos for text in pair]

    counter = Counter(' '.join(flat).split())

    voc = vocab.Vocab(counter, min_freq=max_freq,
                    specials=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])

    return voc

class ConvoDataset(Dataset):
    """ Dataset class of conversations """

    def __init__(self, convos, vocab):

        self.convos = convos
        self.vocab = vocab

    def __len__(self):

        return(len(self.convos))

    def __getitem__(self, idx):

        call, response = self.convos[idx]

        call_tensor = self._to_index(call)
        response_tensor = self._to_index(response)

        return call_tensor, response_tensor

    def _to_index(self, string):
        
        tokens = [SOS_TOKEN]
        tokens += string.split()
        tokens.append(EOS_TOKEN)

        token_idx = [self.vocab.stoi[word]
                    if word in self.vocab.stoi
                    else self.vocab.stoi[UNK_TOKEN]
                    for word in tokens]

        return torch.tensor(token_idx)

def pad_collate(batch):
    (xx, yy) = zip(*batch)

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad

def make_masks(input_tensor, target_tensor):
    input_mask = (input_tensor != 0).type(torch.uint8).unsqueeze(1).unsqueeze(1) # (batch, 1, 1, seq)

    target_mask = (target_tensor != 0).type(torch.uint8) # (batch, seq)
    
    target_sz = target_tensor.size(1)
    lookahead_mask = target_mask.unsqueeze(1)
    nopeek_mask = (torch.triu(torch.ones(target_sz, target_sz)) == 1).transpose(0,1)
    lookahead_mask = lookahead_mask & nopeek_mask # (batch, seq, seq)
    lookahead_mask = lookahead_mask.unsqueeze(1) # (batch, 1, seq, seq)

    #nopeak_mask = target_mask & nopeak_mask
    return input_mask, target_mask, lookahead_mask

def tensor_to_str(tensor, vocab):
    str_list = []
    for i in tensor:
        w = vocab.itos[i]
        str_list.append(w)
        if w == '<pad>':
            break
    return ' '.join(str_list)
        
if __name__ == "__main__":
    convos = pickle.load(open('chat_data/clean_conversations_2020-10-20.pkl', 'rb'))

    vocab = create_vocab(convos)
    test = ConvoDataset(convos, vocab)

    print(test[0])
    print(type(test[0]))
    print('\n')
    print([vocab.itos[i] for i in test[0][0]])
    print([vocab.itos[i] for i in test[0][1]])

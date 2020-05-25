import pickle

from collections import Counter

import torch
from torch.utils.data import Dataset
from torchtext import vocab

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def create_vocab(convos):
    # flatten convos
    flat = [text for pair in convos for text in pair]

    counter = Counter(' '.join(flat).split())

    voc = vocab.Vocab(counter, min_freq=3,
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
        
        tokens = string.split()
        tokens.append(EOS_TOKEN)

        token_idx = [self.vocab.stoi[word]
                    if word in self.vocab.stoi
                    else self.vocab.stoi[UNK_TOKEN]
                    for word in tokens]

        return torch.tensor(token_idx)

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens

        
if __name__ == "__main__":
    convos = pickle.load(open('chat_data/clean_conversations_2020-05-25.pkl', 'rb'))

    vocab = create_vocab(convos)

    test = ConvoDataset(convos, vocab)

    print(test[0])
    print(type(test[0]))
    print('\n')
    print([vocab.itos[i] for i in test[0][0]])
    print([vocab.itos[i] for i in test[0][1]])

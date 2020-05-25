import pickle

from collections import Counter

import torch
from torch.utils.data import Dataset
from torchtext import vocab

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def create_vocab(convos):
    # flatten convos
    flat = [text for pair in convos for text in pair]

    counter = Counter(' '.join(flat).split())

    voc = vocab.Vocab(counter, min_freq=3,
                      specials=[SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN])
    
    return voc

class ConvoDataset(Dataset):
    """ Dataset class of conversations """

    def __init__(self, convos, vocab, max_size):

        self.convos = convos
        self.vocab = vocab
        self.max_size = max_size

    def __len__(self):

        return(len(self.convos))

    def __getitem__(self, idx):

        call, response = self.convos[idx]

        call_tensor = self._vectorize(call)
        response_tensor = self._vectorize(response)

        return {'call': call_tensor,
                'response': response_tensor} 

    def _vectorize(self, string):
        
        tokens = string.split()
        tokens.append(EOS_TOKEN)

        token_idx = [self.vocab.stoi[word]
                    if word in self.vocab.stoi
                    else self.vocab.stoi[UNK_TOKEN]
                    for word in tokens]

        text_dim = min(len(token_idx), self.max_size)
        text_tensor_padded = torch.zeros(self.max_size, dtype = torch.long)
        text_tensor_padded[ :text_dim] = torch.tensor(token_idx[ :text_dim])

        if text_dim < (self.max_size - 1):
            text_tensor_padded[text_dim: ] = self.vocab.stoi[PAD_TOKEN]

        return text_tensor_padded  





if __name__ == "__main__":
    convos = pickle.load(open('chat_data/clean_conversations_2020-05-25.pkl', 'rb'))

    vocab = create_vocab(convos)

    test = ConvoDataset(convos, vocab, 10)

    print(test[0])
    print('\n')
    print([vocab.itos[i] for i in test[0]['call']])
    print([vocab.itos[i] for i in test[0]['response']])

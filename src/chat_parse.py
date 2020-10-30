import os
import torch

import torch.nn.functional as F

from chat_dataset import EOS_TOKEN, SOS_TOKEN, UNK_TOKEN, make_masks
from clean_chats import clean_string
from synsypanet import Transformer


def tensor_to_str(tensor, vocab):

    str_list = []
    for i in tensor:
        w = vocab.itos[i]
        str_list.append(w)

    return ' '.join(str_list)


def predict_reply(sentence, net, voc, max_seq=40):

    input_tokens = [SOS_TOKEN]
    input_tokens += sentence.split()
    input_tokens.append(EOS_TOKEN)
    input_tokens = [voc.stoi[word]
                    if word in voc.stoi
                    else voc.stoi[UNK_TOKEN]
                    for word in input_tokens]
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)
    
    output_tokens = [voc.stoi[SOS_TOKEN]]
    output_tensor = torch.tensor(output_tokens).unsqueeze(0)

    # Construct Masks
    input_mask, _, lookahead_mask = make_masks(input_tensor, output_tensor)

    for i in range(max_seq):
        predictions = net(input_tensor, output_tensor,
                            input_mask, lookahead_mask)
        predictions = predictions[:, -1, :]

        softmax_output = F.softmax(predictions, dim=-1)
        _, result_tensor = softmax_output.data.topk(1)

        # return the result if the predicted_id is equal to the end token
        if result_tensor[0] == voc.stoi[EOS_TOKEN]:
            break

        # concatenated the predicted_id to the output 
        output_tensor = torch.cat([output_tensor, result_tensor], axis=-1)

    return output_tensor


def gen_reply(sentence, net, voc, max_seq=40):

    input_str = clean_string(sentence)
    prediction = predict_reply(sentence, net, voc, max_seq)
    output_str = tensor_to_str(prediction[0][1:], voc)

    return input_str, output_str


if __name__ == '__main__':
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'bin', 'models')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TKTK: Checkpointing should save these values
    model_dim = 256
    heads = 8
    n_layers = 4
    dropout = 0.1
    model_name = 'synsypa_transformer_2020-10-29_epoch200_loss0.18'
    
    checkpoint = torch.load(os.path.join(checkpoint_path, model_name),
                            map_location=device)
    vocab_chk = checkpoint['vocab']

    net = Transformer(vocab_chk, model_dim, n_layers, heads, dropout)
    net.load_state_dict(checkpoint['model_state'])
    net.eval()

    test_input, test_output = gen_reply("who even?", net, vocab_chk)

    print(test_input)
    print(test_output)
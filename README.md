# synsypa_chat_bot
Seq2Seq Transformer Discord Chatbot trained on single user's discord chat history

Fully refactored with a Seq2Seq Transformer architecture trained on sentence pairs 
from Discord where wynsypa is always the respondant.

Discord integration recieved an input sentence and generates a predicted output
in synsypa's style

Trained over 200 epochs on Google Colab

To Dos:

- Different Generation
    - less greedy
    - manage `<unk>` better
- Iterate Hyperparams to adjust fit
- Fine Tune a Pre-trained model (BERT? GPT-2?)
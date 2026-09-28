# %% Importings

import torch
import torch.nn as nn 

# importings from data.py
from data import (
    training_data,
    IT_SOS_INDEX,
    it_index_to_word
)

# importings from model.py
from model import (EncoderRNN, DecoderRNN)

print("Everything has been successfully imported\n")

# %% Creating the models

#defining the embedding size
embedding_size=8
# defining the hidden size
hidden_size=16

# defining the encoder
encoder = EncoderRNN(vocab_size=14, embedding_size=embedding_size, hidden_size=hidden_size)
print("Encoder successfully defined\n")
# defining the decoder
decoder = DecoderRNN(vocab_size=12, embedding_size=embedding_size, hidden_size=hidden_size)
print("Decoder successfully defined\n")

from data import (
    EN_VOCAB_SIZE,
    IT_VOCAB_SIZE,
)
print("English and Italian vocabularies successfully imported\n")

from model import (
    encoder,
    decoder
)
print("Encoder and Decoder successfully imported\n")
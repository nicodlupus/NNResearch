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

# %% Defining the loss function
criterion = nn.CrossEntropyLoss() # how wrong was the predicted italian word

## %% Defining the optimizer
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.01
)

# %% Training example

# defining the language tensors
english_tensor, italian_tensor = training_data[10]
print("English:")
print(english_tensor)

print("Italian:")
print(italian_tensor)

# adding the batch dimensions
english_tensor = english_tensor.unsqueeze(0)

# resetting the gradients
optimizer.zero_grad()

# encoding the english sentence
encoder_outputs, encoder_hidden = encoder(
    english_tensor
)

# starting the decoder
decoder_input = torch.tensor(
    [[IT_SOS_INDEX]],
    dtype=torch.long
)

decoder_hidden = encoder_hidden

# teacher forcing
loss = 0.0

for target_token in italian_tensor:

    predictions, decoder_hidden = decoder(
        decoder_input,
        decoder_hidden
    )

    prediction_scores = predictions[:, 0, :]

    target = target_token.unsqueeze(0)

    loss += criterion(
        prediction_scores,
        target
    )

    decoder_input = target.view(1, 1)
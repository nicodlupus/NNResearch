# %% IMPORTINGS
import torch

from data import (
    EN_VOCAB_SIZE,
    IT_VOCAB_SIZE,
    IT_SOS_INDEX,
    IT_EOS_INDEX,
    en_word_to_index,
    it_index_to_word,
    sentence_to_tensor
)

from model import (
    EncoderRNN,
    DecoderRNN
)

# %% Recreating the structure
# embedding size
embedding_size=8
# hidden size
hidden_size=16

encoder = EncoderRNN(
    EN_VOCAB_SIZE,
    embedding_size,
    hidden_size
)

decoder = DecoderRNN(
    IT_VOCAB_SIZE,
    embedding_size,
    hidden_size
)

# %% Loading the weights

# encoder
encoder.load_state_dict(torch.load("encoder.pth"))


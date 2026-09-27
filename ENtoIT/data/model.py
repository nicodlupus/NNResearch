# %% IMPORTING
import torch 
import torch.nn as nn 

# from data.py import the necessary methods
from data import (
    EN_VOCAB_SIZE,
    EN_PAD_INDEX,
    sentence_to_tensor,
    en_word_to_index
)

# ------------------------------------------------------------------------------
# %% ENCODER
class EncoderRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=EN_PAD_INDEX
        )

        self.rnn = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        def forward(self, x):
            embedded = self.embeddeding(x)
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden
# %% TEST ENCODER
embedding_size = 8
hidden_size = 16

encoder = EncoderRNN(
    vocab_size=EN_VOCAB_SIZE,
    embedding_size=embedding_size,
    hidden_size=hidden_size
)

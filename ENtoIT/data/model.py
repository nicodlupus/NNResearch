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
# %% IMPORTING
import torch 
import torch.nn as nn 

# from data.py import the necessary methods
from data import (
    EN_VOCAB_SIZE,
    EN_PAD_INDEX,
    IT_EOS_INDEX,
    it_index_to_word,
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
        embedded = self.embedding(x)
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

sentence = "i want to drink coffee"

input_tensor = sentence_to_tensor(
    sentence,
    en_word_to_index
)

print(input_tensor)
print(input_tensor.shape)

input_tensor = input_tensor.unsqueeze(0)

print(input_tensor.shape)

outputs, hidden = encoder(input_tensor)

print("\nEncoder outputs:")
print(outputs.shape)

print("\nFinal hidden state:")
print(hidden.shape)
# %%

"""
Importing the Italian Information and creating the italian decoder """

# import from data.py
from data import (
    EN_VOCAB_SIZE,
    EN_PAD_INDEX,
    IT_VOCAB_SIZE,
    IT_PAD_INDEX,
    IT_SOS_INDEX,
    sentence_to_tensor,
    en_word_to_index
)

# DECODER
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=IT_PAD_INDEX
        )
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        
        output, hidden = self.rnn(embedded, hidden)
        predictions = self.output_layer(output)
        
        return predictions, hidden
    
# %% Testing the decoder
decoder = DecoderRNN(
    vocab_size=IT_VOCAB_SIZE,
    embedding_size=embedding_size,
    hidden_size=hidden_size
)

# starting the <SOS>
decoder_input = torch.tensor(
    [[IT_SOS_INDEX]],
    dtype=torch.long
)

print(decoder_input) # one sentence
print(decoder_input.shape) # one token

# %% Passing the encoder memory into the decoder

predictions, decoder_hidden = decoder(
    decoder_input,
    hidden # represents the English language
)

print("Prediction shape:")
print(predictions.shape)

print("Decoder hidden shape:")
print(decoder_hidden.shape)

# %% testing predictions
# the model is still untrained
from data import it_index_to_word

predicted_index = predictions.argmax(dim=-1)
print("Predicted index:")
print(predicted_index)

index = predicted_index.item()

predicted_word = it_index_to_word[index]

print("Predicted word:")
print(predicted_word)

# %% GENERATE A TRANSLATION

decoder_input = torch.tensor(
    [[IT_SOS_INDEX]],
    dtype=torch.long
)

decoder_hidden = hidden

generated_words = []

max_length = 10 # emergency limit

for step in range(max_length):

    predictions, decoder_hidden = decoder(
        decoder_input,
        decoder_hidden
    )

    predicted_index = predictions.argmax(dim=-1)

    index = predicted_index.item()

    predicted_word = it_index_to_word[index]

    print(
        f"Step {step + 1}: "
        f"{predicted_word}"
    )

    if index == IT_EOS_INDEX:
        break

    generated_words.append(predicted_word)

    decoder_input = predicted_index
    
print("\nGenerated translation:")
print(" ".join(generated_words))
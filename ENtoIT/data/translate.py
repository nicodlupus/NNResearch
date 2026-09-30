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

# decoder
decoder.load_state_dict(torch.load("decoder.pth"))

encoder.eval()
decoder.eval()

def translate(sentence, max_length=10):

    input_tensor = sentence_to_tensor(
        sentence,
        en_word_to_index
    )

    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():

        _, encoder_hidden = encoder(
            input_tensor
        )

        decoder_input = torch.tensor(
            [[IT_SOS_INDEX]],
            dtype=torch.long
        )

        decoder_hidden = encoder_hidden

        generated_words = []

        for _ in range(max_length):

            predictions, decoder_hidden = decoder(
                decoder_input,
                decoder_hidden
            )

            predicted_index = predictions.argmax(
                dim=-1
            )

            index = predicted_index.item()

            if index == IT_EOS_INDEX:
                break

            word = it_index_to_word[index]

            generated_words.append(word)

            decoder_input = predicted_index

    return " ".join(generated_words)
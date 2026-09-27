"""
In this file, small datasets are created as examples to be used by the\
Neural Network algs
"""

import torch
import torch.nn as nn # importing the neural networks methods

print("1: imports done")

# defining the random nums reproducible
torch.manual_seed(42)

# ADD SPECIAL TOKENS
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN
]

# small parallel dataset
sentence_pairs = [
    ("i drink coffee", "bevo caffe"),
    ("i drink water", "bevo acqua"),
    ("you drink coffee", "bevi caffe"),
    ("you drink water", "bevi acqua"),
    ("he drinks coffee", "beve caffe"),
    ("she drinks water", "beve acqua"),

    ("i want coffee", "voglio caffe"),
    ("i want water", "voglio acqua"),
    ("you want coffee", "vuoi caffe"),
    ("you want water", "vuoi acqua"),

    ("i want to drink coffee", "voglio bere caffe"),
    ("i want to drink water", "voglio bere acqua"),
    ("you want to drink coffee", "vuoi bere caffe"),
    ("you want to drink water", "vuoi bere acqua"),
]

for en, it in sentence_pairs:
    print(f"{en:<30}->{it}")
    
print("2: dataset done")
    
"""
At this point we have to turn the words into numbers. Computers need words 
reduced to int to cooperate
"""

# building the vocab 

# tokenization function
def tokenize(sentence):
    return sentence.lower().split() # splitting the sentence into collection of words

en_words = set()
it_words = set()

for en, it in sentence_pairs:
    # splitting the english sentences into a collection of words
    en_words.update(tokenize(en))
    # splitting the italian sentences into a collection of words
    it_words.update(tokenize(it))
    
# printing the english vocab
print("Below is the English vocabulary:\n")
print(en_words)
print("\n")

# printing the italian vocab
print("Below is the Italian vocabulary:\n")
print(it_words)
print("\n")

# assigning an integer to each word in the collection
en_word_to_index = {
    word:index
    for index, word in enumerate(SPECIAL_TOKENS + sorted(en_words))
}

it_word_to_index = {
    word:index 
    for index, word in enumerate(SPECIAL_TOKENS + sorted(it_words))
}

# printing the index results
print("English word -> Index:\n")
print(en_word_to_index)
print("\n")
print("Italian word -> Index:\n")
print(it_word_to_index)
print("\n")

# encoding function
def encode_sentence(sentence, word_to_index):
    tokens = tokenize(sentence)
    
    # adding encoded
    encoded = [
        word_to_index.get(word, word_to_index[UNK_TOKEN])
        for word in tokens
    ]
    
    encoded.append(word_to_index[EOS_TOKEN])
    return encoded
    
example = "i want to drink coffee"

encoded = encode_sentence(
    example,
    en_word_to_index
)

print("\nOriginal:")
print(example)

print("Encoded:")

"""
Embeddings
"""

# converting an encoded sentence into a tensor
input_tensor = torch.tensor(encoded, dtype=torch.long)
print(input_tensor)
print(input_tensor.shape) # five words plus the EOS token

# defining the embedding layer
vocabulary_size = len(en_word_to_index)
# defining the embedding size
embedding_size=4

# creating the embedding
embedding=nn.Embedding(
    num_embeddings=vocabulary_size,
    embedding_dim=embedding_size
) # each token, including EOS, gets four embedding values

# creating the embedded sentence
embedded_sentence = embedding(input_tensor)
print(embedded_sentence)
print(embedded_sentence.shape)

coffee_index = en_word_to_index["coffee"]

print("Coffee index:")
print(coffee_index)

print("\nCoffee embedding:")
print(embedding.weight[coffee_index])

# defining the recurrent neural network input ([sentence, words, num/word])
rnn_input = embedded_sentence.unsqueeze(0)
print(rnn_input.shape)

hidden_size=8
rnn=nn.RNN(
    input_size=embedding_size, # each word embedding contains four nums
    hidden_size=hidden_size, # RNNs internal memo contains 8 nums
    batch_first=True
)

# feeding the sentence
outputs, final_hidden = rnn(rnn_input)

print("Outputs shape:")
print(outputs.shape)

print("\nFinal hidden shape:")
print(final_hidden.shape)


# adding the index to word collections
en_index_to_word = {
    index: word
    for word, index in en_word_to_index.items()
}

it_index_to_word = {
    index: word 
    for word, index in it_word_to_index.items()
}

# creating the decoding mechanism
def decode_sentence(indices, index_to_word):
    words = []
    
    for index in indices:
        word = index_to_word[index]
        
        if word == EOS_TOKEN:
            break
        
        if word not in {PAD_TOKEN, SOS_TOKEN}:
            words.append(word)
            
    return " ".join(words)

# TESTING ENCODER-DECODER
print("BELOW IS THE TEST OF THE DECODER AND ENCODER:\n")
encoded = encode_sentence(
    "i want to drink coffee",
    en_word_to_index
)
print(encoded)

decoded = decode_sentence(
    encoded,
    en_index_to_word
)
print(decoded)

def sentence_to_tensor(sentence, word_to_index):
    encoded = encode_sentence(sentence, word_to_index)

    return torch.tensor(
        encoded,
        dtype=torch.long
    )
    
training_data = []

for english, italian in sentence_pairs:

    english_tensor = sentence_to_tensor(
        english,
        en_word_to_index
    )

    italian_tensor = sentence_to_tensor(
        italian,
        it_word_to_index
    )

    training_data.append(
        (english_tensor, italian_tensor)
    )
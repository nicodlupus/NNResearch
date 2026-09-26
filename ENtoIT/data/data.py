"""
In this file, small datasets are created as examples to be used by the\
Neural Network algs
"""

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
    for index, word in enumerate(sorted(en_words))
}

it_word_to_index = {
    word:index 
    for index, word in enumerate(sorted(it_words))
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
    
    return [
        word_to_index[word]
        for word in tokens
    ]
    
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

import torch 
import torch.nn as nn # importing the neural networks methods

# converting an encoded sentence into a tensor
input_tensor = torch.tensor(encoded, dtype=torch.long)
print(input_tensor)
print(input_tensor.shape) # sequence length = 5 in this case because we have five words

# defining the embedding layer
vocabulary_size = len(en_word_to_index)
# defining the embedding size
embedding_size=4

# creating the embedding
embedding=nn.Embedding(
    num_embeddings=vocabulary_size,
    embedding_dim=embedding_size
) # 5 words x 4 numbers representing each of the words

# creating the embedded sentence
embedded_sentence = embedding(input_tensor)
print(embedded_sentence)
print(embedded_sentence.shape)

coffee_index = en_word_to_index["coffee"]

print("Coffee index:")
print(coffee_index)

print("\nCoffee embedding:")
print(embedding.weight[coffee_index])
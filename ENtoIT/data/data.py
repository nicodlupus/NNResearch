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

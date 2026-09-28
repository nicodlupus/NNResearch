from data import (
    sentence_to_tensor,
    en_word_to_index,
    it_index_to_word,
    IT_EOS_INDEX
)

from data import (
    EN_VOCAB_SIZE,
    IT_VOCAB_SIZE,
)

from data import (
    training_data,
    IT_SOS_INDEX,
    it_index_to_word
)

from train import (translate)

print(translate("you drink water"))
print(translate("he drinks coffee"))
print(translate("i want coffee"))
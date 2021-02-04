import pandas as pd
import numpy as np


def build_vocab(all_words, min_feq):
    # use collections.Counter() to build vocab
    all_words = all_words.most_common()
    word2ix = {'<sos>': 0, '<eos>': 1, '<unk>': 2}
    for ix, (word, feq) in enumerate(all_words, start=3):
        if feq < min_feq:
            break
        word2ix[word] = ix
    ix2word = {v: k for k, v in word2ix.items()}

    # output info
    print('number of words in vocab: {}'.format(len(word2ix)))
    print('number of <unk> in vocab: {}'.format(len(all_words) - len(word2ix)))

    return word2ix, ix2word


def


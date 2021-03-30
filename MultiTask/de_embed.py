# import torch
# import torch.nn as nn
# import pathlib as plb
# import numpy as np
# import pickle
# from tqdm import tqdm
#
# cap_input_path = '/media/omnisky/115a4abd-1149-406e-a3de-2c3a5d707b69/lzh/dual/data/cap_input'
# embed_weigth_path = '/media/omnisky/115a4abd-1149-406e-a3de-2c3a5d707b69/lzh/dual/data/embed_weight.npy'
# vocab_path = '/media/omnisky/115a4abd-1149-406e-a3de-2c3a5d707b69/lzh/dual/data/word_vocab_5.pkl'
#
#
# class Vocabulary(object):
#     """Simple vocabulary wrapper."""
#
#     def __init__(self, text_style):
#         self.word2idx = {}
#         self.idx2word = {}
#         self.idx = 0
#         self.text_style = text_style
#
#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.word2idx[word] = self.idx
#             self.idx2word[self.idx] = word
#             self.idx += 1
#
#     def __call__(self, word):
#         if word not in self.word2idx and 'bow' not in self.text_style:
#             return self.word2idx['<unk>']
#         return self.word2idx[word]
#
#     def __len__(self):
#         return len(self.word2idx)
#
#
# if __name__ == '__main__':
#     all_paths = plb.Path(cap_input_path).glob('*.npy')
#     embed_weight = np.load(embed_weigth_path)
#     print("embed weight shape:{}".format(embed_weight.shape))
#
#     with open(vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
#     print("word2idx len:{}".format(len(vocab.word2idx)))
#
#     for path in tqdm(all_paths):
#         data = np.load(str(path))
#         print("data shape:{}".format(data.shape))
#         for sentence in data:
#             for item in sentence:
#
#

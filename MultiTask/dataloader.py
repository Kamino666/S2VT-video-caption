from torch.utils.data import Dataset
import torch
import numpy as np
import json
import pathlib as plb
import random
import pickle
import os

device = torch.device('cuda')


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, text_style):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.text_style = text_style

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx and 'bow' not in self.text_style:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class VideoDataset(Dataset):
    def __init__(self, enc_outputs_dir, embed_cap_dir, captions_dir, vocab_file, ID_file=None):
        """
        encode_outputs [B, ?, 2048] last batch size is 11
        embed_captions [B, ??, 500]
        captions [B, ??]
        """
        if embed_cap_dir is None or captions_dir is None:
            self.mode = 'test'
        else:
            self.mode = 'train'
        # video
        self.all_enc_path = sorted(list(plb.Path(enc_outputs_dir).glob('*.npy')), key=lambda x: int(x.name[:-4]))
        # caption
        if self.mode != 'test':
            self.all_embed_path = sorted(list(plb.Path(embed_cap_dir).glob('*.npy')), key=lambda x: int(x.name[:-4]))
            self.all_cap_path = sorted(list(plb.Path(captions_dir).glob('*.npy')), key=lambda x: int(x.name[:-4]))
        elif ID_file is None:
            self.IDs = None
        else:
            with open(ID_file) as f:
                self.IDs = eval(f.read()[2:])
        # vocab
        with open(vocab_file, 'rb') as f:
            data = pickle.load(f)
            self.word2ix = data.word2idx
            self.ix2word = data.idx2word

        sample = np.load(str(self.all_enc_path[0]))
        self.batch_size = sample.shape[0]
        # *****************DEBUG
        # self.length = len(self.all_enc_path)
        self.length = len(self.all_enc_path) - 1

    def __getitem__(self, idx):
        # print(self.all_enc_path[idx], self.all_embed_path[idx], self.all_cap_path[idx])
        enc_output = np.load(str(self.all_enc_path[idx]))
        enc_output = torch.tensor(enc_output, device=device, dtype=torch.float)

        if self.mode != 'test':
            embed = np.load(str(self.all_embed_path[idx]))
            caption = np.load(str(self.all_cap_path[idx]))  # [B, L]
            embed = torch.tensor(embed, device=device)
            caption = torch.tensor(caption, device=device)
            mask = torch.zeros(caption.shape, dtype=torch.float, device=device)
            for i, sentence in enumerate(caption):
                for j, word in enumerate(sentence):
                    if word.item() != 0:
                        mask[i, j] = 1
            return enc_output, embed, caption, mask

        item_num = enc_output.shape[0]
        if self.IDs is None:
            return enc_output
        return enc_output, self.IDs[idx * self.batch_size: idx * self.batch_size + item_num]

    def __len__(self):
        return self.length


def get_batch_data(dataset, shuffle=True):
    length = dataset.length
    for i in range(length):
        if shuffle is True:
            data_idx = random.randint(0, length-1)
        else:
            data_idx = i
        yield dataset[data_idx]


if __name__ == '__main__':
    data_dir = "../../data/MSR-VTT"
    vocab_file = os.path.join(data_dir, "word_vocab_5.pkl")

    train_vid_output_dir = os.path.join(data_dir, "train/vid_rnn_output")
    train_text_input_dir = os.path.join(data_dir, "train/text_rnn_input")
    train_word_input_dir = os.path.join(data_dir, "train/word_embed_input")
    train_set = VideoDataset(train_vid_output_dir, train_text_input_dir, train_word_input_dir, vocab_file)
    train_loader = get_batch_data(train_set)

    valid_vid_output_dir = os.path.join(data_dir, "val/vid_rnn_output")
    valid_text_input_dir = os.path.join(data_dir, "val/text_rnn_input")
    valid_word_input_dir = os.path.join(data_dir, "val/word_embed_input")
    valid_set = VideoDataset(valid_vid_output_dir, valid_text_input_dir, valid_word_input_dir, vocab_file)
    valid_loader = get_batch_data(valid_set)

    test_vid_output_dir = os.path.join(data_dir, "test/vid_rnn_output")
    test_video_id_dir = os.path.join(data_dir, "test/video_ids.txt")
    test_set = VideoDataset(test_vid_output_dir, None, None, vocab_file, ID_file=test_video_id_dir)
    test_loader = get_batch_data(test_set)

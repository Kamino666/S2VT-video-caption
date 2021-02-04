from torch.utils.data import Dataset
import torch
import numpy as np
import random
import json
import h5py


class VideoDataset(Dataset):
    def __init__(self, v2id_file, captions_file, feat_file,
                 max_len, data_split):
        """
        :param v2id_file:
        :param captions_file:
        :param feat_file:
        :param max_len: Caption的最大长度，大于这个长度的被截断
        :param data_split: list, 标志着被分配到这个数据集的video的id，在外部划分训练集/测试集
        """
        self.data_split = data_split
        self.max_len = max_len
        with open(v2id_file, encoding='utf-8') as f:
            self.v2id = json.load(f)
        with open(captions_file, encoding='utf-8') as f:
            data = json.load(f)
            self.word2ix = data['word2ix']
            self.ix2word = data['ix2word']
            self.captions = data['captions']  # [id, caption]
        f = h5py.File(feat_file)
        self.feat = f['feats']

    def __getitem__(self, ix):
        """
        根据索引，在data_split中找到对应的video id
        然后从相应的caption中随机抽取一个返回
        :param ix: index of data
        :return: tuple(img_feat, label)
        """
        feat = self.feat[self.data_split[ix]]
        labels = self.captions[str(self.data_split[ix])]
        label = random.choice(labels)
        label = label[:self.max_len-1] if self.max_len > len(label) else label
        return feat, label

    def __len__(self):
        return len(self.data_split)



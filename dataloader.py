from torch.utils.data import Dataset
import torch
import numpy as np
import random
import json
import pathlib as plb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VideoDataset(Dataset):
    def __init__(self, captions_file, feat_path):
        with open(captions_file, encoding='utf-8') as f:
            data = json.load(f)
            self.word2ix = data['word2ix']
            self.ix2word = data['ix2word']
            self.captions = data['captions']  # [name, caption]
        self.feat_paths = [i for i in plb.Path(feat_path).glob('*.npy')]

    def __getitem__(self, index):
        """
        抽取一个feat，随机一个对应的caption出来
        :param index: index of data
        :return: tuple(tensor(img_feat), tensor(label))
        """
        feat = np.load(str(self.feat_paths[index]))
        feat = torch.tensor(feat)
        labels = self.captions[self.feat_paths[index].stem]
        label = random.choice(labels)
        return feat, label

    def __len__(self):
        return len(self.feat_paths)


# TODO(Kamino): 改动collate_fn，不用这么复杂，按照feat的长度给这些排好序就行
def collate_fn2(data):
    """
    batch内根据feat的长度进行排序，并且pad成同样的大小
    :param data: list[batch_size, list[feat, label]]
    :return: tensor[batch_size, feat_max, 2048], tensor[batch_size, label_max]
    """
    # extract feat and label from batch
    batch_size = len(data)
    feat_dim = data[0][0].shape[1]
    feat_max = 0
    label_max = 0
    for i, item in enumerate(data):
        feat, label = item  # feat:tensor label:list
        feat_max = feat.shape[0] if feat_max < feat.shape[0] else feat_max
        label_max = len(label) if label_max < len(label) else label_max

    # sort and pad
    data.sort(reverse=True, key=lambda x: x[0].shape[0])
    feats_np = np.zeros([batch_size, feat_max, feat_dim], dtype=np.float)
    labels_np = np.zeros([batch_size, label_max], dtype=np.long)
    for i, (feat, label) in enumerate(data):
        feats_np[i][0:feat.shape[0]] = feat
        labels_np[i][0:len(label)] = label

    # build tensor
    feats_ts = torch.tensor(feats_np, device=device)
    labels_ts = torch.tensor(labels_np, dtype=torch.long, device=device)

    return feats_ts, labels_ts


def collate_fn(data):
    """
    batch内对feat和label分别进行排序，给两者一个序号以便对应。pad成同样的大小。
    :param data: list[batch_size, [feat, label]]
    :return: tensor[batch_size, feat_max, 2048], tensor[batch_size, label_max], list[batch_size], list[batch_size]
    """
    print(data)
    # extract feat and label from batch
    batch_size = len(data)
    feat_dim = data[0][0].shape[1]
    feat_max = 0
    label_max = 0
    feats = []
    labels = []
    for i, item in enumerate(data):
        feat, label = item  # feat:tensor label:list
        feat_max = feat.shape[0] if feat_max < feat.shape[0] else feat_max
        label_max = len(label) if label_max < len(label) else label_max
        labels.append([i, label])
        feats.append([i, feat])

    # sort and pad
    feats.sort(reverse=True, key=lambda x: x[1].shape[1])
    feat_id, feats = zip(*feats)
    feats_np = np.zeros([batch_size, feat_max, feat_dim], dtype=np.float)
    for i in range(batch_size):
        feats_np[i][0:len(feats[i])] = feats[i]

    labels.sort(reverse=True, key=lambda x: len(x[1]))
    label_id, labels = zip(*labels)
    labels_np = np.zeros([batch_size, label_max], dtype=np.long)
    for i in range(batch_size):
        labels_np[i][0:len(labels[i])] = labels[i]

    # build tensor
    feats_ts = torch.tensor(feats_np, device=device)
    labels_ts = torch.tensor(labels_np, dtype=torch.long, device=device)

    return feats_ts, labels_ts, feat_id, label_id


if __name__ == '__main__':
    trainset = VideoDataset('data/captions.json', 'data/feats')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn2)

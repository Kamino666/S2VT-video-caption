from torch.utils.data import Dataset
import torch
import numpy as np
import random
import json
import pathlib as plb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VideoDataset(Dataset):
    def __init__(self, captions_file, feat_path, mode='train'):
        with open(captions_file, encoding='utf-8') as f:
            data = json.load(f)
            self.word2ix = data['word2ix']
            self.ix2word = data['ix2word']
            self.captions = data['captions']  # [name, caption]
            self.splits = data['splits']
        # filter the train/test/valid split
        all_feat_paths = [i for i in plb.Path(feat_path).glob('*.npy')]
        self.feat_paths = []
        for path in all_feat_paths:
            if path.stem in self.splits[mode]:
                self.feat_paths.append(path)

    def __getitem__(self, index):
        """
        抽取一个feat，随机一个对应的caption出来
        :param index: index of data
        :return: tuple(tensor(img_feat), tensor(label), str(ID))
        """
        ID = self.feat_paths[index].stem
        feat = np.load(str(self.feat_paths[index]))
        feat = torch.tensor(feat)
        labels = self.captions[ID]
        label = random.choice(labels)
        return feat, label, ID

    def __len__(self):
        return len(self.feat_paths)


def collate_fn(data):
    """
    batch内根据feat的长度进行排序，并且pad成同样的大小
    :param data: list[batch_size, list[feat, label, ID]]
    :return: tensor[batch_size, feat_max, 2048], tensor[batch_size, label_max], list[batch_size]
    """
    # extract feat and label from batch
    batch_size = len(data)
    feat_dim = data[0][0].shape[1]
    feat_max = 0
    label_max = 0
    for i, item in enumerate(data):
        feat, label, ID = item  # feat:tensor label:list ID:str
        feat_max = feat.shape[0] if feat_max < feat.shape[0] else feat_max
        label_max = len(label) if label_max < len(label) else label_max

    # sort and pad
    data.sort(reverse=True, key=lambda x: x[0].shape[0])
    feats_np = np.zeros([batch_size, feat_max, feat_dim], dtype=np.float)
    labels_np = np.zeros([batch_size, label_max], dtype=np.long)
    IDs = []
    for i, (feat, label, ID) in enumerate(data):
        feats_np[i][0:feat.shape[0]] = feat
        labels_np[i][0:len(label)] = label
        IDs.append(ID)

    # build tensor
    feats_ts = torch.tensor(feats_np, dtype=torch.float, device=device)
    labels_ts = torch.tensor(labels_np, dtype=torch.long, device=device)

    return feats_ts, labels_ts, IDs


if __name__ == '__main__':
    trainset = VideoDataset('data/captions.json', 'data/feats')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn)

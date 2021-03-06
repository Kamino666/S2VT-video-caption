from torch.utils.data import Dataset
import torch
import numpy as np
import random
import json
import pathlib as plb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VideoDataset(Dataset):
    def __init__(self, captions_file, feat_path, max_len=80, mode='train'):
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
        self.max_len = max_len
        print("prepare {} dataset. vocab_size: {}, dataset_size: {}".format(mode, len(self.word2ix), len(self.feat_paths)))

    def __getitem__(self, index):
        """
        select a feature and randomly select a corresponding caption,
        then pad the caption to max_len when mode is 'train' or 'valid'
        :param index: index of data
        :return: tuple(tensor(img_feat), tensor(label), str(ID))
        """
        ID = self.feat_paths[index].stem

        feat = np.load(str(self.feat_paths[index]))
        feat = torch.tensor(feat, dtype=torch.float, device=device, requires_grad=True)

        labels = self.captions[ID]
        label = np.random.choice(labels, 1)[0]
        if len(label) > self.max_len:
            label = label[:self.max_len]
        pad_label = torch.zeros([self.max_len], dtype=torch.long, device=device)
        pad_label[:len(label)] = torch.tensor(label, dtype=torch.long, device=device)
        mask = torch.zeros([self.max_len], dtype=torch.float, device=device)
        mask[:len(label)] = 1

        return feat, pad_label, ID, mask

    def __len__(self):
        return len(self.feat_paths)


if __name__ == '__main__':
    trainset = VideoDataset('data/captions.json', '../S2VT-master/Data/Features_VGG')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    a = next(iter(train_loader))

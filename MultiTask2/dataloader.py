from torch.utils.data import Dataset
import torch
import numpy as np
import json
import pathlib as plb
import random
import os
import array

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BigFile:

    def __init__(self, datadir):
        self.nr_of_images, self.ndims = map(int, open(os.path.join(datadir, 'shape.txt')).readline().split())
        id_file = os.path.join(datadir, "id.txt")
        self.names = open(id_file).read().strip().split()
        assert (len(self.names) == self.nr_of_images)
        self.name2index = dict(zip(self.names, range(self.nr_of_images)))
        self.binary_file = os.path.join(datadir, "feature.bin")
        print("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir))

    def read(self, requested, isname=True):
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert (min(requested) >= 0)
            assert (max(requested) < len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []

        index_name_array.sort(key=lambda v: v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims

        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]

        for next in sorted_index[1:]:
            move = (next - 1 - previous) * offset
            # print next, move
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next

        fr.close()

        return [x[1] for x in index_name_array], [res[i * self.ndims:(i + 1) * self.ndims].tolist() for i in
                                                  range(nr_of_images)]

    def read_one(self, name):
        renamed, vectors = self.read([name])
        return vectors[0]

    def shape(self):
        return [self.nr_of_images, self.ndims]


class VideoDataset(Dataset):
    def __init__(self, captions_file, feat_path, max_len=80, mode='train'):
        with open(captions_file, encoding='utf-8') as f:
            data = json.load(f)
            self.word2ix = data['word2ix']
            self.ix2word = data['ix2word']
            self.captions = data['captions']  # [name, caption]
            self.splits = data['splits']

        # load features
        self.feature_dict = {}
        self.IDs = []
        visual_feat = BigFile(os.path.join(feat_path))
        with open(os.path.join(feat_path, "video2frames.txt")) as f:
            v2f = eval(f.read())  # dict{video_id: frame_list[frame_id]}
            for video_id, frame_list in v2f.items():
                frame_vecs = []
                for frame_id in frame_list:
                    frame_vecs.append(visual_feat.read_one(frame_id))
                frames_tensor = torch.tensor(frame_vecs)
                if frames_tensor.shape[0] > 200:
                    print(video_id)
                    print(frames_tensor.shape[0])
                    continue
                # filter the train/test/valid split
                if video_id in self.splits[mode]:
                    self.IDs.append(video_id)
                    self.feature_dict[video_id] = frames_tensor
                # print(frames_tensor.shape, video_id)

        self.max_len = max_len
        print("prepare {} dataset. vocab_size: {}, dataset_size: {}"
              .format(mode, len(self.word2ix), len(self.IDs)))

    def __getitem__(self, index):
        """
        select a feature and randomly select a corresponding caption,
        then pad the caption to max_len when mode is 'train' or 'valid'
        :param index: index of data
        :return: tuple(tensor(img_feat), tensor(label), str(ID))
        """
        ID = self.IDs[index]
        feat = self.feature_dict[ID].to(device)

        labels = self.captions[ID]
        label = np.random.choice(labels, 1)[0]  # do not use Python random.choice
        # label = random.choice(labels)
        if len(label) > self.max_len:
            label = label[:self.max_len]
        pad_label = torch.zeros([self.max_len], dtype=torch.long, device=device)
        pad_label[:len(label)] = torch.tensor(label, dtype=torch.long, device=device)
        mask = torch.zeros([self.max_len], dtype=torch.float, device=device)
        mask[:len(label)] = 1

        return feat, pad_label, ID, mask

    def __len__(self):
        return len(self.IDs)


def collate_fn(data):
    """
    data is a list which has a length of BATCH_SIZE
    return feat, feat_lengths, pad_label, ID, mask
    """
    batch_size = len(data)
    # sort the data
    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    feats, pad_labels, IDs, masks = zip(*data)
    # pad the feats
    padded_feats = torch.zeros([batch_size, feats[0].shape[0], feats[0].shape[1]], dtype=torch.float, device=device)
    lengths = []
    for i, feat in enumerate(feats):
        padded_feats[i, :feat.shape[0], :] = feat
        lengths.append(feat.shape[0])
    # to tensor
    pad_labels = [i.unsqueeze(dim=0) for i in pad_labels]
    pad_labels = torch.cat(pad_labels, dim=0)
    masks = [i.unsqueeze(dim=0) for i in masks]
    masks = torch.cat(masks, dim=0)
    # if lengths[0] > 100:
    #     print(lengths, IDs)
    return padded_feats, lengths, pad_labels, IDs, masks


if __name__ == '__main__':
    # for debug
    trainset = VideoDataset('multi_task_data/captions_msvd_min5.json',
                            '/media/omnisky/115a4abd-1149-406e-a3de-2c3a5d707b69/lzh/dual/data/MSVD/resnet152_feature')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    a = next(iter(train_loader))

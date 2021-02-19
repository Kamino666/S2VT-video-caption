import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import json
import random
from tqdm import tqdm

from dataloader import VideoDataset, collate_fn2
from S2VTModel import S2VTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Opt:
    train_percentage = 0.9
    batch_size = 128
    max_len = 30
    dim_hidden = 128
    dim_word = 256
    lr = 0.001
    EPOCHS = 100


def get_pad_lengths(data, pad_id=0):
    # print(data.shape)
    lengths = []
    for seq in data:
        count = 0
        for item in seq:
            if (item == torch.zeros(item.shape, dtype=torch.double, device=item.device)).all():
                break
            else:
                count += 1
        lengths.append(count)
    return lengths


def train():
    opt = Opt()
    # prepare data
    trainset = VideoDataset('data/captions.json', 'data/feats')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn2)
    word2ix = trainset.word2ix
    ix2word = trainset.ix2word
    vocab_size = len(word2ix)

    # build model
    model = S2VTModel(
        vocab_size,
        opt.max_len,
        opt.dim_hidden,
        opt.dim_word,
        sos_id=1, eos_id=2,
        rnn_cell='lstm',
        rnn_dropout_p=0
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr
    )
    criterion = nn.NLLLoss()

    ###
    ### start training
    ###

    for epoch in range(opt.EPOCHS):
        running_loss = 0.0
        loss_count = 0
        for index, (feats, targets) in enumerate(
                tqdm(train_loader, desc="training")):
            optimizer.zero_grad()
            model.train()

            feat_lengths = get_pad_lengths(feats)
            probs, preds = model(feats, feat_lengths, targets)
            # probs: [batch_size, label_max, vocab_size] targets: [batch_size, label_max]
            # print(probs.shape, targets.shape)
            loss = criterion(probs.contiguous().view(probs.shape[0]*probs.shape[1], -1)
                             , targets[:, :-1].contiguous().view(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_count += 1
        running_loss /= loss_count
        print(running_loss)


if __name__ == '__main__':
    train()

# TODO(Kamino): 训练部分
# TODO(Kamino): Optical Flow的部分
# TODO(Kamino): METEOR评估指标
# TODO(Kamino): 可视化

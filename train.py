import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import json
import random
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloader import VideoDataset, collate_fn
from S2VTModel import S2VTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


class Opt:
    batch_size = 16
    max_len = 30
    dim_hidden = 1000
    dim_word = 500
    lr = 0.0005
    EPOCHS = 200
    save_freq = 10
    save_path = './checkpoint'
    start_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime())
    feat_dim = 4096


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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    testset = VideoDataset('data/captions.json', 'data/feats', mode='test')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    word2ix = trainset.word2ix
    ix2word = trainset.ix2word
    vocab_size = len(word2ix)

    # build model
    model = S2VTModel(
        vocab_size,
        opt.feat_dim,
        rnn_dropout_p=0
    ).to(device)
    # torch.save(model, 'test.pth')  # check
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr
    )
    criterion = nn.NLLLoss()

    ###
    ### start training
    ###

    for epoch in range(opt.EPOCHS):
        # train
        train_running_loss = 0.0
        loss_count = 0
        for index, (feats, targets, IDs) in enumerate(
                tqdm(train_loader, desc="epoch:{}".format(epoch))):
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

            train_running_loss += loss.item()
            loss_count += 1
        train_running_loss /= loss_count
        writer.add_scalar('train_loss', train_running_loss, global_step=epoch)

        # test
        test_running_loss = 0.0
        loss_count = 0
        for index, (feats, targets, IDs) in enumerate(test_loader):
            optimizer.zero_grad()
            model.eval()

            feat_lengths = get_pad_lengths(feats)
            with torch.no_grad():
                probs, preds = model(feats, feat_lengths, targets)

            loss = criterion(probs.contiguous().view(probs.shape[0] * probs.shape[1], -1)
                             , targets[:, :-1].contiguous().view(-1))
            test_running_loss += loss.item()
            loss_count += 1
        test_running_loss /= loss_count
        writer.add_scalar('test_loss', test_running_loss, global_step=epoch)
        print("train loss:{} test loss: {}".format(train_running_loss, test_running_loss))

        # save checkpoint
        # if epoch % opt.save_freq == 0 and epoch != 0:
        if epoch % opt.save_freq == 0:
            print('epoch:{}, saving checkpoint'.format(epoch))
            torch.save(model, os.path.join(opt.save_path,
                                           opt.start_time + str(epoch) + '.pth'))
    # save model
    torch.save(model, os.path.join(opt.save_path, opt.start_time + 'final.pth'))


if __name__ == '__main__':
    train()

# TODO(Kamino): METEOR评估指标
# TODO(Kamino): Optical Flow的部分
# TODO(Kamino): beam_search



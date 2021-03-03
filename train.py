import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import numpy as np
import json
import random
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloader import VideoDataset, collate_fn
from S2VTModel import S2VTModel
from utils import LengthCriterion, EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


class Opt:
    # data config
    caption_file = r"./data/captions_MSR_VTT.json"
    feats_path = r"./data/feats/resnet152"
    # model config
    max_len = 30
    dim_hidden = 1000
    dim_word = 500
    feat_dim = 2048
    rnn_dropout = 0.5
    num_layers = 1
    bidirectional = False
    # data config
    batch_size = 32
    # train config
    EPOCHS = 200
    save_freq = 30
    save_path = './checkpoint'
    start_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime())
    early_stopping_patience = 10
    # optimizer config
    lr = 0.0004
    learning_rate_decay_every = 50
    learning_rate_decay_rate = 0.8
    weight_decay = 5e-4  # Regularzation


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
    trainset = VideoDataset(opt.caption_file, opt.feats_path)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    testset = VideoDataset(opt.caption_file, opt.feats_path, mode='valid')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    word2ix = trainset.word2ix
    ix2word = trainset.ix2word
    vocab_size = len(word2ix)

    # build model
    model = S2VTModel(
        vocab_size,
        opt.feat_dim,
        rnn_dropout_p=opt.rnn_dropout,
        bidirectional=opt.bidirectional
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt.learning_rate_decay_every,
        gamma=opt.learning_rate_decay_rate
    )
    early_stopping = EarlyStopping(patience=opt.early_stopping_patience,
                                   verbose=True,
                                   path=os.path.join(opt.save_path, opt.start_time + 'stop.pth'))
    criterion = LengthCriterion()

    ###
    ### start training
    ###

    try:
        for epoch in range(opt.EPOCHS):
            # train
            train_running_loss = 0.0
            loss_count = 0
            for index, (feats, targets, IDs) in enumerate(
                    tqdm(train_loader, desc="epoch:{}".format(epoch))):
                optimizer.zero_grad()
                model.train()

                feat_lengths = get_pad_lengths(feats)
                probs, preds = model(feats, feat_lengths, targets, teacher_forcing_rate=0)
                # probs: [batch_size, label_max, vocab_size] targets: [batch_size, label_max]
                # loss = criterion(probs.contiguous().view(probs.shape[0]*probs.shape[1], -1)
                #                  , targets[:, :-1].contiguous().view(-1))
                loss = criterion(probs, targets, feat_lengths)

                loss.backward()
                optimizer.step()

                train_running_loss += loss.item()
                loss_count += 1

            train_running_loss /= loss_count
            writer.add_scalar('train_loss', train_running_loss, global_step=epoch)

            # validate
            valid_running_loss = 0.0
            loss_count = 0
            for index, (feats, targets, IDs) in enumerate(test_loader):
                optimizer.zero_grad()
                model.eval()

                feat_lengths = get_pad_lengths(feats)
                with torch.no_grad():
                    probs, preds = model(feats, feat_lengths, targets)

                # loss = criterion(probs.contiguous().view(probs.shape[0] * probs.shape[1], -1)
                #                  , targets[:, :-1].contiguous().view(-1))
                loss = criterion(probs, targets, feat_lengths)
                valid_running_loss += loss.item()
                loss_count += 1

            valid_running_loss /= loss_count
            writer.add_scalar('valid_loss', valid_running_loss, global_step=epoch)
            print("train loss:{} valid loss: {}".format(train_running_loss, valid_running_loss))

            # early stop
            early_stopping(valid_running_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # save checkpoint
            if epoch % opt.save_freq == 0:
                print('epoch:{}, saving checkpoint'.format(epoch))
                torch.save(model, os.path.join(opt.save_path,
                                               opt.start_time + str(epoch) + '.pth'))
    except KeyboardInterrupt as e:
        print("Training interruption, save tensorboard log...")
        writer.close()
    # save model
    torch.save(model, os.path.join(opt.save_path, opt.start_time + 'final.pth'))


if __name__ == '__main__':
    train()

# TODO(Kamino): beam_search



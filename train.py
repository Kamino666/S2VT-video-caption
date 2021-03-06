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

from dataloader import VideoDataset
from S2VTModel import S2VT
from utils import MaskCriterion, EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


class Opt:
    # data config
    caption_file = r"./data/captions.json"
    feats_path = r"G:\workspace\pytorch\S2VT-master\Data\Features_VGG"
    # model config
    train_length = 80
    dim_hidden = 500
    dim_embed = 500
    feat_dim = 4096
    feat_dropout = 0.5
    out_dropout = 0.5
    rnn_dropout = 0.5
    num_layers = 1
    bidirectional = False
    rnn_type = 'lstm'
    # data config
    batch_size = 32
    # train config
    EPOCHS = 300
    save_freq = 30
    save_path = './checkpoint'
    start_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime())
    early_stopping_patience = 30
    # optimizer config
    lr = 0.0001
    learning_rate_decay_every = 25
    learning_rate_decay_rate = 0.1
    # weight_decay = 5e-4  # Regularzation
    weight_decay = 5e-6  # Regularzation


def train():
    opt = Opt()
    # prepare data
    trainset = VideoDataset(opt.caption_file, opt.feats_path)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
    testset = VideoDataset(opt.caption_file, opt.feats_path, mode='valid')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False)
    word2ix = trainset.word2ix
    ix2word = trainset.ix2word
    vocab_size = len(word2ix)

    # build model
    model = S2VT(
        vocab_size,
        opt.feat_dim,
        dim_hid=opt.dim_hidden,
        dim_embed=opt.dim_embed,
        length=opt.train_length,
        feat_dropout=opt.feat_dropout,
        rnn_dropout=opt.rnn_dropout,
        out_dropout=opt.out_dropout,
        num_layers=opt.num_layers,
        bidirectional=opt.bidirectional,
        rnn_type=opt.rnn_type,
        sos_ix=word2ix['<sos>']
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr,
        # weight_decay=opt.weight_decay
    )
    # lr_scheduler = optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=opt.learning_rate_decay_every,
    #     gamma=opt.learning_rate_decay_rate
    # )
    early_stopping = EarlyStopping(patience=opt.early_stopping_patience,
                                   verbose=True,
                                   path=os.path.join(opt.save_path, opt.start_time + 'stop.pth'))
    criterion = MaskCriterion()

    ###
    ### start training
    ###

    try:
        for epoch in range(opt.EPOCHS):
            # train
            train_running_loss = 0.0
            loss_count = 0
            for index, (feats, targets, IDs, masks) in enumerate(
                    tqdm(train_loader, desc="epoch:{}".format(epoch))):
                optimizer.zero_grad()
                model.train()

                # probs [B, L, vocab_size]
                probs = model(feats, targets=targets[:, :-1], mode='train')

                loss = criterion(probs, targets, masks)

                loss.backward()
                optimizer.step()
                # lr_scheduler.step()

                train_running_loss += loss.item()
                loss_count += 1

            train_running_loss /= loss_count
            writer.add_scalar('train_loss', train_running_loss, global_step=epoch)

            # validate
            valid_running_loss = 0.0
            loss_count = 0
            for index, (feats, targets, IDs, masks) in enumerate(test_loader):
                model.eval()

                with torch.no_grad():
                    probs = model(feats, targets=targets[:, :-1], mode='train')
                    loss = criterion(probs, targets, masks)

                valid_running_loss += loss.item()
                loss_count += 1

            valid_running_loss /= loss_count
            writer.add_scalar('valid_loss', valid_running_loss, global_step=epoch)
            if epoch % 10 == 0:
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(name, param, epoch)

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
        print(e)
        print("Training interruption, save tensorboard log...")
        writer.close()
    # save model
    torch.save(model, os.path.join(opt.save_path, opt.start_time + 'final.pth'))


if __name__ == '__main__':
    train()

# TODO(Kamino): beam_search

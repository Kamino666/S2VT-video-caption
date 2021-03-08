import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

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
    """config class"""
    # - data config
    caption_file = r"./data/captions.json"  # the file generated in prepare_captions.py
    feats_path = r"./data/feats/vgg16_bn"  # the features extracted by extract_features.py
    # - model config
    train_length = 80  # fix length during training, the feats length must be equal to this
    dim_hidden = 500
    dim_embed = 500
    feat_dim = 4096
    feat_dropout = 0.5
    out_dropout = 0.5
    rnn_dropout = 0.5
    num_layers = 1
    bidirectional = False  # do not use True yet
    rnn_type = 'lstm'  # do not change to GRU yet
    # - data config
    batch_size = 32
    # - train config
    EPOCHS = 300
    save_freq = 100  # every n epoch, save once
    save_path = './checkpoint'
    histogram_freq = 10
    start_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime())
    early_stopping_patience = 30
    # - optimizer config
    lr = 0.0001
    learning_rate_patience = 20
    # weight_decay = 5e-5  # Regularzation


def save_opt(opt):
    with open(os.path.join(opt.save_path, opt.start_time + 'opt.txt'), 'w+', encoding='utf-8') as f:
        f.write(str(vars(Opt)))


def train():
    opt = Opt()
    # write log
    save_opt(opt)

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
        sos_ix=word2ix['<sos>'],
        eos_ix=word2ix['<eos>'],
    ).to(device)
    # model.load_glove_weights('./data/glove.6B.300d.txt', 300, trainset.ix2word)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr,
        # weight_decay=opt.weight_decay
    )
    # dynamic learning rate
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=opt.learning_rate_patience
    )
    early_stopping = EarlyStopping(patience=opt.early_stopping_patience,
                                   verbose=True,
                                   path=os.path.join(opt.save_path, opt.start_time + 'stop.pth'))
    criterion = MaskCriterion()

    ###
    ### start training
    ###

    try:
        for epoch in range(opt.EPOCHS):
            # ****************************
            #            train
            # ****************************
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

                train_running_loss += loss.item()
                loss_count += 1

            train_running_loss /= loss_count
            writer.add_scalar('train_loss', train_running_loss, global_step=epoch)

            # ****************************
            #           validate
            # ****************************
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
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            if epoch % opt.histogram_freq == 0:
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(name, param, epoch)

            print("train loss:{} valid loss: {}".format(train_running_loss, valid_running_loss))
            lr_scheduler.step(valid_running_loss)

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

# TODO(Kamino): 更改注释成英文
# TODO(Kamino): 写Readme
# TODO(Kamino): 走通MSR-VTT

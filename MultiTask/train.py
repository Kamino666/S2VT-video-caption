import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from MultiTask.dataloader import VideoDataset, Vocabulary, get_batch_data
from MultiTask.attention_baseline import Att_NoEncoder
from utils import MaskCriterion, EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


class Opt:
    """config class"""
    # - data config
    # caption_file = r"./data/captions.json"  # the file generated in prepare_captions.py
    # feats_path = r"./data/feats/vgg16_bn"  # the features extracted by extract_features.py
    # - model config
    train_length = 80  # fix length during training, the feats length must be equal to this
    dim_hidden = 1024
    dim_embed = 500
    feat_dim = 4096
    feat_dropout = 0.5
    out_dropout = 0.5
    rnn_dropout = 0.5
    num_layers = 1
    bidirectional = False  # do not use True yet
    rnn_type = 'lstm'  # do not change to GRU yet
    # - data config
    batch_size = 16
    embed_weight_path = "../../data/MSR-VTT/embed_weight.npy"
    # - train config
    EPOCHS = 300
    save_freq = 100  # every n epoch, save once
    save_path = '../checkpoint'
    histogram_freq = 10
    start_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime())
    early_stopping_patience = 30
    # - optimizer config
    lr = 0.001
    learning_rate_patience = 20
    # weight_decay = 5e-5  # Regularzation


def save_opt(opt):
    with open(os.path.join(opt.save_path, opt.start_time + 'opt.txt'), 'w+', encoding='utf-8') as f:
        f.write(str(vars(Opt)))


def train(model_checkpoint=None):
    opt = Opt()
    # write log
    save_opt(opt)

    # prepare data
    # data_dir = "/media/omnisky/115a4abd-1149-406e-a3de-2c3a5d707b69/lzh/dual/data/MSR-VTT"
    data_dir = "/media/omnisky/115a4abd-1149-406e-a3de-2c3a5d707b69/lzh/dual/data/MSVD"
    # vocab_file = os.path.join(data_dir, "word_vocab_5.pkl")
    vocab_file = os.path.join(data_dir, "train", "word_vocab_5.pkl")

    train_vid_output_dir = os.path.join(data_dir, "train/vid_rnn_output")
    train_text_input_dir = os.path.join(data_dir, "train/text_rnn_input")
    train_word_input_dir = os.path.join(data_dir, "train/word_embed_input")
    train_set = VideoDataset(train_vid_output_dir, train_text_input_dir, train_word_input_dir, vocab_file)
    train_loader = get_batch_data(train_set)

    valid_vid_output_dir = os.path.join(data_dir, "val/vid_rnn_output")
    valid_text_input_dir = os.path.join(data_dir, "val/text_rnn_input")
    valid_word_input_dir = os.path.join(data_dir, "val/word_embed_input")
    valid_set = VideoDataset(valid_vid_output_dir, valid_text_input_dir, valid_word_input_dir, vocab_file)
    valid_loader = get_batch_data(valid_set, shuffle=False)
    word2ix = train_set.word2ix
    ix2word = train_set.ix2word
    vocab_size = len(word2ix)

    # build model
    if model_checkpoint is None:
        model = Att_NoEncoder(vocab_size, opt.feat_dim, dim_hid=opt.dim_hidden, dim_embed=opt.dim_embed,
                              feat_dropout=opt.feat_dropout, out_dropout=opt.out_dropout,
                              sos_ix=word2ix['<start>'], eos_ix=word2ix['<end>']).to(device)
        model.load_embedding_weight(opt.embed_weight_path)
    else:
        model = torch.load(model_checkpoint)
    optimizer = optim.Adam(
        # model.parameters(),
        filter(lambda p: p.requires_grad, model.parameters()),
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
            for enc_output, embed, caption, mask in \
                    tqdm(train_loader, desc="epoch:{}".format(epoch)):
                # enc_output, embed, caption, mask
                optimizer.zero_grad()
                model.train()

                # probs [B, L, vocab_size]
                probs = model(enc_output, targets=embed[:, :-1, :], mode='train')

                loss = criterion(probs, caption, mask)

                loss.backward()
                optimizer.step()

                train_running_loss += loss.item()
                loss_count += 1

            train_running_loss /= loss_count
            writer.add_scalar('train_loss', train_running_loss, global_step=epoch)

            # ****************************
            #            valid
            # ****************************
            valid_running_loss = 0.0
            loss_count = 0
            for enc_output, embed, caption, mask in valid_loader:
                # print("*****************8")ype)
                # print(enc_output.dtype, embed.dtype, caption.dtype, mask.dt
                # enc_output, embed, caption, mask
                model.eval()

                # probs [B, L, vocab_size]
                with torch.no_grad():
                    probs = model(enc_output, targets=embed[:, :-1, :], mode='train')
                    loss = criterion(probs, caption, mask)

                valid_running_loss += loss.item()
                loss_count += 1

            valid_running_loss /= loss_count
            writer.add_scalar('valid_loss', valid_running_loss, global_step=epoch)

            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            if epoch % opt.histogram_freq == 0:
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(name, param, epoch)

            print("train loss:{}, valid loss:{}".format(train_running_loss, valid_running_loss))
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

            # (tmp) reload dataloader
            train_loader = get_batch_data(train_set)
            valid_loader = get_batch_data(valid_set, shuffle=False)

    except KeyboardInterrupt as e:
        print(e)
        print("Training interruption, save tensorboard log...")
        writer.close()
    # save model
    torch.save(model, os.path.join(opt.save_path, opt.start_time + 'final.pth'))


if __name__ == '__main__':
    train()

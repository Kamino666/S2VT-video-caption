import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import json
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchnlp.metrics import get_moses_multi_bleu

from dataloader import VideoDataset, collate_fn
from S2VTModel import S2VTModel
from train import get_pad_lengths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Opt:
    model_path = r"./checkpoint/21_02_19_13_13_31-final.pth"
    batch_size = 8

def eval():
    opt = Opt()

    # prepare data
    validset = VideoDataset('data/captions.json', 'data/feats')
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    word2ix = validset.word2ix
    ix2word = validset.ix2word
    vocab_size = len(word2ix)

    # load model
    model = torch.load(opt.model_path).to(device)

    ###
    ### start validation
    ###

    criterion = nn.NLLLoss()
    prediction_dict = {}
    for index, (feats, targets, IDs) in enumerate(tqdm(valid_loader, desc="validation")):
        # get prediction and cal loss
        model.train(mode=False)
        feat_lengths = get_pad_lengths(feats)
        with torch.no_grad():
            probs, preds = model(feats, feat_lengths, targets, mode='train')
        # assert 1 == 0
        # save result
        for ID, pred in zip(IDs, preds):
            # truncate padding tail
            for i in range(len(pred)):
                if pred[i] == 0:
                    pred = pred[:i]
                    break
            # to words
            word_preds = [ix2word[str(i.item())] for i in pred]
            prediction_dict[ID] = ' '.join(word_preds)

    print(prediction_dict)


if __name__ == '__main__':
    eval()


import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm

from dataloader import VideoDataset

from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Opt:
    model_path = r"./checkpoint/21_03_18_18_42_32-stop.pth"
    csv_file = r"./data/video_corpus.csv"
    train_source_file = r"./data/annotation2016/train_val_videodatainfo.json"
    caption_file = r"./data/captions_server.json"
    feats_path = r"./data/feats/vgg16_bn"
    batch_size = 10


def eval():
    opt = Opt()

    # prepare data
    validset = VideoDataset(opt.caption_file, opt.feats_path, mode='test')
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=False)
    word2ix = validset.word2ix
    ix2word = validset.ix2word
    vocab_size = len(word2ix)

    # load model
    model = torch.load(opt.model_path).to(device)

    ###
    ### start test
    ###

    pred_dict = {}
    for index, (feats, targets, IDs, masks) in enumerate(tqdm(valid_loader, desc="test")):
        # get prediction and cal loss
        model.eval()
        with torch.no_grad():
            preds = model(feats, mode='test')  # preds [B, L]
        # save result
        for ID, pred in zip(IDs, preds):
            word_preds = [ix2word[str(i.item())] for i in pred]
            if '<eos>' in word_preds:
                word_preds = word_preds[:word_preds.index('<eos>')]
            pred_dict[ID] = ' '.join(word_preds)

    return pred_dict


def beam_eval():
    opt = Opt()

    # prepare data
    validset = VideoDataset(opt.caption_file, opt.feats_path, mode='test')
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=False)
    word2ix = validset.word2ix
    ix2word = validset.ix2word
    vocab_size = len(word2ix)

    # load model
    model = torch.load(opt.model_path).to(device)

    ###
    ### start test
    ###

    pred_dict = {}
    for index, (feats, targets, IDs, masks) in enumerate(tqdm(valid_loader, desc="test")):
        # get prediction and cal loss
        model.eval()
        model.rnn_type = 'lstm'
        model.sos_ix = 3
        model.eos_ix = 4
        with torch.no_grad():
            preds = model(feats, mode='beam_search')  # preds [B, L]
        # save result
        for ID, pred in zip(IDs, preds):
            word_preds = [ix2word[str(i.item())] for i in pred]
            if '<eos>' in word_preds:
                word_preds = word_preds[:word_preds.index('<eos>')]
            if '<sos>' in word_preds:
                word_preds.remove('<sos>')
            pred_dict[ID] = ' '.join(word_preds)
        print(pred_dict)

    return pred_dict


# abandon
def mst_vrr_to_coco_gts(train_source_file):
    # read data
    with open(train_source_file, encoding='utf-8') as f:
        data = json.load(f)
        sentences = data["sentences"]
        videos = data["videos"]
    gts = {}
    max_cap_ids = {}
    """遍历所有的英文描述，找到feature包含的video对应的caption，存储。"""
    for image_id, sentence in zip(videos, sentences):
        image_id = image_id["video_id"]
        # process caption
        tokenized = sentence["caption"].lower()
        tokenized = re.sub(r'[.!,;?:]', ' ', tokenized)
        # gts
        if image_id in gts:
            max_cap_ids[image_id] += 1
            gts[image_id].append({
                u'image_id': image_id,
                u'cap_id': max_cap_ids[image_id],
                u'caption': sentence["caption"],
                u'tokenized': tokenized
            })
        else:
            max_cap_ids[image_id] = 0
            gts[image_id] = [{
                u'image_id': image_id,
                u'cap_id': 0,
                u'caption': sentence["caption"],
                u'tokenized': tokenized
            }]

    return gts


def pred_to_coco_samples_IDs(prediction_dict, gts=None):
    # samples = {
    #   '184321': [{u'image_id': '184321', u'caption': u'train traveling down a track in front of a road'}],
    #   '81922': [{u'image_id': '81922', u'caption': u'plane is flying through the sky'}],
    # }
    samples = {}
    IDs = []
    for item in prediction_dict.items():
        if gts is not None and item[0] in gts:
            IDs.append(item[0])
            samples[item[0]] = [{
                u'image_id': item[0],
                u'caption': item[1]
            }]
    return samples, IDs


class COCOScorer(object):
    """
    codes from https://github.com/tylin/coco-caption
    Microsoft COCO Caption Evaluation
    """
    def __init__(self):
        print('init COCO-EVAL scorer')

    def score(self, GT, RES, IDs):
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
            #            print ID
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        # get token
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print("%s: %0.3f" % (method, score))

        # for metric, score in self.eval.items():
        #    print '%s: %.3f'%(metric, score)
        return self.eval

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score


if __name__ == '__main__':
    # prediction_dict = beam_eval()  # beam search. needs a LOT of time.
    prediction_dict = eval()
    with open('./data/gts.json', encoding='utf-8') as f:
        gts = json.load(f)['gts']
    samples, IDs = pred_to_coco_samples_IDs(prediction_dict, gts)

    scorer = COCOScorer()
    scorer.score(gts, samples, IDs)

    print("***********************")
    print(scorer.eval)
    print("***********************")
    print(scorer.imgToEval)

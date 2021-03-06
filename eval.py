import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json
import re
import random
from tqdm import tqdm
import sys

from dataloader import VideoDataset

from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Opt:
    model_path = r"./checkpoint/21_03_06_17_16_43-40.pth"
    csv_file = r"./data/video_corpus.csv"
    train_source_file = r"./data/annotation2016/train_val_videodatainfo.json"
    caption_file = r"./data/captions.json"
    feats_path = r"G:\workspace\pytorch\S2VT-master\Data\Features_VGG"
    batch_size = 10


def eval():
    opt = Opt()

    # prepare data
    validset = VideoDataset(opt.caption_file, opt.feats_path, mode='valid')
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=False)
    word2ix = validset.word2ix
    ix2word = validset.ix2word
    vocab_size = len(word2ix)

    # load model
    model = torch.load(opt.model_path).to(device)

    ###
    ### start test
    ###

    prediction_dict = {}
    for index, (feats, targets, IDs, masks) in enumerate(tqdm(valid_loader, desc="test")):
        # get prediction and cal loss
        model.eval()
        with torch.no_grad():
            preds = model(feats, mode='test')  # preds [B, L]
        # save result
        for ID, pred in zip(IDs, preds):
            word_preds = [ix2word[str(i.item())] for i in pred]
            # if word_preds[0] == '<sos>':
            #     del word_preds[0]
            prediction_dict[ID] = ' '.join(word_preds)

    return prediction_dict


def csv_to_coco_gts(csv_file, clean_only=False):
    # gts = {
    #     '184321': [
    #         {u'image_id': '184321', u'cap_id': 0, u'caption': u'A train traveling down tracks next to lights.',
    #          'tokenized': 'a train traveling down tracks next to lights'},
    #         {u'image_id': '184321', u'cap_id': 1, u'caption': u'A train coming down the tracks arriving at a station.',
    #          'tokenized': 'a train coming down the tracks arriving at a station'}],
    #     '81922': [
    #         {u'image_id': '81922', u'cap_id': 0, u'caption': u'A large jetliner flying over a traffic filled street.',
    #          'tokenized': 'a large jetliner flying over a traffic filled street'},
    #         {u'image_id': '81922', u'cap_id': 1, u'caption': u'The plane is flying over top of the cars',
    #          'tokenized': 'the plan is flying over top of the cars'}, ]
    # }
    # read csv
    file = pd.read_csv(csv_file, encoding='utf-8')
    data = pd.DataFrame(file)
    data = data.dropna(axis=0)
    eng_data = data[data['Language'] == 'English']
    if clean_only is True:
        eng_data = eng_data[eng_data['Source'] == 'clean']

    gts = {}
    max_cap_ids = {}
    """遍历所有的英文描述，找到feature包含的video对应的caption，存储。"""
    for _, name, start, end, sentence in eng_data[['VideoID', 'Start', 'End', 'Description']].itertuples():
        # get id
        image_id = name + '_' + str(start) + '_' + str(end)  # + '.avi'
        # process caption
        tokenized = sentence.lower()
        tokenized = re.sub(r'[.!,;?:]', ' ', tokenized)
        # gts
        if image_id in gts:
            max_cap_ids[image_id] += 1
            gts[image_id].append({
                u'image_id': image_id,
                u'cap_id': max_cap_ids[image_id],
                u'caption': sentence,
                u'tokenized': tokenized
            })
        else:
            max_cap_ids[image_id] = 0
            gts[image_id] = [{
                u'image_id': image_id,
                u'cap_id': 0,
                u'caption': sentence,
                u'tokenized': tokenized
            }]
    return gts


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


def pred_to_coco_samples_IDs(prediction_dict):
    # samples = {
    #   '184321': [{u'image_id': '184321', u'caption': u'train traveling down a track in front of a road'}],
    #   '81922': [{u'image_id': '81922', u'caption': u'plane is flying through the sky'}],
    # }
    samples = {}
    IDs = []
    for item in prediction_dict.items():
        IDs.append(item[0])
        samples[item[0]] = [{
            u'image_id': item[0],
            u'caption': item[1]
        }]
    return samples, IDs


class COCOScorer(object):
    def __init__(self):
        print('init COCO-EVAL scorer')

    def score(self, GT, RES, IDs):
        self.eval = {}
        self.imgToEval = {}
        # 根据IDs，把要检测的项目提取出来
        gts = {}
        res = {}
        for ID in IDs:
            #            print ID
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        # 获取token
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
    opt = Opt()
    prediction_dict = eval()
    # gts = csv_to_coco_gts(r'./data/video_corpus.csv', clean_only=False)
    # gts = mst_vrr_to_coco_gts(opt.train_source_file)
    # samples, IDs = pred_to_coco_samples_IDs(prediction_dict)
    #
    # scorer = COCOScorer()
    # scorer.score(gts, samples, IDs)
    #
    # print("***********************")
    # print(scorer.eval)
    # print("***********************")
    # print(scorer.imgToEval)

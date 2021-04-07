import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm
import time

from MultiTask2.dataloader import VideoDataset, collate_fn

from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Opt:
    model_path = r"./checkpoint/21_04_07_14_14_48-stop.pth"
    caption_file = r"./multi_task_data/captions_msvd_min5.json"
    feats_path = r"/media/omnisky/115a4abd-1149-406e-a3de-2c3a5d707b69/lzh/dual/data/MSVD/resnet152_feature"
    gts_file = r'./multi_task_data/gts_msvd_min5.json'
    batch_size = 10


def eval():
    opt = Opt()

    # prepare data
    validset = VideoDataset(opt.caption_file, opt.feats_path, mode='test')
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    word2ix = validset.word2ix
    ix2word = validset.ix2word
    vocab_size = len(word2ix)

    # load model
    model = torch.load(opt.model_path).to(device)

    ###
    ### start test
    ###

    pred_dict = {}
    for index, (feats, lengths, targets, IDs, masks) in enumerate(tqdm(valid_loader, desc="test")):
        # get prediction and cal loss
        model.eval()
        with torch.no_grad():
            preds = model(feats, lengths, mode='test')  # preds [B, L]
        # save result
        for ID, pred in zip(IDs, preds):
            word_preds = [ix2word[str(i.item())] for i in pred]
            if '<end>' in word_preds:
                word_preds = word_preds[:word_preds.index('<end>')]
            pred_dict[ID] = ' '.join(word_preds)

    return pred_dict


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
            # print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    # print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                # print("%s: %0.3f" % (method, score))

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


def score_by_coco(predictions, ground_truth):
    samples, IDs = pred_to_coco_samples_IDs(predictions, ground_truth)

    scorer = COCOScorer()
    scorer.score(ground_truth, samples, IDs)

    # print("***********************")
    # print(scorer.eval)
    # print("***********************")
    # print(scorer.imgToEval)
    return scorer.eval, scorer.imgToEval


if __name__ == '__main__':
    opt = Opt()
    # prediction_dict = beam_eval()  # beam search. needs a LOT of time.
    prediction_dict = eval()
    with open(opt.gts_file, encoding='utf-8') as f:
        gts = json.load(f)['gts']

    print("*****************************")
    print("evaluated by coco")
    start_time = time.time()
    score1, rslt1 = score_by_coco(prediction_dict, gts)
    end_time = time.time()
    for k, v in score1.items():
        print("{}: {:.4f}".format(k, v))
    print("evaluation time:{}".format(end_time - start_time))
    print("*****************************")


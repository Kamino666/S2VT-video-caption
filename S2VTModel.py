import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import json


class S2VTModel(nn.Module):
    def __init__(self, vocab_size, dim_ori_feat, dim_hidden=1000, word_embed=500,
                 rnn_dropout_p=0.2, dim_feat=500, num_layers=1, bidirectional=False):
        super(S2VTModel, self).__init__()
        # 视频层
        self.rnn1 = nn.LSTM(dim_feat, dim_hidden, batch_first=True, dropout=rnn_dropout_p
                            , num_layers=num_layers, bidirectional=bidirectional)
        # 文字层
        self.rnn2 = nn.LSTM(dim_hidden + word_embed, dim_hidden, batch_first=True, dropout=rnn_dropout_p
                            , num_layers=num_layers, bidirectional=bidirectional)

        self.dim_ori_feat = dim_ori_feat
        self.dim_feat = dim_feat
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
        self.word_embed = word_embed
        self.embedding = nn.Embedding(self.vocab_size, self.word_embed)
        self.feat_fc = nn.Linear(self.dim_ori_feat, self.dim_feat)

        self.out = nn.Linear(self.dim_hidden, self.vocab_size)

    def forward(self, feats, feat_lengths, targets=None, max_len=30, mode='train', teacher_forcing_rate=0):
        """
        :param teacher_forcing_rate: int: 0~1 0: no teacher forcing
        :param max_len: int: only works when validation
        :param feats: tensor[B, T, dim_ori_feat]
        :param feat_lengths: tensor[B, 1]
        :param targets: tensor[B, T, 1]: only works when mode is train
        :param mode: train or validation
        :return:
        """
        device = feats.device
        batch_size, n_frames, _ = feats.shape  # 获取视频feature的帧数
        # 对于视频层和文字层有两种不同的pad
        # padding_words = torch.zeros([batch_size, n_frames, self.word_embed], dtype=torch.long, device=device)
        padding_words = torch.zeros([batch_size, n_frames, self.word_embed], dtype=torch.float, device=device)
        padding_frames = torch.zeros([batch_size, 1, self.dim_feat], dtype=torch.float, device=device)

        # Encoding Stage
        # feats先压成低一点的维度
        feats = self.feat_fc(feats)
        pack_feats = pack_padded_sequence(feats, feat_lengths, batch_first=True)
        output1, state1 = self.rnn1(pack_feats)
        output1, _ = pad_packed_sequence(output1, batch_first=True, padding_value=0.0)
        input2 = torch.cat((output1, padding_words), dim=2)
        output2, state2 = self.rnn2(input2)

        # Decoding Stage
        seq_probs = []  # probability after softmax-log
        seq_preds = []  # max probability
        if mode == 'train':
            previous_words = None
            for i in range(targets.shape[1] - 1):  # <eos> not included
                # teacher-forcing的时候用目标词来控制长度和监督
                if previous_words is None or random.random() < teacher_forcing_rate:
                    current_words = self.embedding(targets[:, i])
                else:
                    current_words = previous_words
                # 用pad和上一次的state得到结果
                output1, state1 = self.rnn1(padding_frames, state1)
                # 结果和下一个目标词合并
                input2 = torch.cat((output1, current_words.view([batch_size, 1, -1])), dim=2)
                # 生成新的词向量输出和state
                output2, state2 = self.rnn2(input2, state2)
                # 映射到vocab 求概率
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                # 得到预测值 logits: [batch_size, vocab_size] -> preds: [batch_size, 1]
                _, preds = torch.max(logits, 1, keepdim=True)
                # 更新previous_words
                previous_words = self.embedding(preds)
                # 存储结果
                seq_preds.append(preds)
                seq_probs.append(logits.unsqueeze(1))
            # 用cat转成tensor
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        elif mode == 'test':
            current_words = self.embedding(
                torch.ones([batch_size, 1], dtype=torch.long, device=device)
            )
            for i in range(max_len):  # <eos> not included
                # 用pad和上一次的state得到结果
                output1, state1 = self.rnn1(padding_frames, state1)
                # 结果和下一个目标词合并
                input2 = torch.cat((output1, current_words), dim=2)
                # 生成新的词向量输出和state
                output2, state2 = self.rnn2(input2, state2)
                # 映射到vocab 求概率
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                # 得到预测值 logits: [batch_size, vocab_size] -> preds: [batch_size, 1]
                _, preds = torch.max(logits, 1, keepdim=True)
                # 更新current_words
                current_words = self.embedding(preds)
                # 存储结果
                seq_preds.append(preds)
                seq_probs.append(logits.unsqueeze(1))
            # 用cat转成tensor
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
            seq_preds = pad_after_eos(seq_preds)
        # assert 1==0
        return seq_probs, seq_preds


def pad_after_eos(preds):
    """
    把eos之后的全部置零
    :param preds: [batch_size, max_len, 1]
    :return:
    """
    if len(preds.shape) == 3:
        preds = preds.squeeze(2)
    for seq in preds:
        for i in range(preds.shape[1]):
            if seq[i].item() == 2:
                seq[i:] = 0
                break
    return preds


class S2VT(nn.Module):
    def __init__(self, vocab_size, feat_dim, length, dim_hid=500, dim_embed=500, feat_dropout=0, rnn_dropout=0,
                 out_dropout=0, num_layers=1, bidirectional=False, rnn_type='lstm', sos_ix=2):
        super(S2VT, self).__init__()
        # RNN
        if rnn_type.lower() == 'lstm':
            rnn_cell = nn.LSTM
        else:
            rnn_cell = nn.GRU
        self.vid_rnn = rnn_cell(dim_hid, dim_hid, batch_first=True, num_layers=num_layers,
                                bidirectional=bidirectional, dropout=rnn_dropout)
        self.word_rnn = rnn_cell(dim_hid + dim_embed, dim_hid, batch_first=True, num_layers=num_layers,
                                 bidirectional=bidirectional, dropout=rnn_dropout)
        # other layers
        self.feat_drop = nn.Dropout(p=feat_dropout)
        self.out_drop = nn.Dropout(p=out_dropout)
        self.feat_linear = nn.Linear(feat_dim, dim_hid)
        self.out_linear = nn.Linear(dim_hid, vocab_size)
        self.embedding = nn.Embedding(vocab_size, dim_hid)
        # save parameters
        self.feat_dim = feat_dim
        self.length = length
        self.dim_hid = dim_hid
        self.dim_embed = dim_embed
        self.sos_ix = sos_ix
        self.vocab_size = vocab_size

    def forward(self, feats, targets=None, mode='train'):
        device = feats.device
        batch_size = feats.shape[0]

        # feats [B, L, feat_dim]
        feats = self.feat_drop(feats)
        # feats [B, L, dim_hid]
        feats = self.feat_linear(feats)
        # pad_feats [B, 2L-1, dim_hid]
        padding = torch.zeros([batch_size, self.length - 1, self.dim_hid], device=device)
        pad_feats = torch.cat([feats, padding], dim=1)
        # RNN1
        output1, state1 = self.vid_rnn(pad_feats)

        if mode == 'train':
            # targets [B, L-1, 1] embed [B, L-1, hid]
            embed = self.embedding(targets)
            padding = torch.zeros([batch_size, self.length, self.dim_embed], dtype=torch.long, device=device)
            pad_embed = torch.cat([padding, embed], dim=1)
            # input2 [B, 2L-1, 2hid]
            input2 = torch.cat([pad_embed, output1], dim=2)
            # RNN2
            output2, state2 = self.word_rnn(input2)
            result = output2[:, self.length:, :]
            result = self.out_drop(result)
            result = self.out_linear(result)
            return result
        else:
            """Encoding Stage of word_rnn layer"""
            padding = torch.zeros([batch_size, self.length, self.dim_embed], device=device)
            input2 = torch.cat([padding, output1[:, :self.length, :]], dim=2)
            _, state2 = self.word_rnn(input2)

            """Decoding Stage of word_rnn layer"""
            sos = (self.sos_ix * torch.ones([batch_size], dtype=torch.long)).to(device)
            sos = self.embedding(sos).unsqueeze(dim=1)
            input2 = torch.cat([sos, output1[:, self.length, :].unsqueeze(dim=1)], dim=2)
            # output2 [B, 1, hid]
            output2, state2 = self.word_rnn(input2, state2)
            # get first word [B, vocab_size] -> [B]
            current_word = self.out_linear(output2.squeeze(dim=1))
            current_word = torch.argmax(current_word, dim=1)
            pred = [current_word]
            for i in range(self.length - 2):
                # input2 [B, 1, hid]
                input2 = self.embedding(current_word.unsqueeze(1))
                input2 = torch.cat([input2, output1[:, self.length + i + 1, :].unsqueeze(dim=1)], dim=2)
                # [B, 1, 2hid] -> [B, 1, hid]
                output2, state2 = self.word_rnn(input2, state2)
                # get this word [B, vocab_size] -> [B]
                current_word = self.out_linear(output2.squeeze(dim=1))
                current_word = torch.argmax(current_word, dim=1)
                pred.append(current_word)
            pred = torch.cat(pred, dim=0).view(self.length - 1, batch_size)
            # print(pred.shape)
            return pred.transpose(dim0=0, dim1=1)

    def _get_word2embed_from_glove(self, glove_path, ix2word):
        f = open(glove_path, encoding='utf-8')
        word2embed = {}
        for line in tqdm(f.readlines(), desc='loading GloVe'):
            vector = line.split(' ')
            word = vector[0]  # str
            if word not in ix2word.values():  # 优化
                continue
            embed = []
            for str_num in vector[1:]:
                str_num = str_num.replace('\n', '')
                embed.append(eval(str_num))
            word2embed[word] = embed
            # word2embed[word] = torch.tensor(embed, dtype=torch.float, device=torch.device('cuda'))
        f.close()
        return word2embed

    def load_glove_weights(self, glove_path, glove_dim, ix2word, word2embed='./data/word2embed.json'):
        assert glove_dim == self.dim_embed
        if word2embed is None:
            word2embed = self._get_word2embed_from_glove(glove_path, ix2word)
            with open('./data/word2embed.json', 'w+', encoding='utf-8') as fp:
                json.dump(word2embed, fp)
        else:
            with open('./data/word2embed.json', encoding='utf-8') as fp:
                word2embed = json.load(fp)

        print('get {} word2embed'.format(len(word2embed)))

        weights = torch.zeros([self.vocab_size, glove_dim], dtype=torch.float, device=torch.device('cuda'))
        torch.nn.init.xavier_normal_(weights)
        for ix, word in ix2word.items():
            if word in word2embed:
                single_word_embed = torch.tensor(word2embed[word], dtype=torch.float, device=torch.device('cuda'))
                weights[int(ix)] = single_word_embed
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)

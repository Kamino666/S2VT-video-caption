import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        padding_words = torch.zeros([batch_size, n_frames, self.word_embed], dtype=torch.long, device=device)
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

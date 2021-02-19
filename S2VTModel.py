import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class S2VTModel(nn.Module):
    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, sos_id=1, eos_id=0,
                 n_layers=1, rnn_cell='gru', rnn_dropout_p=0.2):
        super(S2VTModel, self).__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        # 视频层
        self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers,
                                  batch_first=True, dropout=rnn_dropout_p)
        # 文字层
        self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers,
                                  batch_first=True, dropout=rnn_dropout_p)

        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

    def forward(self, feats, feat_lengths, targets, mode='train'):
        device = feats.device
        batch_size, n_frames, _ = feats.shape  # 获取视频feature的帧数
        # 对于视频层和文字层有两种不同的pad
        padding_words = torch.zeros([batch_size, n_frames, self.dim_word], dtype=torch.long, device=device)
        padding_frames = torch.zeros([batch_size, 1, self.dim_vid], dtype=torch.float, device=device)

        # Encoding Stage
        # 把视频序列特征喂给rnn1，然后用word的pad结果之后喂给rnn2
        state1 = None
        state2 = None
        pack_feats = pack_padded_sequence(feats, feat_lengths, batch_first=True)
        output1, state1 = self.rnn1(pack_feats, state1)
        output1, _ = pad_packed_sequence(output1, batch_first=True, padding_value=0.0)
        input2 = torch.cat((output1, padding_words), dim=2)
        output2, state2 = self.rnn2(input2, state2)

        # Decoding Stage
        # rnn1接收pad，结果和上一个词合并之后输入到rnn2，一次循环输出一个词
        seq_probs = []
        seq_preds = []
        if mode == 'train':
            # init current_words with <sos> 1    current_words: [batch_size, 1]
            # current_words = self.embedding(torch.ones([batch_size, 1], dtype=torch.long, device=device))
            for i in range(targets.shape[1] - 1):  # <eos> not included
                # 用目标词来控制长度和监督
                current_words = self.embedding(targets[:, i])
                # 用pad和上一次的state得到结果
                output1, state1 = self.rnn1(padding_frames, state1)
                # 结果和下一个目标词合并
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
                # 生成新的词向量输出和state
                output2, state2 = self.rnn2(input2, state2)
                # 映射到vocab 求概率
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                # logits: [batch_size, vocab_size] -> preds: [batch_size, 1]
                _, preds = torch.max(logits, 1, keepdim=True)
                # 存储结果
                seq_preds.append(preds)
                seq_probs.append(logits.unsqueeze(1))
            # 用cat转成tensor
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        # TODO(Kamino): 写好teacher-forcing的部分
        # teacher-forcing 暂时写不出来
        # else:
        #     # 预测的时候，current_words初始化为[b, 1]的序列，即batch全都是SOS
        #     current_words = self.embedding(
        #         Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())
        #     for i in range(self.max_length - 1):
        #         # 莫名其妙的优化
        #         self.rnn1.flatten_parameters()
        #         self.rnn2.flatten_parameters()
        #         # 老样子
        #         output1, state1 = self.rnn1(padding_frames, state1)
        #         # 老样子，不过current_words是实际预测出来的
        #         input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
        #         # 老样子
        #         output2, state2 = self.rnn2(input2, state2)
        #         # 老样子，求概率
        #         logits = self.out(output2.squeeze(1))
        #         logits = F.log_softmax(logits, dim=1)
        #         seq_probs.append(logits.unsqueeze(1))
        #         # 另外直接用max求出结果，元组第二个是下标
        #         _, preds = torch.max(logits, 1)
        #         current_words = self.embedding(preds)
        #         seq_preds.append(preds.unsqueeze(1))
        #     seq_probs = torch.cat(seq_probs, 1)
        #     seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds
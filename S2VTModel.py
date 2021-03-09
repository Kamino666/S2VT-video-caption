import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import json
from queue import PriorityQueue


class S2VT(nn.Module):
    def __init__(self, vocab_size, feat_dim, length, dim_hid=500, dim_embed=500, feat_dropout=0, rnn_dropout=0,
                 out_dropout=0, num_layers=1, bidirectional=False, rnn_type='lstm', sos_ix=3, eos_ix=4):
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
        self.embedding = nn.Embedding(vocab_size, dim_embed)
        # save parameters
        self.feat_dim = feat_dim
        self.length = length
        self.dim_hid = dim_hid
        self.dim_embed = dim_embed
        self.sos_ix = sos_ix
        self.eos_ix = eos_ix
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type

    def forward(self, feats, targets=None, mode='train', beam_width=3, max_beam_depth=30):
        """
        :param feats: [B, L, feat_dim]
        :param targets: [B, L-1, 1]
        :param mode: train: fix length training   test: greedy   beam_search: beam search
        :param beam_width:
        :param max_beam_depth:
        :return:
        """
        device = feats.device
        batch_size = feats.shape[0]

        # feats [B, L, feat_dim]
        feats = self.feat_drop(feats)
        # feats [B, L, dim_hid]
        feats = self.feat_linear(feats)

        if mode == 'beam_search':
            output1, state1 = self.vid_rnn(feats)
            padding = torch.zeros([batch_size, self.length, self.dim_embed], dtype=torch.long, device=device)
            input2 = torch.cat([padding, output1], dim=2)
            _, state2 = self.word_rnn(input2)
            return self.beam_search(state1, state2, beam_width=beam_width, max_depth=max_beam_depth)

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
            # input2 [B, 2L-1, hid+embed]
            input2 = torch.cat([pad_embed, output1], dim=2)
            # RNN2
            output2, state2 = self.word_rnn(input2)
            result = output2[:, self.length:, :]
            result = self.out_drop(result)
            result = self.out_linear(result)
            return result
        elif mode == 'test':
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

    def beam_search(self, state1, state2, beam_width=3, max_depth=30):
        """
        modified from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        beam_search after vid_rnn encoding and word_rnn encoding
        DO NOT SUPPORT GRU
        :param max_depth: max depth of search
        :param state1: the hidden state of vid_rnn layer
        :param state2: the hidden state of word_rnn encoding stage
                    LSTM: (h_n, c_n)
                    all hidden shape is (num_layers * num_directions, B, hidden_size)
        :param beam_width: width of beam search
        :return: a sequence of index list([seq], [seq], ..., [seq])
        """
        state1 = state1[0].transpose(dim0=0, dim1=1), state1[1].transpose(dim0=0, dim1=1)
        state2 = state2[0].transpose(dim0=0, dim1=1), state2[1].transpose(dim0=0, dim1=1)
        state1 = list(zip(state1[0], state1[1]))  # [[h, c], [h, c], ..., [h, c]]
        state2 = list(zip(state2[0], state2[1]))
        batch_size = len(state1)
        device = state1[0][0].device

        sentences = []
        for batch in tqdm(range(batch_size), desc='dealing with sentences'):
            # prepare
            # state[batch] LSTM: [h, c]
            vid_state = (state1[batch][0].unsqueeze(dim=0), state1[batch][1].unsqueeze(dim=0))
            word_state = (state2[batch][0].unsqueeze(dim=0), state2[batch][1].unsqueeze(dim=0))

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[self.sos_ix]]).to(device)
            # starting node -  vid hid, word hid, previous node, word id, logp, length
            node = BeamSearchNode(vid_state, word_state, None, decoder_input, 0, 1)
            nodes = PriorityQueue()
            # put <sos> in, start the queue  (score, node)
            nodes.put((-node.eval(), node))

            # start beam search
            depth_count = 0
            while depth_count < max_depth:
                # print("depth++ nodes size:{}".format(len(nodes.queue)))
                depth_count += 1
                # fetch the best beam_width node, put them into the beam_nodes, then reset the PriorityQueue
                beam_nodes = []  # length: beam_width
                for i in range(beam_width):
                    if not nodes.empty():
                        beam_nodes.append(nodes.get())
                nodes.queue.clear()
                # print("get beam_nodes")

                # put the next probs of beam_nodes into the PriorityQueue
                for score, n in beam_nodes:
                    # if detect a full sentence
                    if n.wordid.item() == self.eos_ix and n.prevNode is not None:
                        nodes.put((score, n))
                        continue

                    # prepare data
                    vid_hid = n.vid_hid
                    word_hid = n.word_hid
                    embed_word = self.embedding(n.wordid)
                    padding = torch.zeros([1, 1, self.dim_hid]).to(device)
                    # get prob
                    vid_out, next_vid_hid = self.vid_rnn(padding, vid_hid)
                    decoder_input = torch.cat([embed_word.view([1, 1, -1]), vid_out], dim=2)
                    next_word, next_word_hid = self.word_rnn(decoder_input, word_hid)
                    next_word = self.out_linear(next_word).view(-1)  # [vocab_size]
                    next_word = F.log_softmax(next_word, dim=0)
                    # put node into the PriorityQueue
                    _, top_ixs = next_word.topk(20)
                    for i, prob in enumerate(next_word):
                        if i not in top_ixs:
                            continue
                        node = BeamSearchNode(next_vid_hid, next_word_hid, n, torch.tensor(i, device=device), prob,
                                              n.leng + 1)
                        # python PriorityQueue only support get smallest, use -1 to get biggest
                        nodes.put((-node.eval(), node))

                # print("put the next probs")
                # if length of PriorityQueue is equal to beam_width, beam search done
                if len(nodes.queue) <= beam_width:
                    break

            # get the result
            _, final_node = nodes.get()
            sentence = [final_node.wordid]
            # back trace
            while final_node.prevNode is not None:
                final_node = final_node.prevNode
                sentence.append(final_node.wordid)
            sentence = sentence[::-1]  # reverse
            sentences.append(sentence)

        return sentences


class BeamSearchNode(object):
    def __init__(self, vid_hid, word_hid, previousNode, wordId, logProb, length):
        """
        :param vid_hid:
        :param word_hid:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.vid_hid = vid_hid[0].view([1, 1, -1]), vid_hid[1].view([1, 1, -1])
        self.word_hid = word_hid[0].view([1, 1, -1]), word_hid[1].view([1, 1, -1])
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb  # -9999999~0, to maxmize
        self.leng = length
        self.score = None

    def eval(self, alpha=0.7):
        """
        :param alpha: 1: normalizing by length 0: no normalizing
        """
        # score = self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        if self.score is None:
            score = self.logp / pow(float(self.leng), alpha)
            self.score = score
        return self.score

    def __gt__(self, other):  # usually no use, add this to prevent Error
        if other.eval() > self.eval():
            return True
        return False

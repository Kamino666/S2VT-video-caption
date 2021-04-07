import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VidEncoder(nn.Module):
    def __init__(self, feat_dim, dim_hid=1024, rnn_type="gru", bidirectional=True):
        super(VidEncoder, self).__init__()
        if rnn_type.lower() == 'gru':
            self.rnn = torch.nn.GRU(feat_dim, dim_hid, batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = torch.nn.LSTM(feat_dim, dim_hid, batch_first=True, bidirectional=bidirectional)

    def forward(self, feats, lengths):
        batch_size = feats.shape[0]
        packed_feats = pack_padded_sequence(feats, lengths, batch_first=True)
        outputs, _ = self.rnn(packed_feats)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def freeze(self):
        self.rnn.weight_hh_l0.requires_grad = False
        self.rnn.weight_ih_l0.requires_grad = False
        self.rnn.bias_hh_l0.requires_grad = False
        self.rnn.bias_ih_l0.requires_grad = False


class Att_Baseline(nn.Module):
    def __init__(self, vocab_size, dim_feat, length, dim_hid=500, dim_embed=500,
                 out_dropout=0, sos_ix=1, eos_ix=2, rnn_type='lstm'):
        super(Att_Baseline, self).__init__()
        # save parameters
        self.dim_feat = dim_feat
        self.length = length
        self.dim_hid = dim_hid
        self.dim_embed = dim_embed
        self.sos_ix = sos_ix
        self.eos_ix = eos_ix
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type

        # layers
        self.encoder = VidEncoder(dim_feat, dim_hid=dim_hid).to(device)
        if rnn_type.lower() == 'lstm':
            self.decoder = nn.LSTM(dim_hid * 2 + dim_embed, dim_hid, batch_first=True)
        else:
            self.decoder = nn.GRU(dim_hid * 2 + dim_embed, dim_hid, batch_first=True)
        # self.feat_linear = nn.Linear(dim_feat, dim_hid)
        # self.feat_drop = nn.Dropout(p=feat_dropout)
        self.embedding = nn.Embedding(vocab_size, dim_embed, padding_idx=0)
        self.out_linear = nn.Linear(dim_hid, vocab_size)
        self.out_drop = nn.Dropout(p=out_dropout)
        # attention layers
        self.att_enc = nn.Linear(dim_hid * 2, dim_hid, bias=True)
        self.att_prev_hid = nn.Linear(dim_hid, dim_hid, bias=True)
        self.att_apply = nn.Linear(dim_hid, 1, bias=False)

    def attention(self, enc_outputs, dec_prev_hid=None):
        """
        :param enc_outputs: [B, L, dim_hid * 2]
        :param dec_prev_hid: [1, B, dim_hid]
        :return:
        """
        batch_size = enc_outputs.shape[0]
        if dec_prev_hid is None:
            dec_prev_hid = torch.zeros([1, batch_size, self.dim_hid], device=device)
        length = enc_outputs.shape[1]
        # enc_outputs[B, L, dim_hid * 2] -> enc_W_h[B, L, dim_hid]
        enc_W_h = self.att_enc(enc_outputs)
        # dec_prev_hid[1, B, dim_hid] --repeat--> repeat_hid[B, L, dim_hid]
        repeat_hid = dec_prev_hid.transpose(dim0=1, dim1=0)
        repeat_hid = repeat_hid.repeat([1, length, 1])
        dec_W_h = self.att_prev_hid(repeat_hid)

        # et[B, L, 1]
        et = self.att_apply(torch.tanh(torch.add(enc_W_h, dec_W_h)))
        # at[B, 1, L] The probability of each frame feature
        at = torch.softmax(et, dim=2).squeeze(2).unsqueeze(1)
        # ct = sum{at_i * h_i}  context[B, 1, dim_hid * 2]
        context = torch.bmm(at, enc_outputs)
        return context

    def forward(self, feats, feat_lengths, targets=None, mode='train'):
        # save info
        batch_size = feats.shape[0]

        # encoding stage
        # feats = self.feat_linear(self.feat_drop(feats))
        # feats[B,L,dim_hid] -> enc_outputs[B, L, dim_hid * 2]
        enc_outputs = self.encoder(feats, feat_lengths)

        # decoding stage
        if mode == 'train':
            context = self.attention(enc_outputs)  # [B, 1, dim_hid * 2]
            embed_targets = self.embedding(targets)  # [B, L-1, dim_embed]
            state = None
            probs = []
            for i in range(self.length - 1):  # using L-1 target to predict L-1 words
                current_word = embed_targets[:, i, :].unsqueeze(1)  # [B, 1, dim_embed]
                dec_input = torch.cat([current_word, context], dim=2)

                # output[B, 1, dim_hid*2] hidden[2, b, dim_hid]
                dec_output, state = self.decoder(dec_input, state)
                if self.rnn_type.lower() == 'lstm':
                    context = self.attention(enc_outputs, state[0])
                else:
                    context = self.attention(enc_outputs, state)

                # prob[B, 1, vocab_size]
                prob = self.out_linear(self.out_drop(dec_output))
                probs.append(prob)
            return torch.cat(probs, dim=1)
        elif mode == 'test':
            # current_word[B, 1, dim_embed]
            current_word = (self.sos_ix * torch.ones([batch_size], dtype=torch.long)).to(device)
            current_word = self.embedding(current_word).view([batch_size, 1, -1])
            context = self.attention(enc_outputs)  # [B, 1, dim_hid * 2]
            state = None
            preds = []
            for i in range(self.length):
                dec_input = torch.cat([current_word, context], dim=2)

                # output[B, 1, dim_hid*2] hidden[2, b, dim_hid]
                dec_output, state = self.decoder(dec_input, state)
                # context = self.attention(enc_outputs, state[0])
                if self.rnn_type.lower() == 'lstm':
                    context = self.attention(enc_outputs, state[0])
                else:
                    context = self.attention(enc_outputs, state)

                # prob[B, 1, vocab_size]
                prob = self.out_linear(self.out_drop(dec_output))
                pred = torch.argmax(prob, dim=2)  # [B, 1]
                current_word = self.embedding(pred)  # [B, 1, dim_embed]
                preds.append(pred)
            return torch.cat(preds, dim=1)  # [B, L]

    def freeze_encoder_weights(self):
        self.encoder.freeze()
        self.embedding.weight.requires_grad = False
        print("********************\nfreeze weights success\n********************")

    def load_embedding_weight(self, weight_path, padding_idx=0):
        weight = np.load(weight_path)
        weight = torch.tensor(weight, device=device)
        self.embedding.from_pretrained(weight, padding_idx=padding_idx)
        self.embedding.weight.requires_grad = False

    def load_encoder_weight(self, weight_path):
        encoder_state = torch.load(weight_path)
        self.encoder.load_state_dict(encoder_state)


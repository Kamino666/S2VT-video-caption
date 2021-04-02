import torch
from torch import nn
import os

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Att_Baseline(nn.Module):
    def __init__(self, vocab_size, dim_feat, length, dim_hid=500, dim_embed=500, feat_dropout=0,
                 out_dropout=0, sos_ix=3, eos_ix=4, rnn_type='lstm'):
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
        self.encoder = nn.LSTM(dim_hid, dim_hid, batch_first=True, bidirectional=True)
        if rnn_type.lower() == 'lstm':
            self.decoder = nn.LSTM(dim_hid * 2 + dim_embed, dim_hid, batch_first=True)
        else:
            self.decoder = nn.GRU(dim_hid * 2 + dim_embed, dim_hid, batch_first=True)
        self.feat_linear = nn.Linear(dim_feat, dim_hid)
        self.feat_drop = nn.Dropout(p=feat_dropout)
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
        if dec_prev_hid is None:
            batch_size = enc_outputs.shape[0]
            dec_prev_hid = torch.zeros([1, batch_size, self.dim_hid], device=device)
        # enc_outputs[B, L, dim_hid * 2] -> enc_W_h[B, L, dim_hid]
        enc_W_h = self.att_enc(enc_outputs)
        # dec_prev_hid[1, B, dim_hid] --repeat--> repeat_hid[B, L, dim_hid]
        repeat_hid = dec_prev_hid.transpose(dim0=1, dim1=0)
        repeat_hid = repeat_hid.repeat([1, self.length, 1])
        dec_W_h = self.att_prev_hid(repeat_hid)

        # et[B, L, 1]
        et = self.att_apply(torch.tanh(torch.add(enc_W_h, dec_W_h)))
        # at[B, 1, L] The probability of each frame feature
        at = torch.softmax(et, dim=2).squeeze(2).unsqueeze(1)
        # ct = sum{at_i * h_i}  context[B, 1, dim_hid * 2]
        context = torch.bmm(at, enc_outputs)
        return context

    def forward(self, feats, targets=None, mode='train'):
        # save info
        batch_size = feats.shape[0]

        # encoding stage
        feats = self.feat_linear(self.feat_drop(feats))
        # feats[B,L,dim_hid] -> enc_outputs[B, L, dim_hid * 2]
        enc_outputs, _ = self.encoder(feats)

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

    def save_encoder_weights(self, path="./data/weights"):
        encoder_weight_hh_l0 = self.encoder.weight_hh_l0
        encoder_weight_ih_l0 = self.encoder.weight_ih_l0
        encoder_bias_hh_l0 = self.encoder.bias_hh_l0
        encoder_bias_ih_l0 = self.encoder.bias_ih_l0
        feat_linear_weight = self.feat_linear.weight
        feat_linear_bias = self.feat_linear.bias
        embedding_weight = self.embedding.weight
        torch.save(encoder_weight_hh_l0, os.path.join(path, "enc_weight_hh_l0"))
        torch.save(encoder_weight_ih_l0, os.path.join(path, "enc_weight_ih_l0"))
        torch.save(encoder_bias_hh_l0, os.path.join(path, "enc_bias_hh_l0"))
        torch.save(encoder_bias_ih_l0, os.path.join(path, "enc_bias_ih_l0"))
        torch.save(feat_linear_weight, os.path.join(path, "feat_linear_weight"))
        torch.save(feat_linear_bias, os.path.join(path, "feat_linear_bias"))
        torch.save(embedding_weight, os.path.join(path, "embedding_weight"))
        print("********************\nsave weights success\n********************")

    def load_encoder_weights(self, path="./data/weights"):
        self.encoder.weight_hh_l0 = torch.load(os.path.join(path, "enc_weight_hh_l0")).to(device)
        self.encoder.weight_ih_l0 = torch.load(os.path.join(path, "enc_weight_ih_l0")).to(device)
        self.encoder.bias_hh_l0 = torch.load(os.path.join(path, "enc_bias_hh_l0")).to(device)
        self.encoder.bias_ih_l0 = torch.load(os.path.join(path, "enc_bias_ih_l0")).to(device)
        self.feat_linear.weight = torch.load(os.path.join(path, "feat_linear_weight")).to(device)
        self.feat_linear.bias = torch.load(os.path.join(path, "feat_linear_bias")).to(device)
        self.embedding.weight = torch.load(os.path.join(path, "embedding_weight")).to(device)
        self.encoder.flatten_parameters()
        print("********************\nload weights success\n********************")

    def freeze_encoder_weights(self):
        self.encoder.weight_hh_l0.requires_grad = False
        self.encoder.weight_ih_l0.requires_grad = False
        self.encoder.bias_hh_l0.requires_grad = False
        self.encoder.bias_ih_l0.requires_grad = False
        self.feat_linear.weight.requires_grad = False
        self.feat_linear.bias.requires_grad = False
        self.embedding.weight.requires_grad = False
        print("********************\nfreeze weights success\n********************")


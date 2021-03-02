import torch
import torch.nn as nn


class MaskCriterion(nn.Module):

    def __init__(self):
        super(MaskCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


class LengthCriterion(nn.Module):

    def __init__(self):
        super(LengthCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss()

    def forward(self, probs, targets, lengths):
        """
        probs: shape of (N, seq_len, vocab_size)
        targets: shape of (N, seq_len)
        lengths: shape of (N, 1)
        """
        # truncate to the same size
        batch_size = probs.shape[0]
        seq_len = probs.shape[1]

        probs = probs.contiguous().view(probs.shape[0]*probs.shape[1], -1)
        targets = targets[:, :-1].contiguous().view(-1)
        loss = self.loss_fn(probs, targets)

        mask = torch.zeros([batch_size, seq_len], device=probs.device)
        for mask_item, length in zip(mask, lengths):
            mask_item[:length] = 1

        output = torch.sum(loss * mask) / batch_size
        return output


# 暂时不使用
class TeacherForcingScheduler:
    def __init__(self, rate_list, step_size=30):
        """
        :param rate_list: 范围从0到1, 列表形式
        :param step_size: 每经过step_size个step后，跳到下一个rate
        """
        self.rate_list = rate_list
        self.step_size = step_size
        self.length = len(rate_list)
        self.global_step = 0

    def step(self):
        if self.global_step + 1 >= self.length:
            return
        self.global_step += 1

    def get(self):
        return self.rate_list[self.global_step]

import torch
import torch.nn as nn
import numpy as np


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

        probs = probs.contiguous().view(probs.shape[0] * probs.shape[1], -1)
        targets = targets[:, :-1].contiguous().view(-1)
        loss = self.loss_fn(probs, targets)

        mask = torch.zeros([batch_size, seq_len], device=probs.device)
        for mask_item, length in zip(mask, lengths):
            mask_item[:length] = 1

        output = torch.sum(loss * mask) / batch_size
        return output


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """This class is from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# # 暂时不使用
# class TeacherForcingScheduler:
#     def __init__(self, rate_list, step_size=30):
#         """
#         :param rate_list: 范围从0到1, 列表形式
#         :param step_size: 每经过step_size个step后，跳到下一个rate
#         """
#         self.rate_list = rate_list
#         self.step_size = step_size
#         self.length = len(rate_list)
#         self.global_step = 0
#
#     def step(self):
#         if self.global_step + 1 >= self.length:
#             return
#         self.global_step += 1
#
#     def get(self):
#         return self.rate_list[self.global_step]

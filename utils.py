import torch
import torch.nn as nn
import numpy as np


class MaskCriterion(nn.Module):

    def __init__(self):
        super(MaskCriterion, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        item_sum = logits.shape[0]*logits.shape[1]  # N * seq_len
        target, mask = target[:, 1:], mask[:, 1:]
        # loss [N*seq_len]
        loss = self.loss_fn(logits.contiguous().view(item_sum, -1),
                            target.contiguous().view(-1))
        mask_loss = loss * mask.contiguous().view(-1)
        output = torch.sum(mask_loss) / torch.sum(mask)
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
        # torch.save(model.state_dict(), self.path)
        torch.save(model, self.path)
        self.val_loss_min = val_loss


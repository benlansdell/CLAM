import torch
import torch.autograd as ag

#from topk.polynomial.sp import log_sum_exp, LogSumExp
#from topk.logarithm import LogTensor
#from topk.utils import delta, split


def Top1_Hard_SVM(labels, alpha=1.):
    def fun(x, y):
        # max oracle
        max_, _ = (x + delta(y, labels, alpha)).max(1)
        # subtract ground truth
        loss = max_ - x.gather(1, y[:, None]).squeeze()
        return loss
    return fun


def Topk_Hard_SVM(labels, k, alpha=1.):
    def fun(x, y):
        x_1, x_2 = split(x, y, labels)

        max_1, _ = (x_1 + alpha).topk(k, dim=1)
        max_1 = max_1.mean(1)

        max_2, _ = x_1.topk(k - 1, dim=1)
        max_2 = (max_2.sum(1) + x_2) / k

        loss = torch.clamp(max_1 - max_2, min=0)

        return loss
    return fun


def Top1_Smooth_SVM(labels, tau, alpha=1.):
    def fun(x, y):
        # add loss term and subtract ground truth score
        x = x + delta(y, labels, alpha) - x.gather(1, y[:, None])
        # compute loss
        loss = tau * log_sum_exp(x / tau)

        return loss
    return fun


def Topk_Smooth_SVM(labels, k, tau, alpha=1.):

    lsp = LogSumExp(k)

    def fun(x, y):
        x_1, x_2 = split(x, y, labels)
        # all scores are divided by (k * tau)
        x_1.div_(k * tau)
        x_2.div_(k * tau)

        # term 1: all terms that will *not* include the ground truth score
        # term 2: all terms that will include the ground truth score
        res = lsp(x_1)
        term_1, term_2 = res[1], res[0]
        term_1, term_2 = LogTensor(term_1), LogTensor(term_2)

        X_2 = LogTensor(x_2)
        cst = x_2.data.new(1).fill_(float(alpha) / tau)
        One_by_tau = LogTensor(ag.Variable(cst, requires_grad=False))
        Loss_ = term_2 * X_2

        loss_pos = (term_1 * One_by_tau + Loss_).torch()
        loss_neg = Loss_.torch()
        loss = tau * (loss_pos - loss_neg)

        return loss
    return fun

import torch
import torch.autograd as ag

from numbers import Number


def log(x, like):
    """
    Get log-value of x.
    If x is a LogTensor, simply access its stored data
    If x is a Number, transform it to a tensor / variable,
    in the log space, with the same type and size as like.
    """
    if isinstance(x, LogTensor):
        return x.torch()

    if not isinstance(x, Number):
        raise TypeError('Not supported type: received {}, '
                        'was expected LogTensor or Number'
                        .format(type(x)))

    # transform x to variable / tensor of
    # same type and size as like
    like_is_var = isinstance(like, ag.Variable)
    data = like.data if like_is_var else like
    new = data.new(1).fill_(x).log_().expand_as(data)
    new = ag.Variable(new) if like_is_var else new
    return new


def _imul_inplace(x1, x2):
    return x1.add_(x2)


def _imul_outofplace(x1, x2):
    return x1 + x2


def _add_inplace(x1, x2):
    M = torch.max(x1, x2)
    M.add_(((x1 - M).exp_().add_((x2 - M).exp_())).log_())
    return M


def _add_outofplace(x1, x2):
    M = torch.max(x1, x2)
    return M + torch.log(torch.exp(x1 - M) + torch.exp(x2 - M))


class LogTensor(object):
    """
    Stable log-representation for torch tensors
    _x stores the value in the log space
    """
    def __init__(self, x):
        super(LogTensor, self).__init__()

        self.var = isinstance(x, ag.Variable)
        self._x = x
        self.add = _add_outofplace if self.var else _add_inplace
        self.imul = _imul_outofplace if self.var else _imul_inplace

    def __add__(self, other):
        other_x = log(other, like=self._x)
        return LogTensor(self.add(self._x, other_x))

    def __imul__(self, other):
        other_x = log(other, like=self._x)
        self._x = self.imul(self._x, other_x)
        return self

    def __iadd__(self, other):
        other_x = log(other, like=self._x)
        self._x = self.add(self._x, other_x)
        return self

    def __radd__(self, other):
        """
        Addition is commutative.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        NB: assumes self - other > 0.
        Will return nan otherwise.
        """
        other_x = log(other, like=self._x)
        diff = other_x - self._x
        x = self._x + log1mexp(diff)
        return LogTensor(x)

    def __pow__(self, power):
        return LogTensor(self._x * power)

    def __mul__(self, other):
        other_x = log(other, like=self._x)
        x = self._x + other_x
        return LogTensor(x)

    def __rmul__(self, other):
        """
        Multiplication is commutative.
        """
        return self.__mul__(other)

    def __div__(self, other):
        """
        Division (python 2)
        """
        other_x = log(other, like=self._x)
        x = self._x - other_x
        return LogTensor(x)

    def __truediv__(self, other):
        """
        Division (python 3)
        """
        return self.__div__(other)

    def torch(self):
        """
        Returns value of tensor in torch format (either variable or tensor)
        """
        return self._x

    def __repr__(self):
        tensor = self._x.data if self.var else self._x
        s = 'Log Tensor with value:\n{}'.format(tensor)
        return s


def log1mexp(U, eps=1e-3):
    """
    Compute log(1 - exp(u)) for u <= 0.
    """
    res = torch.log1p(-torch.exp(U))

    # |U| << 1 requires care for numerical stability:
    # 1 - exp(U) = -U + o(U)
    small = torch.lt(U.abs(), eps)
    res[small] = torch.log(-U[small])

    return res

import torch
import torch.nn as nn
import numpy as np
#import topk.functional as F

#from topk.utils import detect_large


class _SVMLoss(nn.Module):

    def __init__(self, n_classes, alpha):

        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer('labels', torch.from_numpy(np.arange(n_classes)))
        self.n_classes = n_classes
        self._tau = None

    def forward(self, x, y):

        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self._tau != tau:
            print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.get_losses()
        return self

    def cpu(self):
        nn.Module.cpu()
        self.get_losses()
        return self


class MaxTop1SVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None):

        super(MaxTop1SVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.get_losses()

    def forward(self, x, y):
        return self.F(x, y).mean()

    def get_losses(self):
        self.F = Top1_Hard_SVM(self.labels, self.alpha)


class MaxTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, k=5):

        super(MaxTopkSVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.k = k
        self.get_losses()

    def forward(self, x, y):
        return self.F(x, y).mean()

    def get_losses(self):
        self.F = Topk_Hard_SVM(self.labels, self.k, self.alpha)


class SmoothTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.):
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)

        return loss

    def get_losses(self):
        self.F_h = Top1_Hard_SVM(self.labels, self.alpha)
        self.F_s = Top1_Smooth_SVM(self.labels, self.tau, self.alpha)


class SmoothTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, tau=1., k=5):
        super(SmoothTopkSVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.k = k
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, self.k, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)

        return loss

    def get_losses(self):
        self.F_h = Topk_Hard_SVM(self.labels, self.k, self.alpha)
        self.F_s = Topk_Smooth_SVM(self.labels, self.k, self.tau, self.alpha)

import math
import torch

import torch.autograd as ag


def delta(y, labels, alpha=None):
    """
    Compute zero-one loss matrix for a vector of ground truth y
    """

    if isinstance(y, ag.Variable):
        labels = ag.Variable(labels, requires_grad=False)

    delta = torch.ne(y[:, None], labels[None, :]).float()

    if alpha is not None:
        delta = alpha * delta
    return delta


def split(x, y, labels):
    labels = ag.Variable(labels, requires_grad=False)
    mask = torch.ne(labels[None, :], y[:, None])

    # gather result:
    # x_1: all scores that do contain the ground truth
    x_1 = x[mask].view(x.size(0), -1)
    # x_2: scores of the ground truth
    x_2 = x.gather(1, y[:, None]).view(-1)
    return x_1, x_2


def detect_large(x, k, tau, thresh):
    top, _ = x.topk(k + 1, 1)
    # switch to hard top-k if (k+1)-largest element is much smaller
    # than k-largest element
    hard = torch.ge(top[:, k - 1] - top[:, k], k * tau * math.log(thresh)).detach()
    smooth = hard.eq(0)
    return smooth, hard

import torch


def divide_and_conquer(x, k, mul):
    """
    Divide and conquer method for polynomial expansion
    x is a 2d tensor of size (n_classes, n_roots)
    The objective is to obtain the k first coefficients of the expanded
    polynomial
    """

    to_merge = []

    while x[0].dim() > 1 and x[0].size(0) > 1:
        size = x[0].size(0)
        half = size // 2
        if 2 * half < size:
            to_merge.append([t[-1] for t in x])
        x = mul([t[:half] for t in x],
                [t[half: 2 * half] for t in x])

    for row in to_merge:
        x = mul(x, row)
    x = torch.cat(x)
    return x

import torch

from future.builtins import range
#from topk.logarithm import LogTensor


def recursion(S, X, j):
    """
    Apply recursive formula to compute the gradient
    for coefficient of degree j.
    d S[j] / d X = S[j-1] - X * (S[j-2] - X * (S[j-3] - ...) ... )
                 = S[j-1] + X ** 2 * S[j-3] + ...
                 - (X * S[j-2] + X ** 3 * S[j-4] + ...)
    """

    # Compute positive and negative parts separately
    _P_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 1, -1, -2))
    _N_ = sum(S[i] * X ** (j - 1 - i) for i in range(j - 2, -1, -2))

    return _N_, _P_


def approximation(S, X, j, p):
    """
    Compute p-th order approximation for d S[j] / d X:
    d S[j] / d X ~ S[j] / X - S[j + 1] /  X ** 2 + ...
                   + (-1) ** (p - 1) * S[j + p - 1] / X ** p
    """

    # Compute positive and negative parts separately
    _P_ = sum(S[j + i] / X ** (i + 1) for i in range(0, p, 2))
    _N_ = sum(S[j + i] / X ** (i + 1) for i in range(1, p, 2))

    return _N_, _P_


def d_logS_d_expX(S, X, j, p, grad, thresh, eps=1e-5):
    """
    Compute the gradient of log S[j] w.r.t. exp(X).
    For unstable cases, use p-th order approximnation.
    """

    # ------------------------------------------------------------------------
    # Detect unstabilites
    # ------------------------------------------------------------------------

    _X_ = LogTensor(X)
    _S_ = [LogTensor(S[i]) for i in range(S.size(0))]

    # recursion of gradient formula (separate terms for stability)
    _N_, _P_ = recursion(_S_, _X_, j)

    # deal with edge case where _N_ or _P_ is 0 instead of a LogTensor (happens for k=2):
    # fill with large negative values (numerically equivalent to 0 in log-space)
    if not isinstance(_N_, LogTensor):
        _N_ = LogTensor(-1.0 / eps * torch.ones_like(X))
    if not isinstance(_P_, LogTensor):
        _P_ = LogTensor(-1.0 / eps * torch.ones_like(X))

    P, N = _P_.torch(), _N_.torch()

    # detect instability: small relative difference in log-space
    diff = (P - N) / (N.abs() + eps)

    # split into stable and unstable indices
    u_indices = torch.lt(diff, thresh)  # unstable
    s_indices = u_indices.eq(0)  # stable

    # ------------------------------------------------------------------------
    # Compute d S[j] / d X
    # ------------------------------------------------------------------------

    # make grad match size and type of X
    grad = grad.type_as(X).resize_as_(X)

    # exact gradient for s_indices (stable) elements
    if s_indices.sum():
        # re-use positive and negative parts of recursion (separate for stability)
        _N_ = LogTensor(_N_.torch()[s_indices])
        _P_ = LogTensor(_P_.torch()[s_indices])
        _X_ = LogTensor(X[s_indices])
        _S_ = [LogTensor(S[i][s_indices]) for i in range(S.size(0))]

        # d log S[j] / d exp(X) = (d S[j] / d X) * X / S[j]
        _SG_ = (_P_ - _N_) * _X_ / _S_[j]
        grad.masked_scatter_(s_indices, _SG_.torch().exp())

    # approximate gradients for u_indices (unstable) elements
    if u_indices.sum():
        _X_ = LogTensor(X[u_indices])
        _S_ = [LogTensor(S[i][u_indices]) for i in range(S.size(0))]

        # positive and negative parts of approximation (separate for stability)
        _N_, _P_ = approximation(_S_, _X_, j, p)

        # d log S[j] / d exp(X) = (d S[j] / d X) * X / S[j]
        _UG_ = (_P_ - _N_) * _X_ / _S_[j]
        grad.masked_scatter_(u_indices, _UG_.torch().exp())

    return grad

import operator
import itertools

from future.builtins import range
from functools import reduce
#from topk.logarithm import LogTensor


def Multiplication(k):
    """
    Generate a function that performs a polynomial multiplication and return coefficients up to degree k
    """
    assert isinstance(k, int) and k > 0

    def isum(factors):
        init = next(factors)
        return reduce(operator.iadd, factors, init)

    def mul_function(x1, x2):

        # prepare indices for convolution
        l1, l2 = len(x1), len(x2)
        M = min(k + 1, l1 + l2 - 1)
        indices = [[] for _ in range(M)]
        for (i, j) in itertools.product(range(l1), range(l2)):
            if i + j >= M:
                continue
            indices[i + j].append((i, j))

        # wrap with log-tensors for stability
        X1 = [LogTensor(x1[i]) for i in range(l1)]
        X2 = [LogTensor(x2[i]) for i in range(l2)]

        # perform convolution
        coeff = []
        for c in range(M):
            coeff.append(isum(X1[i] * X2[j] for (i, j) in indices[c]).torch())
        return coeff

    return mul_function

import torch
import torch.nn as nn
import torch.autograd as ag

#from topk.polynomial.divide_conquer import divide_and_conquer
#from topk.polynomial.multiplication import Multiplication
#from topk.polynomial.grad import d_logS_d_expX


class LogSumExp(nn.Module):
    def __init__(self, k, p=None, thresh=1e-5):
        super(LogSumExp, self).__init__()
        self.k = k
        self.p = int(1 + 0.2 * k) if p is None else p
        self.mul = Multiplication(self.k + self.p - 1)
        self.thresh = thresh

        self.register_buffer('grad_k', torch.Tensor(0))
        self.register_buffer('grad_km1', torch.Tensor(0))

        self.buffers = (self.grad_km1, self.grad_k)

    def forward(self, x):
        f = LogSumExp_F(self.k, self.p, self.thresh, self.mul, self.buffers)
        return f(x)


class LogSumExp_F(ag.Function):
    def __init__(self, k, p, thresh, mul, buffers):
        self.k = k
        self.p = p
        self.mul = mul
        self.thresh = thresh

        # unpack buffers
        self.grad_km1, self.grad_k = buffers

    def forward(self, x):
        """
        Returns a matrix of size (2, n_samples) with sigma_{k-1} and sigma_{k}
        for each sample of the mini-batch.
        """
        self.save_for_backward(x)

        # number of samples and number of coefficients to compute
        n_s = x.size(0)
        kp = self.k + self.p - 1

        assert kp <= x.size(1)

        # clone to allow in-place operations
        x = x.clone()

        # pre-compute normalization
        x_summed = x.sum(1)

        # invert in log-space
        x.t_().mul_(-1)

        # initialize polynomials (in log-space)
        x = [x, x.clone().fill_(0)]

        # polynomial multiplications
        log_res = divide_and_conquer(x, kp, mul=self.mul)

        # re-normalize
        coeff = log_res + x_summed[None, :]

        # avoid broadcasting issues (in particular if n_s = 1)
        coeff = coeff.view(kp + 1, n_s)

        # save all coeff for backward
        self.saved_coeff = coeff

        return coeff[self.k - 1: self.k + 1]

    def backward(self, grad_sk):
        """
        Compute backward pass of LogSumExp.
        Python variables with an upper case first letter are in
        log-space, other are in standard space.
        """

        # tensors from forward pass
        X, = self.saved_tensors
        S = self.saved_coeff

        # extend to shape (self.k + 1, n_samples, n_classes) for backward
        S = S.unsqueeze(2).expand(S.size(0), X.size(0), X.size(1))

        # compute gradients for coeff of degree k and k - 1
        self.grad_km1 = d_logS_d_expX(S, X, self.k - 1, self.p, self.grad_km1, self.thresh)
        self.grad_k = d_logS_d_expX(S, X, self.k, self.p, self.grad_k, self.thresh)

        # chain rule: combine with incoming gradients (broadcast to all classes on third dim)
        grad_x = grad_sk[0, :, None] * self.grad_km1 + grad_sk[1, :, None] * self.grad_k

        return grad_x


def log_sum_exp(x):
    """
    Compute log(sum(exp(x), 1)) in a numerically stable way.
    Assumes x is 2d.
    """
    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))


def log_sum_exp_k_autograd(x, k):
    # number of samples and number of coefficients to compute
    n_s = x.size(0)

    assert k <= x.size(1)

    # clone to allow in-place operations
    x = x.clone()

    # pre-compute normalization
    x_summed = x.sum(1)

    # invert in log-space
    x.t_().mul_(-1)

    # initialize polynomials (in log-space)
    x = [x, x.clone().fill_(0)]

    # polynomial mulitplications
    log_res = divide_and_conquer(x, k, mul=Multiplication(k))

    # re-normalize
    coeff = log_res + x_summed[None, :]

    # avoid broadcasting issues (in particular if n_s = 1)
    coeff = coeff.view(k + 1, n_s)

    return coeff[k - 1: k + 1]


import numpy as np
import torch
from utils.utils import *
import os
import pickle 
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
#from topk import SmoothTop1SVM

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        #from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    #Original version commeted out here wouldn't resize if model type is 'mil',
    #below we remove that...
    # if args.model_size is not None and args.model_type != 'mil':
    #     #Here we want to detect what the dimension of our feature set is... and 
    #     if args.model_size == 'custom': args.model_size = datasets[0][0][0].shape[1]
    #     model_dict.update({"size_arg": args.model_size})
    
    if args.model_size is not None:
        #Here we want to detect what the dimension of our feature set is... and 
        if args.model_size == 'custom': args.model_size = datasets[0][0][0].shape[1]
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            #from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    train_eval_loader = get_split_loader(train_split, testing = args.testing)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=30, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    results_dict_train, train_error, train_auc, _= summary(model, train_loader, args.n_classes, save_activations = True)
    results_dict_train_eval, train_error, train_auc, _= summary(model, train_eval_loader, args.n_classes, save_activations = True)
    print('Train error: {:.4f}, ROC AUC: {:.4f}'.format(train_error, train_auc))

    results_dict_val, val_error, val_auc, _= summary(model, val_loader, args.n_classes, save_activations = True)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, save_activations = True)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    if args.save_activations:
        print("Saving activations!")
        #Make a new directory, if not already one
        activation_dir = os.path.join(args.results_dir, 'activations')
        if not os.path.isdir(activation_dir):
            os.mkdir(activation_dir)
        #Setup the saving filename
        fn_activations = os.path.join(activation_dir, f'activations_fold_{cur}.pkl')
        def save_activations(results_dict):
            t_labels = []
            t_activations = []
            t_ids = []
            for s_id in results_dict.keys():
                t_labels.append(results_dict[s_id]['label'])
                t_ids.append(s_id)
                t_activations.append(results_dict[s_id]['activations'].cpu().numpy())
            return t_ids, t_labels, t_activations
        assets = {'train': save_activations(results_dict_train), \
                  'val': save_activations(results_dict_val), \
                  'test': save_activations(results_dict)}
        with open(fn_activations, 'wb') as handle:
            pickle.dump(assets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if acc is None: acc = 0
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if val_error is None: val_error = 0
    if val_auc is None: val_auc = 0
    if test_error is None: test_error = 0
    if test_auc is None: test_auc = 0

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, results_dict_train_eval


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data[data == float("Inf")] = 1
        data[data > 1] = 1
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data_np = data.numpy()
            #test_line_ = 0
            data[data == float("Inf")] = 1
            data[data > 1] = 1
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, save_activations = False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data, return_features = True)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), \
                            'prob': probs, 'label': label.item()}})
        if save_activations:
            patient_results[slide_id]['activations'] = results_dict['features']
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger

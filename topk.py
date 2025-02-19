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

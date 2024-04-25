import torch.nn as nn
import torch

import torch.nn.functional as F

def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()
    activations = torch.reshape(activations, (activations.shape[0], -1))
    activations = F.softmax(activations,dim=-1)
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))
    ema_activations = F.softmax(activations,dim=-1)

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss

def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

cri  = nn.L1Loss().cuda()

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from utils import SinkhornDistance
import numpy as np
import random
def mixup_process(out, t, lam):
    out = out*lam + t*(1-lam)
    return out
def mixup_aligned(out, t, lam):
    # out:source_fea,
    # out shape = batch_size x 2048 x 8 x 8 (cifar10/100)

    indices = np.random.permutation(out.size(0))
    feat1 = out.view(out.shape[0], out.shape[1], -1)  # batch_size x 2048 x 64
    feat2 = t.view(out.shape[0], out.shape[1], -1)  # batch_size x 2048 x 64

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    P = sinkhorn(feat1.permute(0, 2, 1), feat2.permute(0, 2, 1)).detach()  # optimal plan batch x 16 x 16

    P = P * (out.size(2) * out.size(3))  # assignment matrix

    align_mix = random.randint(0, 1)  # uniformly choose at random, which alignmix to perform

    if (align_mix == 0):
        # \tilde{A} = A'R^{T}
        f1 = torch.matmul(feat2, P.permute(0, 2, 1).cuda()).view(out.shape)
        final = feat1.view(out.shape) * lam + f1 * (1 - lam)

    elif (align_mix == 1):
        # \tilde{A}' = AR
        f2 = torch.matmul(feat1, P.cuda()).view(out.shape).cuda()
        final = f2 * lam + feat2.view(out.shape) * (1 - lam)


    return final

lam = 0.4
layer_mix = random.randint(0,1)
x = torch.randn(32,3,256,256).cuda()
t = torch.randn(32,3,256,256).cuda()
if layer_mix == 0:
    x = mixup_process(x, t, lam)
    print(x.shape)

t = torch.randn(32,2048,8,8).cuda()
out = torch.randn(32,2048,8,8).cuda()

if layer_mix == 1:
    out = mixup_aligned(out, t, lam)
    print(out.shape)

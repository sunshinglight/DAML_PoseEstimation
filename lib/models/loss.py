"""
Modified from https://github.com/microsoft/human-pose-estimation.pytorch
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class JointsMSELoss(nn.Module):
    """
    Typical MSE loss for keypoint detection.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean'):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_gt = target.reshape((B, K, -1))
        loss = self.criterion(heatmaps_pred, heatmaps_gt) * 0.5
        if target_weight is not None:
            loss = loss * target_weight.view((B, K, 1))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)


class JointsKLLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt + self.epsilon
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
        if target_weight is not None:
            loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)

class EntLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(EntLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, threshold=-1):
        n, c, h, w = x.size()
        x = x.reshape((n, c, -1))
        P = F.softmax(x, dim=2) # [B, 18, 4096]
        logP = F.log_softmax(x, dim=2) # [B, 18, 4096]
        PlogP = P * logP               # [B, 18, 4096]
        ent = -1.0 * PlogP.sum(dim=2)  # [B, 18, 1]
        ent = ent / np.log(h * w)        # 8.3177661667: log_e(4096)
        if threshold > 0:
            ent = ent[ent < threshold]
        # compute robust entropy
        if self.reduction == 'mean':
            return ent.mean()
        elif self.reduction == 'none':
            return ent.mean(dim=-1)


class CstLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CstLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target, threshold=-1):
        n, c, h, w = output.size()
        # x = x.reshape((n, c, -1))
        output_h = output.mean(dim=-1)  # mean best
        output_w = output.mean(dim=-2)
        target_h = target.mean(dim=-1)
        target_w = target.mean(dim=-2)

        output_P_h = F.softmax(output_h, dim=2)  # [B, 18, 64]
        target_P_h = F.softmax(target_h, dim=2)
        h_sim_pos = F.cosine_similarity(output_P_h, target_P_h, dim=2)

        output_P_w = F.softmax(output_w, dim=2)  # [B, 18, 64]
        target_P_w = F.softmax(target_w, dim=2)
        w_sim_pos = F.cosine_similarity(output_P_w, target_P_w, dim=2)

        sim_pos = torch.mean((h_sim_pos + w_sim_pos) * 0.5)

        sim = torch.tensor(0.).cuda()

        for i in range(c):
            for j in range(c):
                val_h = F.cosine_similarity(output_P_h[:, i, :], target_P_h[:, j, :], dim=-1).cuda()
                val_w = F.cosine_similarity(output_P_w[:, i, :], target_P_w[:, j, :], dim=-1).cuda()
                sim += torch.mean((val_h + val_w) * 0.5)

        loss = - torch.log(sim_pos / sim)
        loss /= c
        loss /= n

        return loss


class ResLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(ResLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        mark_i = torch.argmax(heatmaps_pred, dim=-1)

        heatmaps_gt = target.reshape((B, K, -1))
        mark_j = torch.argmax(heatmaps_gt, dim=-1)

        mask = torch.ones_like(heatmaps_pred)
        mask[:, :, mark_i] = 0
        mask[:, :, mark_j] = 0
        heatmaps_pred *= mask
        heatmaps_gt *= mask

        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = F.log_softmax(heatmaps_gt, dim=-1)

        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
        if target_weight is not None:
            loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)

class ConsLoss(nn.Module):

    def __init__(self):
        super(ConsLoss, self).__init__()

    def forward(self, stu_out, tea_out, valid_mask=None, tea_mask=None): # b, c, h, w
        diff = stu_out - tea_out # b, c, h, w
        if tea_mask is not None:
            diff *= tea_mask[:, :, None, None] # b, c, h, w
        loss_map = torch.mean((diff) ** 2, dim=1) # b, h, w
        if valid_mask is not None:
            loss_map = loss_map[valid_mask]

        return loss_map.mean()
    # def forward(self, stu_out, tea_out, valid_mask=None, tea_mask=None):
    #     diff = stu_out - tea_out
    #     if tea_mask is not None:
    #         diff *= tea_mask[:, :, None, None]
    #     loss_map = torch.mean(torch.abs(diff), dim=1)  # 使用平滑L1损失函数替代平方损失函数
    #     if valid_mask is not None:
    #         loss_map = loss_map[valid_mask]
    #     return loss_map.mean()

class ConsSoftmaxLoss(nn.Module):

    def __init__(self):
        super(ConsSoftmaxLoss, self).__init__()

    def forward(self, stu_out, tea_out, valid_mask=None, tea_mask=None): # b, c, h, w
        B, K, H, W = stu_out.shape

        stu_out = F.softmax(stu_out.reshape((B, K, -1)), dim=-1).reshape(B, K, H, W)
        tea_out = F.softmax(tea_out.reshape((B, K, -1)), dim=-1).reshape(B, K, H, W)

        diff = stu_out - tea_out # b, c, h, w
        if tea_mask is not None:
            diff *= tea_mask[:, :, None, None] # b, c, h, w
        loss_map = torch.mean((diff) ** 2, dim=1) # b, h, w
        if valid_mask is not None:
            loss_map = loss_map[valid_mask]

        return loss_map.mean()

class ConsKLLoss(nn.Module):

    def __init__(self):
        super(ConsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, stu_out, tea_out, valid_mask=None, tea_mask=None): # b, c, h, w
        B, K, H, W = stu_out.shape
        stu_out = stu_out.reshape((B, K, -1))
        stu_out = F.log_softmax(stu_out, dim=-1)
        tea_out = tea_out.reshape((B, K, -1))
        tea_out = F.log_softmax(tea_out, dim=-1)
        loss_map = self.criterion(stu_out, tea_out).reshape((B, K, H, W))
        if tea_mask is not None:
            loss_map *= tea_mask[:, :, None, None] # b, c, h, w
        loss_map = torch.mean(loss_map, dim=1) # b, h, w
        if valid_mask is not None:
            loss_map = loss_map[valid_mask]

        return loss_map.mean()


class CoralLoss(nn.Module):

    def __init__(self, coral_downsample, prior=None):
        super(CoralLoss, self).__init__()
        self.coral_downsample = coral_downsample
        self.prior = prior

    def forward(self, src_out, tgt_out): 
        if self.coral_downsample > 1:
            tgt_out = F.interpolate(tgt_out, scale_factor=1/self.coral_downsample, mode='bilinear')

        n, c, h, w = tgt_out.size()
        tgt_out = tgt_out.view(n, -1)

        if self.prior is not None:
            cs = self.prior
        else:
            # source covariance
            if self.coral_downsample > 1:
                src_out = F.interpolate(src_out, scale_factor=1/self.coral_downsample, mode='bilinear')
            src_out = src_out.view(n, -1)
            tmp_s = torch.ones((1, n)).cuda() @ src_out
            cs = (src_out.T @ src_out - (tmp_s.T @ tmp_s) / n) / (n - 1)

        # target covariance
        tmp_t = torch.ones((1, n)).cuda() @ tgt_out
        ct = (tgt_out.T @ tgt_out - (tmp_t.T @ tmp_t) / n) / (n - 1)

        # frobenius norm
        loss = (cs - ct).pow(2).sum().sqrt()
        loss = loss / (4 * (c * h * w)**2)

        return loss


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



def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()
    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))
    activations = F.softmax(activations,dim=-1)
    ema_activations = F.softmax(ema_activations,dim=-1)

    similarity = activations.mm(activations.t())
    epsilon = torch.tensor(1e-8)
    norm = torch.reshape(torch.norm(similarity+epsilon, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity+epsilon, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss


class SimCCLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SimCCLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, threshold=-1):
        n, c, h, w = x.size()
        #x = x.reshape((n, c, -1))
        x_h= x.mean(dim=-1) # mean best
        x_w= x.mean(dim=-2)

        P_h = F.softmax(x_h, dim=2) # [B, 18, 64]
        logP_h = F.log_softmax(x_h, dim=2) # [B, 18, 64]
        PlogP_h = P_h * logP_h
        ent_h = -1.0 * PlogP_h.sum(dim=2)
        ent_h = ent_h / (h * w)
        if threshold > 0:
            ent_h = ent_h[ent_h < threshold]
        if self.reduction == 'mean':
            ent_h = ent_h.mean()
        elif self.reduction == 'none':
            ent_h = ent_h.mean(dim=-1)

        P_w = F.softmax(x_w, dim=2) # [B, 18, 64]
        logP_w = F.log_softmax(x_w, dim=2) # [B, 18, 64]
        PlogP_w = P_w * logP_w
        ent_w = -1.0 * PlogP_w.sum(dim=2)
        ent_w = ent_w / (h * w)
        if threshold > 0:
            ent_w = ent_w[ent_w < threshold]
        if self.reduction == 'mean':
            ent_w = ent_w.mean()
        elif self.reduction == 'none':
            ent_w = ent_w.mean(dim=-1)

        x = x.reshape((n, c, -1)) # [b, 18, 4096]
        x_m = x.mean(dim=0) # [18, 4096]
        P_m = F.softmax(x_m, dim=-1)
        logP_m = F.log_softmax(x_m, dim=-1)
        PlogP_m = P_m * logP_m
        ent_m = -1.0 * PlogP_m.sum(dim=-1)
        ent_m = ent_m / (h * w)

        if self.reduction == 'mean':
            ent_m = ent_m.mean()
        elif self.reduction == 'none':
            ent_m = ent_m.mean(dim=-1)

        x_c = x.mean(dim=1)#[b,4096]
        P_c = F.softmax(x_c, dim=-1)
        logP_c = F.log_softmax(x_c, dim=-1)
        PlogP_c = P_c * logP_c
        ent_c = -1.0 * PlogP_c.sum(dim=-1)
        ent_c = ent_c / (h * w)
        if self.reduction == 'mean':
            ent_c = ent_c.mean()
        elif self.reduction == 'none':
            ent_c = ent_c.mean(dim=-1)

        return ent_h + ent_w - ent_m




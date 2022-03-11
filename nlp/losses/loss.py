# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: loss
    Author: czh
    Create Date: 2021/8/9
--------------------------------------
    Change Activity: 
======================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as func


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素非0即1，
           1表示对应的类为目标类，0表示对应的类为非目标类；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss
    return torch.mean(loss)


def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """
    稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    :param y_true:
    :param y_pred:
    :param mask_zero:
    :return:
    """
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)

    if mask_zero:
        infs = zeros + 1e12
        y_pred = torch.cat([y_pred, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=1e-07, max=1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = pos_loss + neg_loss
    return loss.sum(dim=1).mean()


def global_pointer_crossentropy(y_pred, y_true, sparse=False, mask_zero=False):
    """给GlobalPointer设计的交叉熵
    """
    shape = y_pred.size()
    if not sparse:
        y_true = y_true.reshape(shape[0] * shape[1], -1)
        y_pred = y_pred.reshape(shape[0] * shape[1], -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
    else:
        y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
        y_pred = torch.reshape(y_pred, (shape[0], -1, torch.prod(torch.tensor(shape[2:])).item()))
        loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero)
    return loss


class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    """
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = func.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = func.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class DiceLoss(nn.Module):
    """
    DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """
    def __init__(self, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        :param inputs: [N, C]
        :param target: [N, ]
        """
        prob = torch.softmax(inputs, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        if self.reduction == "mean":
            dice_loss = dsc_i.mean()
        elif self.reduction == "max":
            dice_loss = dsc_i.max()
        else:
            dice_loss = dsc_i
        return dice_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = func.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1-self.eps) * func.nll_loss(log_preds, target, reduction=self.reduction,
                                                                  ignore_index=self.ignore_index)


class LabelSmoothingBinaryCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingBinaryCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = output
        target = target.to(torch.float)
        if self.reduction == 'sum':
            loss = -log_preds.sum().to(torch.float)
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean().to(torch.float)
        loss_ = loss * self.eps / c + (1 - self.eps) * func.binary_cross_entropy(log_preds, target,
                                                                                 reduction=self.reduction)
        return loss_


def compute_kl_loss(p, q, pad_mask=None, merge_mode="sum"):
    """
    计算kl损失，适用于RDrop方法中
    https://kexue.fm/archives/8496
    :param p:
    :param q:
    :param pad_mask:
    :param merge_mode:
    :return:
    """
    p_loss = func.kl_div(func.log_softmax(p, dim=-1), func.softmax(q, dim=-1), reduction='none')
    q_loss = func.kl_div(func.log_softmax(q, dim=-1), func.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        if pad_mask.shape != p.shape:
            pad_mask = pad_mask.unsqueeze(-1)
        p_loss = p_loss * pad_mask
        q_loss = q_loss * pad_mask

    # You can choose whether to use function "sum" and "mean" depending on your task
    if merge_mode == "sum":
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
    elif merge_mode == "mean":
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

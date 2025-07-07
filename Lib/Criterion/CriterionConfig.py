# -*- coding:utf-8  -*-
"""
用于配置所使用的损失函数
Time: 2025/8/4 10:39
Author: zhuangqingyuan
Software: vscode
"""
import cv2
import numpy as np
import torch
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import torchmetrics
from torch import nn
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits,mse_loss
from torch.nn import functional as F, MSELoss
import torchmetrics.functional as F1
from torch.autograd import Variable
def CriterionSet(loss='cross_entropy'):
    """
    @param loss:
    @return:
    """
    if loss == 'cross_entropy':
        return CrossEntropyLoss()
    elif loss == 'bce_logits_loss':
        return BCEWithLogitsLoss()
    elif loss =='softmax_mse_loss':
        return SoftmaxMseLoss()
    elif loss =='softmax_kl_loss':
        return SoftmaxKlLoss()
    elif loss == 'cross_entropy_ignore':
        return CrossEntropyLossIgnore()
    elif loss == 'cross_entropy_reduction':
        return CrossEntropyLossReduction()
    elif loss == 'entropy_map_loss':
        return EntropyMapLoss()
    elif loss == 'mse_loss_default':
        return MseLoss()
    elif loss == 'kl_loss':
        return KLDivLoss()
    elif loss == 'bce_loss':
        return Bcewithlogitsloss2d()
    elif loss == 'proto_learning_loss':
        return ProtoLearningLoss()
    elif loss == 'contrast_loss':
        return ContrastLoss()
    elif loss == 'proto_consistency_loss':
        return ProtoConsistencyLoss()
    else:
        raise KeyError('没有该损失函数:{}'.format(loss))

class ProtoConsistencyLoss(nn.Module):
    def __call__(self, student_proto, teacher_proto,temperature=0.1):
        s_proto = F.normalize(student_proto, p=2, dim=1)
        t_proto = F.normalize(teacher_proto, p=2, dim=1)
        sim_matrix = torch.mm(s_proto, t_proto.t()) / temperature
        targets = torch.arange(s_proto.size(0)).to(s_proto.device)
        loss = F.cross_entropy(sim_matrix, targets) + F.cross_entropy(sim_matrix.t(), targets)
        return loss


class  ContrastLoss(nn.Module):
    def __call__(self, features, labels,prototype,ignore_label,temperature=0.1,topk_ratio=0.2,eps = 1e-8):
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # 展平特征 [B*H*W, C]
        labels_flat = labels.view(-1)  # 展平标签 [B*H*W]
        valid_mask = (labels_flat != ignore_label) & \
                     (labels_flat >= 0) & \
                     (labels_flat < prototype.shape[0])
        features = features[valid_mask]  # [N_valid, C]
        labels_valid = labels_flat[valid_mask]  # [N_valid]
        if features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        assert labels_valid.min() >= 0, f"发现非法负标签 {labels_valid.min().item()}"
        assert labels_valid.max() < prototype.shape[0], f"标签值{labels_valid.max().item()}超出类别数{prototype.shape[0]}"
        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(prototype, p=2, dim=1)
        # 计算相似度矩阵
        sim = torch.mm(features, prototypes.t()) / temperature  # [N_valid, C]
        # 构建正负样本对
        pos_mask = F.one_hot(labels_valid, num_classes=prototypes.shape[0]).float()
        neg_mask = 1 - pos_mask
        # 正样本相似度
        pos_sim = (sim * pos_mask).sum(dim=1)  # [N_valid]
        # 困难负样本挖掘
        with torch.no_grad():
            neg_counts = neg_mask.sum(dim=1)  # [N_valid]
            k = max(1, int(neg_counts.float().mean().item() * topk_ratio))
        neg_sim = sim * neg_mask
        topk_neg = neg_sim.topk(k, dim=1).values  # [N_valid, k]
        # 对比损失计算
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.exp(topk_neg).sum(dim=1)
        loss = -torch.log(numerator / (denominator + eps)).mean()
        return loss


class ProtoLearningLoss(nn.Module):
    def __call__(self, contrast_logits, contrast_target, ignore_label):
        loss_ppc = F.cross_entropy(contrast_logits/0.1, contrast_target.long(), ignore_index=ignore_label)
        contrast_logits = contrast_logits[contrast_target != ignore_label, :]
        contrast_target = contrast_target[contrast_target != ignore_label]
        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()
        return 0.01 * loss_ppc + 0.001 * loss_ppd


class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss

class Bcewithlogitsloss2d:
    def __call__(self, inputs, target,device):
        criterion = BCEWithLogitsLoss2d()
        loss = criterion(inputs, target)
        return loss

class CrossEntropyLoss:
    def __call__(self, inputs, target,device):
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(inputs, target)
        return loss
class KLDivLoss:
    def __call__(self, inputs, targets):
        criterion = nn.KLDivLoss(reduction='none')
        loss = criterion(inputs,targets)
        return loss

class CrossEntropyLossIgnore:
    def __call__(self, inputs, target,ignore_index,device):
        criterion = nn.CrossEntropyLoss(ignore_index = ignore_index).to(device)
        loss = criterion(inputs, target)
        return loss

class CrossEntropyLossReduction:
    def __call__(self, inputs, target,reduction,device):
        criterion = nn.CrossEntropyLoss(reduction = reduction).to(device)
        loss = criterion(inputs, target)
        return loss

class MseLoss:
    def __call__(self, inputs, targets,device):
        criterion = torch.nn.MSELoss()
        loss = criterion(inputs,targets)
        return loss

class EntropyMapLoss:
    def __call__(self, inputs, targets,ignore_label,device):
        # mask = (targets != ignore_label)
        # criterion = torch.nn.MSELoss(reduction='none')
        # entropy_map_loss = criterion(inputs,targets)
        # masked_loss = entropy_map_loss * mask.float()
        # valid_pixel_count = mask.sum()
        # if valid_pixel_count > 0:
        #     entropy_map_loss = masked_loss.sum() / valid_pixel_count
        # else:
        #     entropy_map_loss = torch.tensor(0.0, device=inputs.device)
        # return entropy_map_loss
        criterion = torch.nn.MSELoss()
        entropy_map_loss = criterion(inputs,targets).to(device)
        return entropy_map_loss

class BCEWithLogitsLoss:
    def __call__(self, inputs, target):
        losses = {}
        for name, x in inputs.items():
            xh, xw = x.size(2), x.size(3)
            h, w = target.size(2), target.size(3)

            if xh != h or xw != w:
                x = F.interpolate(
                    input=x, size=(h, w),
                    mode='bilinear', align_corners=True
                )

            losses[name] = binary_cross_entropy_with_logits(x, target)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']

class SoftmaxKlLoss:
    def __call__(self, inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
        """

        @param inputs: 输入
        @param targets: 目标值
        @param conf_mask: 是否启用置信度的计算方法
        @param threshold:置信度
        @param use_softmax:kl是概率的损失函数必须使用softmax对输出结果进行处理
        @return:
        """
        assert inputs.size() == targets.size()
        input_log_softmax = F.log_softmax(inputs, dim=1)
        if use_softmax:
            targets = F.softmax(targets, dim=1)
        if conf_mask:
            loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
            mask = (targets.max(1)[0] > threshold)
            loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
            if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
            return loss_mat.sum() / mask.shape.numel()
        else:
            return F.kl_div(input_log_softmax, targets, reduction='mean')

class SoftmaxMseLoss:
    def __call__(self, inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
        """
        @param inputs: 输入
        @param targets: 目标值
        @param conf_mask: 是否启用置信度的LOSS计算
        @param threshold: 阈值可以设置为0.5
        @param use_softmax: mse必须使用softmax
        @return:
        """
        assert inputs.requires_grad == True and targets.requires_grad == False
        assert inputs.size() == targets.size()  # (batch_size * num_classes * H * W)
        inputs = F.softmax(inputs, dim=1)
        if use_softmax:
            targets = F.softmax(targets, dim=1)

        if conf_mask:
            loss_mat = F.mse_loss(inputs, targets, reduction='none')
            mask = (targets.max(1)[0] > threshold)
            loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
            if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
            return loss_mat.mean()
        else:
            return F.mse_loss(inputs, targets, reduction='mean')


if __name__ == '__main__':
    torch.manual_seed(0)
    predict = OrderedDict()
    predict['out'] = torch.randn((1, 3, 3, 3), dtype=torch.float32)
    mask = torch.zeros((1, 3, 3), dtype=torch.long)
    loss = CrossEntropyLoss()
    print(loss(predict, mask))

#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import NLLLoss
from torch.nn.modules.module import Module


class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self, ignore_index=255):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=ignore_index)
        self.criterion_no_reduce = nn.NLLLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        probs = self.softmax(out)
        loss = self.criterion(probs, label)
        # Handle if the loss has any NaNs by clamping and recomputing
        if torch.isnan(loss):
            return self.criterion_no_reduce(probs, label).mean()
        return loss


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert output.size() == label.size()
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero))
        )

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class BinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert output.size() == label.size()

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero))
        )

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.
    """

    def __init__(self, loss="l1"):
        super(DepthLoss, self).__init__()
        if loss == "l1":
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError(
                "Loss {} currently not supported in DepthLoss".format(loss)
            )

    def forward(self, out, label):
        mask = label != 255
        return self.loss(
            torch.masked_select(out, mask), torch.masked_select(label, mask)
        )


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """

    def __init__(self, size_average=True, normalize=False, norm=1):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            # print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            # print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = label != ignore_label
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(
                torch.masked_select(out_norm, mask),
                torch.masked_select(label, mask),
                reduction="sum",
            )
        else:
            loss = self.loss_func(
                torch.masked_select(out, mask),
                torch.masked_select(label, mask),
                reduction="sum",
            )

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss


class SingleTaskLoss(nn.Module):
    def __init__(self, loss_ft, task):
        super(SingleTaskLoss, self).__init__()
        self.loss_ft = loss_ft
        self.task = task

    def forward(self, pred, gt):
        out = {self.task: self.loss_ft(pred[self.task], gt[self.task])}
        out["total"] = out[self.task]
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert set(loss_weights.keys()) == set(loss_ft.keys())
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in pred.keys()}
        out["total"] = torch.sum(
            torch.stack([self.loss_weights[t] * out[t] for t in pred.keys()])
        )
        return out["total"], out


class PADNetLoss(nn.Module):
    def __init__(
        self,
        tasks: list,
        auxilary_tasks: list,
        loss_ft: nn.ModuleDict,
        loss_weights: dict,
    ):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        total = 0.0
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_ = F.interpolate(
                pred["initial_%s" % (task)], img_size, mode="bilinear"
            )
            gt_ = gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out["deepsup_%s" % (task)] = loss_
            total += self.loss_weights[task] * loss_

        # Losses at output
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_

        out["total"] = total

        return out["total"], out


class CelebABinaryAttributeLoss(Module):

    """

    BCEWithLogitsLoss for CelebA binary attribute classification.

    Expected:

        output: logits of shape [B]

        label: binary targets of shape [B]

    """

    def __init__(self):

        super(CelebABinaryAttributeLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, output, label):
        return self.criterion(output, label.float())
"""
    Loss functions
"""


def get_loss(task_cfg, task=None, config={"DATA": {}}):
    """Return loss function for a specific task"""

    if config.get("DATA", {}).get("DATASET", None) == "CelebA":
        return CelebABinaryAttributeLoss()
        
    if task == "edge":
        criterion = BalancedCrossEntropyLoss(
            size_average=True, pos_weight=task_cfg.get("edge_w", 0.95)
        )

    elif task == "semseg" or task == "human_parts":
        criterion = SoftMaxwithLoss(ignore_index=255)

    elif task == "normals":
        criterion = NormalsLoss(normalize=True, size_average=True, norm=1)

    elif task == "sal":
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == "depth":
        criterion = DepthLoss("l1")
    elif task in ["task_L", "task_R"]:
        criterion = NLLLoss()
    else:
        raise NotImplementedError(
            "Undefined Loss: Choose a task among "
            "edge, semseg, human_parts, sal, depth, or normals"
        )

    return criterion

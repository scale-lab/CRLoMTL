# Adapted from https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch (MTI-Net)
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SABlock(nn.Module):
    """Spatial self-attention block"""

    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


class MultiTaskDistillationModule(nn.Module):
    """
    Perform Multi-Task Distillation
    We apply an attention mask to features from other tasks and
    add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels, passthrough):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        self.passthrough = passthrough

        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict(
                {a: SABlock(channels, channels) for a in other_tasks}
            )
        self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        if self.passthrough:
            out = {t: x["features_%s" % (t)] for t in self.tasks}
        else:
            adapters = {
                t: {
                    a: self.self_attention[t][a](x["features_%s" % (a)])
                    for a in self.auxilary_tasks
                    if a != t
                }
                for t in self.tasks
            }
            out = {
                t: x["features_%s" % (t)]
                + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0)
                for t in self.tasks
            }
        return out


class SEBlock(nn.Module):
    """Squeeze-and-excitation block"""

    def __init__(self, channels, r=16, passthrough=False):
        super(SEBlock, self).__init__()
        self.r = r
        self.passthrough = passthrough
        self.squeeze = nn.Sequential(
            nn.Linear(channels, channels // self.r),
            nn.ReLU(),
            nn.Linear(channels // self.r, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.passthrough:
            return x
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2, 3))).view(B, C, 1, 1)
        return torch.mul(x, squeeze)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiScaleInitialTaskPredictionModule(nn.Module):
    """Module to make the inital task predictions"""

    def __init__(self, num_outputs, auxilary_tasks, input_channels, task_channels):
        super(MultiScaleInitialTaskPredictionModule, self).__init__()
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict(
                {
                    task: nn.Sequential(
                        BasicBlock(channels, channels), BasicBlock(channels, channels)
                    )
                    for task in self.auxilary_tasks
                }
            )

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(
                    nn.Conv2d(input_channels, task_channels, 1, bias=False),
                    nn.BatchNorm2d(task_channels),
                )
                refinement[t] = nn.Sequential(
                    BasicBlock(input_channels, task_channels, downsample=downsample),
                    BasicBlock(task_channels, task_channels),
                )
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict(
            {
                task: nn.Conv2d(task_channels, num_outputs[task], 1)
                for task in self.auxilary_tasks
            }
        )

    def forward(self, features_curr_scale, features_prev_scales=[]):
        if (
            len(features_prev_scales) > 0
        ):  # Concat features that were propagated from previous scale
            x = {}
            for t in self.auxilary_tasks:
                temp = features_curr_scale
                for idx, f_p_s in enumerate(features_prev_scales):
                    scale = features_curr_scale.shape[-1] // f_p_s[t].shape[-1]
                    temp = torch.cat(
                        (
                            temp,
                            F.interpolate(
                                f_p_s[t], scale_factor=scale, mode="bilinear"
                            ),
                        ),
                        1,
                    )
                x[t] = temp
        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out["features_%s" % (t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out["features_%s" % (t)])

        return out


class InitialTaskPredictionModule(nn.Module):
    """Module to make the inital task predictions"""

    def __init__(self, num_outputs, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPredictionModule, self).__init__()
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict(
                {
                    task: nn.Sequential(
                        BasicBlock(channels, channels), BasicBlock(channels, channels)
                    )
                    for task in self.auxilary_tasks
                }
            )

        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(
                    nn.Conv2d(input_channels, task_channels, 1, bias=False),
                    nn.BatchNorm2d(task_channels),
                )
                refinement[t] = nn.Sequential(
                    BasicBlock(input_channels, task_channels, downsample=downsample),
                    BasicBlock(task_channels, task_channels),
                )
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict(
            {
                task: nn.Conv2d(task_channels, num_outputs[task], 1)
                for task in self.auxilary_tasks
            }
        )

    def forward(self, features_curr_scale, features_prev_scale=None):
        if (
            features_prev_scale is not None
        ):  # Concat features that were propagated from previous scale
            x = {
                t: torch.cat(
                    (
                        features_curr_scale,
                        F.interpolate(
                            features_prev_scale[t], scale_factor=2, mode="bilinear"
                        ),
                    ),
                    1,
                )
                for t in self.auxilary_tasks
            }

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out["features_%s" % (t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out["features_%s" % (t)])

        return out


class FPM(nn.Module):
    """Feature Propagation Module"""

    def __init__(self, auxilary_tasks, per_task_channels, passthrough, se_pathrough):
        super(FPM, self).__init__()
        # General
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N * per_task_channels)
        self.passthrough = passthrough

        # Non-linear function f
        downsample = nn.Sequential(
            nn.Conv2d(self.shared_channels, self.shared_channels // 4, 1, bias=False),
            nn.BatchNorm2d(self.shared_channels // 4),
        )
        self.non_linear = nn.Sequential(
            BasicBlock(
                self.shared_channels, self.shared_channels // 4, downsample=downsample
            ),
            BasicBlock(self.shared_channels // 4, self.shared_channels // 4),
            nn.Conv2d(self.shared_channels // 4, self.shared_channels, 1),
        )

        # Dimensionality reduction
        downsample = nn.Sequential(
            nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
            nn.BatchNorm2d(self.per_task_channels),
        )
        self.dimensionality_reduction = BasicBlock(
            self.shared_channels, self.per_task_channels, downsample=downsample
        )

        # SEBlock
        self.se = nn.ModuleDict(
            {
                task: SEBlock(self.per_task_channels, passthrough=se_pathrough)
                for task in self.auxilary_tasks
            }
        )

    def forward(self, x):
        # Get shared representation
        concat = torch.cat(
            [x["features_%s" % (task)] for task in self.auxilary_tasks], 1
        )
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        # Per task attention mask
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)
        shared = torch.mul(mask, concat.view(B, C // self.N, self.N, H, W)).view(
            B, -1, H, W
        )

        # Perform dimensionality reduction
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        out = {}
        for task in self.auxilary_tasks:
            if self.passthrough:
                out[task] = x["features_%s" % (task)]
            else:
                out[task] = self.se[task](shared) + x["features_%s" % (task)]

        return out


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Sigmoid(nn.Module):
    def __init__(self, *args):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class Softmax(nn.Module):
    def __init__(self, *args):
        super(Softmax, self).__init__()

    def forward(self, x):
        return torch.softmax(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MTINetDistilScale(nn.Module):
    def __init__(
        self,
        tasks,
        dims,
        num_outputs,
        input_res,
        channels,
        window_size,
        fpm_passthrough=True,
        dist_passthrough=True,
        se_passthrough=True,
        se_reduction=16,
        se_bias=False,
        se_act=nn.ReLU(inplace=True),
        se_gate_act=nn.Sigmoid(),
    ):
        super(MTINetDistilScale, self).__init__()
        self.tasks = tasks
        self.dims = dims
        self.num_outputs = num_outputs
        self.channels = channels
        self.input_res = input_res

        self.fpm_scale_3 = FPM(
            self.tasks,
            self.channels[3],
            passthrough=fpm_passthrough,
            se_pathrough=se_passthrough,
        )
        self.fpm_scale_2 = FPM(
            self.tasks,
            self.channels[2],
            passthrough=fpm_passthrough,
            se_pathrough=se_passthrough,
        )
        self.fpm_scale_1 = FPM(
            self.tasks,
            self.channels[1],
            passthrough=fpm_passthrough,
            se_pathrough=se_passthrough,
        )

        # Initial task predictions at multiple scales
        self.scale_0 = MultiScaleInitialTaskPredictionModule(
            self.num_outputs,
            self.tasks,
            self.channels[0] + self.channels[1] + self.channels[2] + self.channels[3],
            self.channels[0],
        )
        self.scale_1 = MultiScaleInitialTaskPredictionModule(
            self.num_outputs,
            self.tasks,
            self.channels[1] + self.channels[2] + self.channels[3],
            self.channels[1],
        )
        self.scale_2 = MultiScaleInitialTaskPredictionModule(
            self.num_outputs,
            self.tasks,
            self.channels[2] + self.channels[3],
            self.channels[2],
        )
        self.scale_3 = MultiScaleInitialTaskPredictionModule(
            self.num_outputs, self.tasks, self.channels[3], self.channels[3]
        )

        # Distillation at multiple scales
        self.distillation_scale_0 = MultiTaskDistillationModule(
            self.tasks, self.tasks, self.channels[0], passthrough=dist_passthrough
        )
        self.distillation_scale_1 = MultiTaskDistillationModule(
            self.tasks, self.tasks, self.channels[1], passthrough=dist_passthrough
        )
        self.distillation_scale_2 = MultiTaskDistillationModule(
            self.tasks, self.tasks, self.channels[2], passthrough=dist_passthrough
        )
        self.distillation_scale_3 = MultiTaskDistillationModule(
            self.tasks, self.tasks, self.channels[3], passthrough=dist_passthrough
        )

    def forward(self, x):
        [s_0, s_1, s_2, s_3] = x

        x_3 = self.scale_3(s_3)
        x_3_fpm = self.fpm_scale_3(x_3)
        # Scale 2
        x_2 = self.scale_2(s_2, [x_3_fpm])
        x_2_fpm = self.fpm_scale_2(x_2)
        # Scale 1

        x_1 = self.scale_1(s_1, [x_2_fpm, x_3_fpm])
        x_1_fpm = self.fpm_scale_1(x_1)
        # Scale 0

        x_0 = self.scale_0(s_0, [x_1_fpm, x_2_fpm, x_3_fpm])

        # Distillation + Output
        features_0 = self.distillation_scale_0(x_0)

        features_1 = self.distillation_scale_1(x_1)

        features_2 = self.distillation_scale_2(x_2)
        features_3 = self.distillation_scale_3(x_3)
        multi_scale_features = {
            t: [features_0[t], features_1[t], features_2[t], features_3[t]]
            for t in self.tasks
        }

        multi_scale_initial_predictions = {
            t: [x_0[t], x_1[t], x_2[t], x_3[t]] for t in self.tasks
        }
        return multi_scale_features, multi_scale_initial_predictions

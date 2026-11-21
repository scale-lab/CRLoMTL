import math
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from crlomtl.resolver import (
    coeff_resolver,
    coeff_soft_resolver,
    kernel_resolver,
    coeff_soft_noisy_resolver,
    matrix_resolver,
    conv_resolver,
    mean_resolver,
    projection_resolver,
    projection_hold_resolver,
    svd_mean_resolver,
    svd_resolver_v1,
    svd_resolver_v2,
    rbf_v1_resolver,
    moe_resolver,
    shared_subspace_resolver,
    MoEFusion,
    soft_coeff_then_shared_subspace_svd_resolver,
    soft_coeff_then_shared_subspace_lowrank_svd_resolver,
    shared_subspace_B_resolver,
    shared_subspace_normalized_resolver
)

resolvers = {
    "mean": mean_resolver,
    "coeff": coeff_resolver,
    "coeff_soft": coeff_soft_resolver,
    "coeff_soft_noisy": coeff_soft_noisy_resolver,
    "projection": projection_resolver,
    "projection_hold": projection_hold_resolver,
    "svd_v1": svd_resolver_v1,
    "svd_v2": svd_resolver_v2,
    "svd_mean": svd_mean_resolver,
    "matrix": matrix_resolver,
    "conv": conv_resolver,
    "rbf_v1": rbf_v1_resolver,
    "kernel": kernel_resolver,
    "shared_subspace": shared_subspace_resolver,
    "moe": moe_resolver,
    "soft_coeff_then_svd": soft_coeff_then_shared_subspace_svd_resolver,
    "soft_coeff_then_lowrank_svd": soft_coeff_then_shared_subspace_lowrank_svd_resolver,
    "shared_subspace_b": shared_subspace_B_resolver,
    "shared_subspace_norm": shared_subspace_normalized_resolver
}


class FusionLayer(nn.Module):
    def __init__(self, num_tensors, num_features, feature_length):
        super(FusionLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=num_tensors * num_features,
            out_channels=num_features,
            kernel_size=1,
        )
        self.num_features = num_features
        self.feature_length = feature_length

    def forward(self, x):
        x = x.reshape(1, -1, self.feature_length)
        x = self.conv(x)
        x = x.view(self.num_features, self.feature_length)
        return x


class MetaNet(torch.nn.Module):
    def __init__(self, input_dim, num_tasks):
        super().__init__()
        self.lora_meta_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_tasks),
        )

    def forward(self, x):
        if x.dim() > 2:
            x_pooled = x.mean(dim=1)
        else:
            x_pooled = x

        return F.softmax(self.lora_meta_net(x_pooled), dim=1)


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        task_r: Optional[Dict[str, int]] = {},
        task_alpha: Optional[Dict[str, int]] = {},
        task_dropout: Optional[Dict[str, float]] = {},
        tasks: Optional[List[str]] = None,
        resolver: Optional[str] = "mean",
        full_rank_resolver: bool = False,
        noise_factor: float = 0.2,
        softmax_temperature: float = 1.0,
        atten_emb_dim: int = 64,
        kernel_type: str = 'rbf',
        shared_rank_ratio: Optional[float] = None,
    ):
        self.r = r
        self.tasks = tasks if tasks is not None and len(tasks) > 0 else None
        self.has_task_specialization = self.tasks is not None and len(
            self.tasks) > 0
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.task_r = task_r or {}
        self.task_alpha = task_alpha or {}
        self.task_dropout = task_dropout
        # Use default r alpha and dropout if not provided for the given task
        for task in self.tasks:
            if task not in self.task_r:
                self.task_r[task] = r
            if task not in self.task_alpha:
                self.task_alpha[task] = lora_alpha
            if task not in self.task_dropout:
                self.task_dropout[task] = lora_dropout
        self.merged = False
        self.merge_weights = merge_weights
        self.resolver = resolver
        self.resolver_fn = partial(resolvers[resolver], full_rank_resolver)
        if shared_rank_ratio is not None:
            assert resolver.startswith("shared_subspace"), "SHARED_RANK_RATIO needs a shared_subspace resolver"
            self.resolver_fn = partial(self.resolver_fn, rank_ratio=shared_rank_ratio)
        self.full_rank_resolver = full_rank_resolver
        self.noise_factor = noise_factor
        self.softmax_temperature = softmax_temperature
        self.atten_emb_dim = atten_emb_dim
        self.kernel_type = kernel_type

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        task_r: Optional[Dict[str, int]] = {},
        task_alpha: Optional[Dict[str, int]] = {},
        task_dropout: Optional[Dict[str, float]] = {},
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        tasks: Optional[List[str]] = None,
        resolver: Optional[str] = "mean",
        full_rank_resolver: bool = False,
        noise_factor: float = 0.2,
        softmax_temperature: float = 1.0,
        atten_emb_dim: int = 64,
        kernel_type: str = 'rbf',
        shared_rank_ratio: Optional[float] = None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            tasks=tasks,
            task_r=task_r,
            task_alpha=task_alpha,
            task_dropout=task_dropout,
            resolver=resolver,
            full_rank_resolver=full_rank_resolver,
            noise_factor=noise_factor,
            softmax_temperature=softmax_temperature,
            atten_emb_dim=atten_emb_dim,
            kernel_type=kernel_type,
            shared_rank_ratio=shared_rank_ratio,
        )
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            if not self.has_task_specialization:
                self.lora_A = nn.Parameter(
                    self.weight.new_zeros((r, in_features)))
                self.lora_B = nn.Parameter(
                    self.weight.new_zeros((out_features, r)))
            else:
                self.lora_A = nn.ParameterDict(
                    {
                        task: nn.Parameter(
                            self.weight.new_zeros((task_r[task], in_features))
                        )
                        for task in tasks
                    }
                )
                self.lora_B = nn.ParameterDict(
                    {
                        task: nn.Parameter(
                            self.weight.new_zeros((out_features, task_r[task]))
                        )
                        for task in tasks
                    }
                )
                if "project" in resolver:
                    if not self.full_rank_resolver:
                        self.lora_proj_A = nn.Parameter(
                            self.weight.new_zeros((r, in_features))
                        )
                        self.lora_proj_B = nn.Parameter(
                            self.weight.new_zeros((out_features, r))
                        )
                    else:
                        self.lora_proj_A = nn.Parameter(
                            self.weight.new_zeros((1, out_features))
                        )
                        self.lora_proj_B = None
                elif "matrix" in resolver:
                    assert (
                        not self.full_rank_resolver
                    ), "Matrix resolver does not support full rank"
                    self.lora_proj_A = nn.Parameter(
                        self.weight.new_zeros((r, in_features))
                    )
                    self.lora_proj_B = nn.Parameter(
                        self.weight.new_zeros((out_features, r))
                    )
                elif "conv" in resolver:
                    assert (
                        not self.full_rank_resolver
                    ), "Matrix resolver does not support full rank"
                    self.lora_proj_A = FusionLayer(
                        num_tensors=len(tasks),
                        num_features=r,
                        feature_length=in_features,
                    )
                    self.lora_proj_B = FusionLayer(
                        num_tensors=len(tasks),
                        num_features=r,
                        feature_length=out_features,
                    )
                elif 'rbf_v1' in resolver:
                    embed_dim = 16
                    self.lora_proj_A = nn.Parameter(
                        self.weight.new_zeros(len(tasks), embed_dim))
                    self.lora_proj_B = nn.Parameter(
                        self.weight.new_zeros(embed_dim))
                elif 'moe' in resolver:
                    assert (
                        not self.full_rank_resolver
                    ), "MoE resolver does not support full rank"
                    self.moe_module = MoEFusion(in_features, len(
                        tasks), noise_factor=self.noise_factor)
                    self.lora_proj_A = None
                    self.lora_proj_B = None

                else:
                    self.lora_proj_A = None
                    self.lora_proj_B = None

                if resolver in ["coeff", "coeff_soft", "coeff_soft_noisy", "kernel", "attention", "sampling", "soft_coeff_then_svd","soft_coeff_then_lowrank_svd"]:
                    if not self.full_rank_resolver:
                        self.lora_coeff_A = nn.ParameterDict(
                            {task: nn.Parameter(torch.zeros(1))
                             for task in tasks}
                        )
                        self.lora_coeff_B = nn.ParameterDict(
                            {task: nn.Parameter(torch.zeros(1))
                             for task in tasks}
                        )
                    else:
                        self.lora_coeff_A = nn.ParameterDict(
                            {task: nn.Parameter(torch.zeros(1))
                             for task in tasks}
                        )
                        self.lora_coeff_B = None
                else:
                    self.lora_coeff_A = None
                    self.lora_coeff_B = None

                if "meta" in resolver:
                    self.meta_net = MetaNet(in_features, len(tasks))
                    self.resolver_fn = partial(
                        self.resolver_fn, meta_net=self.meta_net)
                else:
                    self.meta_net = None

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            if not self.has_task_specialization:
                # initialize B the same way as the default for nn.Linear and A to zero
                # this is different than what is described in the paper but should not affect performance
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            else:
                for task in self.tasks:
                    nn.init.kaiming_uniform_(self.lora_A[task], a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[task])
                if self.lora_proj_A is not None:
                    if "matrix" in self.resolver:
                        nn.init.kaiming_uniform_(
                            self.lora_proj_A, a=math.sqrt(5))
                        nn.init.zeros_(self.lora_proj_B)
                    elif 'rbf_v1' in self.resolver:
                        nn.init.kaiming_uniform_(
                            self.lora_proj_A, a=math.sqrt(5))
                        nn.init.uniform_(self.lora_proj_B)
                    elif "project" in self.resolver:
                        nn.init.orthogonal_(self.lora_proj_A)
                        # nn.init.orthogonal_(self.lora_proj_B)
                        nn.init.zeros_(self.lora_proj_B)
                    elif "conv" in self.resolver:
                        pass
                    else:
                        raise ValueError(
                            "Projection or matrix resolver must be used with projection matrices"
                        )
                if self.lora_coeff_A is not None:
                    for task in self.tasks:
                        nn.init.uniform_(self.lora_coeff_A[task])
                        if self.lora_coeff_B is not None:
                            nn.init.uniform_(self.lora_coeff_B[task])

    @torch.no_grad()
    def get_merged_lora_update(self):
        """
        Return the LoRA/CRLoMTL fused update matrix with shape
        [out_features, in_features].

        Standard LoRA:
            delta = lora_B @ lora_A

        CRLoMTL task-specialized LoRA:
            delta = resolver_fn(lora_A, lora_B, tasks, ...)
        """
        if self.r <= 0:
            return None

        if not self.has_task_specialization:
            return self.lora_B @ self.lora_A

        resolver_args = {
            'x': None,
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
            'tasks': self.tasks,
            'direction_A': self.lora_proj_A,
            'direction_B': self.lora_proj_B,
            'coeff_A': self.lora_coeff_A,
            'coeff_B': self.lora_coeff_B,
            'training': False,
            'noise_factor': self.noise_factor,
            'temperature': self.softmax_temperature,
            'atten_emb_dim': self.atten_emb_dim,
            'kernel_type': self.kernel_type,
        }

        # Add MoE module if available
        if hasattr(self, 'moe_module') and self.moe_module is not None:
            resolver_args['moe_module'] = self.moe_module

        return self.resolver_fn(**resolver_args)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)

        if self.r <= 0:
            return self

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                delta = self.get_merged_lora_update()
                if delta is not None:
                    self.weight.data -= T(delta).to(self.weight.dtype) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                delta = self.get_merged_lora_update()
                if delta is not None:
                    self.weight.data += T(delta).to(self.weight.dtype) * self.scaling
                self.merged = True

        return self

    def forward(self, x: torch.Tensor, task: Optional[str] = None):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            if self.has_task_specialization:
                if task is not None:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    result += (
                        self.lora_dropout(x)
                        @ self.lora_A[task].transpose(0, 1)
                        @ self.lora_B[task].transpose(0, 1)
                    ) * self.scaling
                    return result
                else:
                    resolver_args = {
                        'x': self.lora_dropout(x),
                        'lora_A': self.lora_A,
                        'lora_B': self.lora_B,
                        'tasks': self.tasks,
                        'direction_A': self.lora_proj_A,
                        'direction_B': self.lora_proj_B,
                        'coeff_A': self.lora_coeff_A,
                        'coeff_B': self.lora_coeff_B,
                        'training': self.training,
                        'noise_factor': self.noise_factor,
                        'temperature': self.softmax_temperature,
                        'atten_emb_dim': self.atten_emb_dim,
                        'kernel_type': self.kernel_type,
                    }

                    # Add MoE module if available
                    if hasattr(self, 'moe_module') and self.moe_module is not None:
                        resolver_args['moe_module'] = self.moe_module
                    combined = self.resolver_fn(**resolver_args)

                    result = F.linear(x, T(self.weight), bias=self.bias)
                    if not self.full_rank_resolver:

                        result += (self.lora_dropout(x) @
                                   combined.t()) * self.scaling
                    else:
                        result += combined * self.scaling
                    return result

            else:
                result = F.linear(x, T(self.weight), bias=self.bias)
                result += (
                    self.lora_dropout(x)
                    @ self.lora_A.transpose(0, 1)
                    @ self.lora_B.transpose(0, 1)
                ) * self.scaling
                return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(
        self,
        conv_module,
        in_channels,
        out_channels,
        kernel_size,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
        tasks=None,
        resolver="mean",
        full_rank_resolver=False,
        task_r={},
        task_alpha={},
        task_dropout={},
        noise_factor=0.2,
        softmax_temperature=1.0,
        atten_emb_dim=64,
        kernel_type='rbf',
        shared_rank_ratio=None,
        **kwargs
    ):
        if resolver in ["rbf_v1", "attention", "meta"]:
            resolver = "coeff_soft"
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(
            in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            task_r=task_r,
            task_alpha=task_alpha,
            task_dropout=task_dropout,
            tasks=tasks,
            resolver=resolver,
            full_rank_resolver=False,
            noise_factor=noise_factor,
            softmax_temperature=softmax_temperature,
            atten_emb_dim=atten_emb_dim,
            kernel_type=kernel_type,
            shared_rank_ratio=shared_rank_ratio,
        )
        assert isinstance(kernel_size, int) or (
            isinstance(kernel_size, tuple)
            and len(kernel_size) == 2
            and kernel_size[0] == kernel_size[1]
            and isinstance(kernel_size[0], int)
        )
        kernel_size = kernel_size if isinstance(
            kernel_size, int) else kernel_size[0]

        # Actual trainable parameters
        if r > 0:
            if not self.has_task_specialization:
                self.lora_A = nn.Parameter(
                    self.conv.weight.new_zeros(
                        (r * kernel_size, in_channels * kernel_size)
                    )
                )
                self.lora_B = nn.Parameter(
                    self.conv.weight.new_zeros(
                        (
                            out_channels // self.conv.groups * kernel_size,
                            r * kernel_size,
                        )
                    )
                )
            else:
                self.lora_A = nn.ParameterDict(
                    {
                        task: nn.Parameter(
                            self.conv.weight.new_zeros(
                                (task_r[task] * kernel_size,
                                 in_channels * kernel_size)
                            )
                        )
                        for task in tasks
                    }
                )
                self.lora_B = nn.ParameterDict(
                    {
                        task: nn.Parameter(
                            self.conv.weight.new_zeros(
                                (
                                    out_channels // self.conv.groups * kernel_size,
                                    task_r[task] * kernel_size,
                                )
                            )
                        )
                        for task in tasks
                    }
                )
                if "project" in resolver:
                    self.lora_proj_A = nn.Parameter(
                        self.conv.weight.new_zeros(
                            (r * kernel_size, in_channels * kernel_size)
                        )
                    )
                    self.lora_proj_B = nn.Parameter(
                        self.conv.weight.new_zeros(
                            (
                                out_channels // self.conv.groups * kernel_size,
                                r * kernel_size,
                            )
                        )
                    )
                elif "matrix" in resolver:
                    assert (
                        not self.full_rank_resolver
                    ), "Matrix resolver does not support full rank"
                    self.lora_proj_A = nn.Parameter(
                        self.conv.weight.new_zeros(
                            (r * kernel_size, in_channels * kernel_size)
                        )
                    )
                    self.lora_proj_B = nn.Parameter(
                        self.conv.weight.new_zeros(
                            (
                                out_channels // self.conv.groups * kernel_size,
                                r * kernel_size,
                            )
                        )
                    )
                elif "conv" in resolver:
                    assert (
                        not self.full_rank_resolver
                    ), "Matrix resolver does not support full rank"
                    self.lora_proj_A = FusionLayer(
                        num_tensors=len(tasks),
                        num_features=r * kernel_size,
                        feature_length=in_channels * kernel_size,
                    )
                    self.lora_proj_B = FusionLayer(
                        num_tensors=len(tasks),
                        num_features=r * kernel_size,
                        feature_length=out_channels // self.conv.groups * kernel_size,
                    )
                elif 'moe' in resolver:
                    assert (
                        not self.full_rank_resolver
                    ), "MoE resolver does not support full rank"
                    self.moe_module = MoEFusion(in_channels, len(
                        tasks), noise_factor=self.noise_factor)
                    self.lora_proj_A = None
                    self.lora_proj_B = None
                else:
                    self.lora_proj_A = None
                    self.lora_proj_B = None
                if resolver in ["coeff", "coeff_soft", "coeff_soft_noisy", "kernel", "attention", "sampling", "soft_coeff_then_svd","soft_coeff_then_lowrank_svd"]:
                    self.lora_coeff_A = nn.ParameterDict(
                        {task: nn.Parameter(torch.zeros(1)) for task in tasks}
                    )
                    self.lora_coeff_B = nn.ParameterDict(
                        {task: nn.Parameter(torch.zeros(1)) for task in tasks}
                    )
                else:
                    self.lora_coeff_A = None
                    self.lora_coeff_B = None
                if "meta" in resolver:
                    self.meta_net = MetaNet(in_channels, len(tasks))
                    self.resolver_fn = partial(
                        self.resolver_fn, meta_net=self.meta_net)
                else:
                    self.meta_net = None
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            if not self.has_task_specialization:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            else:
                for task in self.tasks:
                    nn.init.kaiming_uniform_(self.lora_A[task], a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[task])
                if self.lora_proj_A is not None:
                    if "matrix" in self.resolver:
                        nn.init.kaiming_uniform_(
                            self.lora_proj_A, a=math.sqrt(5))
                        nn.init.zeros_(self.lora_proj_B)
                    elif "project" in self.resolver:
                        nn.init.orthogonal_(self.lora_proj_A)
                        # nn.init.orthogonal_(self.lora_proj_B)
                        nn.init.zeros_(self.lora_proj_B)
                    elif "conv" in self.resolver:
                        pass
                    else:
                        raise ValueError(
                            "Projection or matrix resolver must be used with projection matrices"
                        )
                if self.lora_coeff_A is not None:
                    for task in self.tasks:
                        nn.init.uniform_(self.lora_coeff_A[task])
                        nn.init.uniform_(self.lora_coeff_B[task])

    @torch.no_grad()
    def get_merged_lora_update(self):
        """
        Return the LoRA/CRLoMTL fused update matrix before reshaping
        to self.conv.weight.shape.

        Standard ConvLoRA:
            delta = lora_B @ lora_A

        CRLoMTL task-specialized ConvLoRA:
            delta = resolver_fn(lora_A, lora_B, tasks, ...)
        """
        if self.r <= 0:
            return None

        if not self.has_task_specialization:
            return self.lora_B @ self.lora_A

        resolver_args = {
            'x': None,
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
            'tasks': self.tasks,
            'direction_A': self.lora_proj_A,
            'direction_B': self.lora_proj_B,
            'coeff_A': self.lora_coeff_A,
            'coeff_B': self.lora_coeff_B,
            'training': False,
            'noise_factor': self.noise_factor,
            'temperature': self.softmax_temperature,
            'atten_emb_dim': self.atten_emb_dim,
            'kernel_type': self.kernel_type,
        }

        # Add MoE module if available
        if hasattr(self, 'moe_module') and self.moe_module is not None:
            resolver_args['moe_module'] = self.moe_module

        return self.resolver_fn(**resolver_args)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)

        if self.r <= 0:
            return self

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                delta = self.get_merged_lora_update()
                if delta is not None:
                    self.conv.weight.data -= delta.reshape(
                        self.conv.weight.shape
                    ).to(self.conv.weight.dtype) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                delta = self.get_merged_lora_update()
                if delta is not None:
                    self.conv.weight.data += delta.reshape(
                        self.conv.weight.shape
                    ).to(self.conv.weight.dtype) * self.scaling
                self.merged = True

        return self

    def forward(self, x, task: Optional[str] = None):
        if self.r > 0 and not self.merged:
            if self.has_task_specialization:
                if task is not None:
                    return self.conv._conv_forward(
                        x,
                        self.conv.weight
                        + (self.lora_B[task] @ self.lora_A[task]).view(
                            self.conv.weight.shape
                        )
                        * self.scaling,
                        self.conv.bias,
                    )
                else:
                    resolver_args = {
                        'x': x,
                        'lora_A': self.lora_A,
                        'lora_B': self.lora_B,
                        'tasks': self.tasks,
                        'direction_A': self.lora_proj_A,
                        'direction_B': self.lora_proj_B,
                        'coeff_A': self.lora_coeff_A,
                        'coeff_B': self.lora_coeff_B,
                        'training': self.training,
                        'noise_factor': self.noise_factor,
                        'temperature': self.softmax_temperature,
                        'atten_emb_dim': self.atten_emb_dim,
                        'kernel_type': self.kernel_type,
                    }

                    # Add MoE module if available
                    if hasattr(self, 'moe_module') and self.moe_module is not None:
                        resolver_args['moe_module'] = self.moe_module

                    combined = self.resolver_fn(**resolver_args)
                    return self.conv._conv_forward(
                        x,
                        self.conv.weight
                        + combined.reshape(self.conv.weight.shape) *
                        self.scaling,
                        self.conv.bias,
                    )
            else:
                return self.conv._conv_forward(
                    x,
                    self.conv.weight
                    + (self.lora_B @ self.lora_A).view(self.conv.weight.shape)
                    * self.scaling,
                    self.conv.bias,
                )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin.swin_transformer import SwinTransformer
#from models.swin.swin_transformer_shared_lora import SwinTransformerSharedLoRA
from models.swin.mti import MTINetDistilScale
from models.swin.adapter_registry import (
    get_adapter_name,
    get_backbone_class,
    build_backbone_kwargs,
    adapter_returns_task_stage_dict,
)

def get_head(
    task, backbone_channels, num_outputs, logger, config=None, multiscale=None
):
    """Return the decoder head"""
    head_type = config.MODEL.DECODER_HEAD.get(task, "hrnet")

    if head_type == "hrnet":
        from models.swin.seg_hrnet import HighResolutionHead

        logger.info(
            f"Using hrnet for task {task} with backbone channels {backbone_channels}"
        )
        return HighResolutionHead(backbone_channels, num_outputs)
    else:
        raise ValueError(f"Unknown head type {head_type}")


class DecoderGroup(nn.Module):
    def __init__(
        self, tasks, num_outputs, channels, out_size, logger, config, multiscale=True
    ):
        super(DecoderGroup, self).__init__()
        self.tasks = tasks
        self.num_outputs = num_outputs
        self.channels = channels
        self.decoders = nn.ModuleDict()
        self.out_size = out_size
        self.multiscale = multiscale
        if not self.multiscale:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        for task in self.tasks:
            self.decoders[task] = get_head(
                task,
                self.channels,
                self.num_outputs[task],
                config=config,
                logger=logger,
                multiscale=self.multiscale,
            )

    def forward(self, x):
        result = {
            task: F.interpolate(
                self.decoders[task](x[task]), self.out_size, mode="bilinear"
            )
            for task in self.tasks
            if task in x
        }
        return result


class Downsampler(nn.Module):
    def __init__(self, dims, channels, input_res, bias=False, enabled=True):
        super(Downsampler, self).__init__()
        self.dims = dims
        self.input_res = input_res
        self.enabled = enabled
        if self.enabled:
            self.downsample_0 = torch.nn.Conv2d(
                dims[0], channels[0], 1, bias=bias)
            self.downsample_1 = torch.nn.Conv2d(
                dims[1], channels[1], 1, bias=bias)
            self.downsample_2 = torch.nn.Conv2d(
                dims[2], channels[2], 1, bias=bias)
            self.downsample_3 = torch.nn.Conv2d(
                dims[3], channels[3], 1, bias=bias)

    def forward(self, x):
        s_3 = (
            x[3]
            .view(-1, self.input_res[3], self.input_res[3], self.dims[3])
            .permute(0, 3, 1, 2)
        )

        s_2 = (
            x[2]
            .view(-1, self.input_res[2], self.input_res[2], self.dims[2])
            .permute(0, 3, 1, 2)
        )
        s_1 = (
            x[1]
            .view(-1, self.input_res[1], self.input_res[1], self.dims[1])
            .permute(0, 3, 1, 2)
        )
        s_0 = (
            x[0]
            .view(-1, self.input_res[0], self.input_res[0], self.dims[0])
            .permute(0, 3, 1, 2)
        )

        if self.enabled:
            return [
                self.downsample_0(s_0),
                self.downsample_1(s_1),
                self.downsample_2(s_2),
                self.downsample_3(s_3),
            ]
        else:
            return [s_0, s_1, s_2, s_3]

class SwinMTL(torch.nn.Module):
    def __init__(self, config, logger, lora_config=None, tasks=None):
        super(SwinMTL, self).__init__()

        self.config = config

        if config.FUSED_LAYERNORM:
            try:
                import apex as amp
                layernorm = amp.normalization.FusedLayerNorm
            except:
                layernorm = None
                print("To use FusedLayerNorm, please install apex.")
        else:
            layernorm = nn.LayerNorm

        self.num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT
        self.tasks = config.TASKS
        self.mti = config.MODEL.MTI

        self.adapter_name = get_adapter_name(config)

        Backbone = get_backbone_class(self.adapter_name)

        logger.info(f"Using adapter: {self.adapter_name}")
        logger.info(f"Using backbone class: {Backbone.__name__}")


        backbone_kwargs = build_backbone_kwargs(
            config=config,
            layernorm=layernorm,
            tasks=tasks,
            lora_config=lora_config,
            adapter_name=self.adapter_name,
        )

        self.backbone = Backbone(**backbone_kwargs)

        patches_resolution = self.backbone.patch_embed.patches_resolution
        self.embed_dim = self.backbone.embed_dim
        num_layers = self.backbone.num_layers

        self.dims = [
            int((self.embed_dim * 2 ** ((i + 1) if i < num_layers - 1 else i)))
            for i in range(num_layers)
        ]

        self.input_res = [
            patches_resolution[0] // (2 ** ((i + 1) if i < num_layers - 1 else i))
            for i in range(num_layers)
        ]

        self.window_size = self.backbone.layers[0].blocks[0].window_size
        self.img_size = self.backbone.patch_embed.img_size

        self.channels = (
            config.MODEL.DECODER_CHANNELS
            if config.MODEL.DECODER_DOWNSAMPLER
            else self.dims
        )

        if self.mti:
            self.channels = [int(c / 2) for c in self.channels]

        self.per_task_downsampler = (
            config.MODEL.PER_TASK_DOWNSAMPLER
            and not self.mti
        )

        if config.MODEL.DECODER_DOWNSAMPLER:
            logger.info(f"Decoder channels: {self.channels}")
            logger.info(f"Per task downsampler: {self.per_task_downsampler}")

        if self.per_task_downsampler:
            self.downsampler = nn.ModuleDict(
                {
                    task: Downsampler(
                        dims=self.dims,
                        channels=self.channels,
                        input_res=self.input_res,
                        bias=False,
                        enabled=config.MODEL.DECODER_DOWNSAMPLER,
                    )
                    for task in self.tasks
                }
            )
        else:
            self.downsampler = Downsampler(
                dims=self.dims,
                channels=self.channels,
                input_res=self.input_res,
                bias=False,
                enabled=config.MODEL.DECODER_DOWNSAMPLER,
            )

        self.decoders = DecoderGroup(
            self.tasks,
            self.num_outputs,
            channels=self.channels,
            out_size=self.img_size,
            logger=logger,
            config=config,
            multiscale=config.MODEL.MULTISCALE,
        )

        if self.mti:
            self.mti_net = MTINetDistilScale(
                self.tasks,
                self.dims,
                self.num_outputs,
                self.input_res,
                self.channels,
                self.window_size,
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.backbone, "no_weight_decay"):
            return self.backbone.no_weight_decay()
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.backbone, "no_weight_decay_keywords"):
            return self.backbone.no_weight_decay_keywords()
        return {}

    def shared_parameters(self):
        # All non-task-specific parameters.
        return [
            p
            for name, p in self.named_parameters()
            if not any([task in name for task in self.tasks])
        ][:-2]

    def last_shared_parameters(self):
        last_layer = self.backbone.layers[-1]
        return list(last_layer.parameters())

    def task_specific_parameters(self):
        return [
            p
            for name, p in self.named_parameters()
            if any([task in name for task in self.tasks])
        ]
    
    def task_specific_parameters_grouped(self,tasks=None):
        if tasks is None:
            tasks = self.tasks

        grouped = []
        for task in tasks:
            task_params = [
            p
            for name, p in self.named_parameters()
            if task in name and p.requires_grad]

            grouped.append(task_params)

        return grouped
    def _uses_task_stage_dict(self, representation):
        """Some backbones return [(shared_stage, task_stage_dict), ...]
        instead of [stage0, stage1, stage2, stage3]."""

        return (
            isinstance(representation, (list, tuple))
            and len(representation) > 0
            and isinstance(representation[0], tuple)
            and len(representation[0]) == 2
            and isinstance(representation[0][1], dict)
        )

    def forward(
        self,
        x,
        task=None,
        return_representation=False,
        return_shared_ft=False,
        last_only=False,
    ):
        assert not (
            return_shared_ft and return_representation
        ), "Not implemented"

        active_tasks = self.tasks if task is None else [task]

        shared_representation = self.backbone(
            x,
            return_stages=True,
            task=task,
        )

        uses_task_stage_dict = self._uses_task_stage_dict(
            shared_representation
        )

        if uses_task_stage_dict:
            # [(shared_stage, task_stage_dict), ...] format
            #   shared_representation = [
            #       (shared_stage0, task_stage_dict0),
            #       (shared_stage1, task_stage_dict1),
            #       ...
            #   ]

            shared_stages = []
            task_stages = {t: [] for t in active_tasks}

            for shared_stage, task_stage_dict in shared_representation:
                shared_stages.append(shared_stage)

                for t in active_tasks:
                    if task_stage_dict is not None and t in task_stage_dict:
                        task_stages[t].append(task_stage_dict[t])
                    else:
                        task_stages[t].append(shared_stage)

            if task is not None:
                representation_for_return = task_stages[task]
            else:
                representation_for_return = shared_stages

            if self.per_task_downsampler:
                shared_ft = {
                    t: self.downsampler[t](task_stages[t])
                    for t in active_tasks
                }

            else:
                if self.mti:
                    downsampled_representation = self.downsampler(
                        representation_for_return
                    )

                    shared_ft, _ = self.mti_net(
                        downsampled_representation
                    )

                    representation_for_return = downsampled_representation

                    if task is not None:
                        shared_ft = {
                            task: shared_ft[task]
                        }

                else:
                    shared_ft = {
                        t: self.downsampler(task_stages[t])
                        for t in active_tasks
                    }

        else:
            # one tensor per stage
            #   shared_representation = [stage0, stage1, stage2, stage3]

            representation_for_return = shared_representation

            if self.per_task_downsampler:
                shared_ft = {
                    t: self.downsampler[t](shared_representation)
                    for t in active_tasks
                }

            else:
                if self.mti:
                    downsampled_representation = self.downsampler(
                        shared_representation
                    )

                    shared_ft, _ = self.mti_net(
                        downsampled_representation
                    )

                    representation_for_return = downsampled_representation

                    if task is not None:
                        shared_ft = {
                            task: shared_ft[task]
                        }

                else:
                    downsampled_representation = self.downsampler(
                        shared_representation
                    )

                    shared_ft = {
                        t: downsampled_representation
                        for t in active_tasks
                    }

                    representation_for_return = downsampled_representation

        result = self.decoders(shared_ft)

        if return_representation:
            if not last_only:
                return result, representation_for_return
            else:
                return result, [representation_for_return[-1]]

        if return_shared_ft:
            return result, shared_ft

        return result
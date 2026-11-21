import torch
import torch.nn as nn

from models.swin.adapter_registry import (
    get_adapter_name,
    get_backbone_class,
    build_backbone_kwargs,
)


DEFAULT_CELEBA_GROUPS = {
    "hair": [
        "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Gray_Hair",
        "Receding_Hairline",
        "Straight_Hair",
        "Wavy_Hair",
    ],
    "facial_hair": [
        "5_o_Clock_Shadow",
        "Goatee",
        "Mustache",
        "No_Beard",
        "Sideburns",
    ],
    "expression": [
        "Smiling",
        "Mouth_Slightly_Open",
        "Narrow_Eyes",
    ],
    "makeup": [
        "Arched_Eyebrows",
        "Bags_Under_Eyes",
        "Heavy_Makeup",
        "Rosy_Cheeks",
        "Wearing_Lipstick",
    ],
    "accessories": [
        "Eyeglasses",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Necklace",
        "Wearing_Necktie",
    ],
    "face_shape": [
        "Attractive",
        "Big_Lips",
        "Big_Nose",
        "Chubby",
        "Double_Chin",
        "High_Cheekbones",
        "Oval_Face",
        "Pointy_Nose",
        "Pale_Skin",
    ],
    "identity_style": [
        "Blurry",
        "Bushy_Eyebrows",
        "Male",
        "Young",
    ],
}


class CelebAHeadGroup(nn.Module):
    """Lightweight binary classification heads for CelebA attributes."""

    def __init__(self, tasks, in_features, dropout=0.0):
        super(CelebAHeadGroup, self).__init__()

        self.tasks = list(tasks)
        self.in_features = in_features
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(in_features, 1)
                for task in self.tasks
            }
        )

    def forward(self, x, active_tasks=None):
        if active_tasks is None:
            active_tasks = self.tasks

        x = self.dropout(x)

        return {
            task: self.heads[task](x).squeeze(-1)
            for task in active_tasks
            if task in self.heads
        }


class SwinCelebA(nn.Module):
    def __init__(
        self,
        config,
        logger,
        lora_config=None,
        tasks=None,  # kept for compatibility with build_model.py
        task_groups=None,
        head_dropout=0.0,
    ):
        super(SwinCelebA, self).__init__()

        self.config = config

        if config.FUSED_LAYERNORM:
            try:
                import apex as amp
                layernorm = amp.normalization.FusedLayerNorm
            except Exception:
                layernorm = None
                print("To use FusedLayerNorm, please install apex.")
        else:
            layernorm = nn.LayerNorm

        # ---------------------------------------------------------------------
        # CelebA output task space: 40 binary attributes.
        # These are used by dataset labels, heads, loss, metrics, and global pass.
        # ---------------------------------------------------------------------
        self.output_tasks = list(config.TASKS)
        self.tasks = self.output_tasks
        self.num_outputs = {task: 1 for task in self.output_tasks}

        # ---------------------------------------------------------------------
        # CelebA LoRA task space:
        #   grouped mode     -> 7 cluster adapters
        #   non-grouped mode -> 40 attribute adapters
        # ---------------------------------------------------------------------
        if config.MODEL.CELEBA.USE_GROUPED_LORA:
            self.task_groups = self._prepare_task_groups(
                task_groups=task_groups,
                output_tasks=self.output_tasks,
            )

            self.lora_tasks = list(config.CRLOMTL.LORA_TASKS)

            self.task_to_lora_task = {}
            for group_name, group_tasks in self.task_groups.items():
                for task_name in group_tasks:
                    self.task_to_lora_task[task_name] = group_name

        else:
            self.task_groups = {
                task: [task]
                for task in self.output_tasks
            }

            self.lora_tasks = list(self.output_tasks)

            self.task_to_lora_task = {
                task: task
                for task in self.output_tasks
            }

        self._validate_lora_routing()

        # Useful for trainer: task-specific CRLoMTL passes should iterate over
        # groups for CelebA grouped LoRA, not over all 40 attributes.
        self.training_tasks = self.lora_tasks

        self.adapter_name = get_adapter_name(config)
        Backbone = get_backbone_class(self.adapter_name)

        logger.info(f"Using adapter: {self.adapter_name}")
        logger.info(f"Using backbone class: {Backbone.__name__}")
        logger.info(f"CelebA output tasks: {len(self.output_tasks)}")
        logger.info(f"CelebA LoRA tasks/clusters: {self.lora_tasks}")

        backbone_kwargs = build_backbone_kwargs(
            config=config,
            layernorm=layernorm,
            tasks=self.lora_tasks,
            lora_config=lora_config,
            adapter_name=self.adapter_name,
        )

        self.backbone = Backbone(**backbone_kwargs)

        self.feature_dim = self.backbone.num_features
        self.embed_dim = self.backbone.embed_dim
        self.img_size = self.backbone.patch_embed.img_size

        self.heads = CelebAHeadGroup(
            tasks=self.output_tasks,
            in_features=self.feature_dim,
            dropout=head_dropout,
        )

    @staticmethod
    def _prepare_task_groups(task_groups, output_tasks):
        if task_groups is None:
            task_groups = DEFAULT_CELEBA_GROUPS

        output_task_set = set(output_tasks)
        clean_groups = {}

        for group_name, group_tasks in dict(task_groups).items():
            valid_tasks = [
                task
                for task in group_tasks
                if task in output_task_set
            ]

            if len(valid_tasks) > 0:
                clean_groups[group_name] = valid_tasks

        covered = set()
        for group_tasks in clean_groups.values():
            covered.update(group_tasks)

        missing = [
            task
            for task in output_tasks
            if task not in covered
        ]

        if len(missing) > 0:
            clean_groups["other"] = missing

        return clean_groups

    def _validate_lora_routing(self):
        missing = [
            task
            for task in self.output_tasks
            if task not in self.task_to_lora_task
        ]

        if len(missing) > 0:
            raise ValueError(
                f"Some CelebA output tasks are not mapped to any LoRA task: {missing}"
            )

        unknown_lora_tasks = sorted(
            set(self.task_to_lora_task.values()) - set(self.lora_tasks)
        )

        if len(unknown_lora_tasks) > 0:
            raise ValueError(
                "Some CelebA tasks route to LoRA groups that are not in "
                f"self.lora_tasks: {unknown_lora_tasks}. "
                "Check DEFAULT_CELEBA_GROUPS and config.CRLOMTL.LORA_TASKS."
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

    def _specific_name_tokens(self):
        tokens = []

        # Binary heads are task-specific.
        tokens.extend(
            [
                f"heads.heads.{task}"
                for task in self.output_tasks
            ]
        )

        # LoRA parameters are task/cluster-specific.
        if self.lora_tasks is not None:
            for task in self.lora_tasks:
                tokens.extend(
                    [
                        f"lora_A.{task}",
                        f"lora_B.{task}",
                        f"lora_coeff_A.{task}",
                        f"lora_coeff_B.{task}",
                    ]
                )

        # Fusion/projection modules are not shared backbone params.
        tokens.extend(
            [
                "lora_proj",
                "moe_module",
                "meta_net",
            ]
        )

        return tokens

    def shared_parameters(self):
        tokens = self._specific_name_tokens()

        return [
            p
            for name, p in self.named_parameters()
            if not any(token in name for token in tokens)
        ]

    def last_shared_parameters(self):
        return list(self.backbone.layers[-1].parameters())

    def task_specific_parameters(self):
        tokens = self._specific_name_tokens()

        return [
            p
            for name, p in self.named_parameters()
            if any(token in name for token in tokens)
        ]

    def task_specific_parameters_grouped(self, tasks=None):
        if tasks is None:
            tasks = self.lora_tasks

        grouped = []

        for item in tasks:
            tokens = []

            if item in self.task_groups:
                # Cluster group: LoRA cluster + heads inside that cluster.
                tokens.extend(
                    [
                        f"lora_A.{item}",
                        f"lora_B.{item}",
                        f"lora_coeff_A.{item}",
                        f"lora_coeff_B.{item}",
                    ]
                )

                tokens.extend(
                    [
                        f"heads.heads.{task}"
                        for task in self.task_groups[item]
                    ]
                )

            elif item in self.output_tasks:
                # Single output task: its head + mapped LoRA task.
                tokens.append(f"heads.heads.{item}")

                lora_task = self.task_to_lora_task.get(item, None)

                if lora_task is not None:
                    tokens.extend(
                        [
                            f"lora_A.{lora_task}",
                            f"lora_B.{lora_task}",
                            f"lora_coeff_A.{lora_task}",
                            f"lora_coeff_B.{lora_task}",
                        ]
                    )

            else:
                tokens.append(str(item))

            params = [
                p
                for name, p in self.named_parameters()
                if p.requires_grad and any(token in name for token in tokens)
            ]

            grouped.append(params)

        return grouped

    def resolve_forward_routing(self, task=None):
        if task is None:
            # Global/fused pass or plain MTL baseline.
            return None, self.output_tasks

        if task in self.task_groups:
            # CRLoMTL cluster-specific pass.
            return task, self.task_groups[task]

        if task in self.output_tasks:
            # Single attribute pass.
            return self.task_to_lora_task.get(task, None), [task]

        raise ValueError(
            f"Unknown task or cluster '{task}'. "
            f"Expected output task from {self.output_tasks} "
            f"or LoRA task/cluster from {list(self.task_groups.keys())}."
        )

    def logits_dict_to_tensor(self, logits_dict, tasks=None):
        if tasks is None:
            tasks = self.output_tasks

        return torch.stack(
            [logits_dict[task] for task in tasks],
            dim=1,
        )

    def forward(
        self,
        x,
        task=None,
        return_representation=False,
        return_shared_ft=False,
        last_only=False,
        return_tensor=False,
    ):
        assert not (
            return_shared_ft and return_representation
        ), "return_shared_ft and return_representation cannot both be True."

        backbone_task, active_output_tasks = self.resolve_forward_routing(task)

        # For CelebA classification, use final normalized pooled Swin feature.
        pooled = self.backbone(
            x,
            return_stages=False,
            task=backbone_task,
            apply_norm=True,
        )

        result = self.heads(
            pooled,
            active_tasks=active_output_tasks,
        )

        if return_tensor:
            result = self.logits_dict_to_tensor(
                result,
                tasks=active_output_tasks,
            )

        if return_representation:
            return result, [pooled]

        if return_shared_ft:
            return result, {
                task_name: pooled
                for task_name in active_output_tasks
            }

        return result
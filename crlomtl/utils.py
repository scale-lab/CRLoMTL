#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from crlomtl.crlora import LoRALayer


def mark_only_lora_as_trainable(
    model: nn.Module,
    bias: str = "none",
    freeze_tasks: Optional[List[str]] = [],
    freeze_coeff: bool = False,
) -> None:
    def is_task_lora_and_should_freeze(k):
        return (
            freeze_tasks is not None
            and len(freeze_tasks) > 0
            and any([k.endswith(t) for t in freeze_tasks])
        )

    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        elif "coeff" in n and not freeze_coeff:
            p.requires_grad = True
        elif is_task_lora_and_should_freeze(n):
            p.requires_grad = False

    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(
    model: nn.Module, bias: str = "none", task: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()

    def check_for_task(k):
        return task is None or task in k

    if bias == "none":
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if "lora_" in k and check_for_task(k)
        }
    elif bias == "all":
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if ("lora_" in k or "bias" in k) and check_for_task(k)
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k and check_for_task(k):
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

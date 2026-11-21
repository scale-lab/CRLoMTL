import copy
import logging
import os

import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from crlomtl.utils import lora_state_dict

def adapter_state_dict(config, model):
    return lora_state_dict(model)

def non_adapter_state_dict(model):
    """
    Return model weights after merging, excluding LoRA/adapter-only tensors.

    Also maps ConvLoRA patch embedding keys back to vanilla Swin keys:
        backbone.patch_embed.proj.conv.weight -> backbone.patch_embed.proj.weight
        backbone.patch_embed.proj.conv.bias   -> backbone.patch_embed.proj.bias
    """
    filtered = {}

    for k, v in model.state_dict().items():
        key_lower = k.lower()

        # Remove adapter-only tensors.
        if "lora_" in key_lower:
            continue
        if "lora_a" in key_lower:
            continue
        if "lora_b" in key_lower:
            continue
        if "lora_coeff" in key_lower:
            continue
        if "lora_proj" in key_lower:
            continue

        new_k = k

        # Map CRLoMTL ConvLoRA patch embedding back to vanilla Swin.
        new_k = new_k.replace(
            "backbone.patch_embed.proj.conv.weight",
            "backbone.patch_embed.proj.weight",
        )
        new_k = new_k.replace(
            "backbone.patch_embed.proj.conv.bias",
            "backbone.patch_embed.proj.bias",
        )

        new_k = new_k.replace(
            "patch_embed.proj.conv.weight",
            "patch_embed.proj.weight",
        )
        new_k = new_k.replace(
            "patch_embed.proj.conv.bias",
            "patch_embed.proj.bias",
        )

        filtered[new_k] = v

    return filtered

@torch.no_grad()
def _merge_single_lora_module(module):
    """
    Merge one LoRA/CRLoMTL Linear-like module into its base weight.
    """
    if not hasattr(module, "r") or module.r <= 0:
        return

    if hasattr(module, "merged") and module.merged:
        return

    # ConvLoRA (patch embedding) keeps its weight in module.conv
    if hasattr(module, "conv") and hasattr(module, "get_merged_lora_update"):
        delta = module.get_merged_lora_update()
        if delta is not None:
            w = module.conv.weight
            w.data += delta.reshape(w.shape).to(w.dtype) * module.scaling
        module.merged = True
        return

    if not hasattr(module, "weight"):
        return

    def T(w):
        return w.transpose(0, 1) if getattr(module, "fan_in_fan_out", False) else w

    scaling = getattr(module, "scaling", 1.0)
    has_task_specialization = getattr(module, "has_task_specialization", False)

    # Standard LoRA case.
    if not has_task_specialization:
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            delta = module.lora_B @ module.lora_A
            module.weight.data += T(delta).to(module.weight.dtype) * scaling

            if hasattr(module, "merged"):
                module.merged = True
        return

    # CRLoMTL task-specialized case.
    if not hasattr(module, "resolver_fn"):
        return
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return
    if not hasattr(module, "tasks"):
        return

    # Match the exact resolver_args used in Linear.forward().
    resolver_args = {
        "x": None,
        "lora_A": module.lora_A,
        "lora_B": module.lora_B,
        "tasks": module.tasks,
        "direction_A": getattr(module, "lora_proj_A", None),
        "direction_B": getattr(module, "lora_proj_B", None),
        "coeff_A": getattr(module, "lora_coeff_A", None),
        "coeff_B": getattr(module, "lora_coeff_B", None),
        "training": False,
        "noise_factor": getattr(module, "noise_factor", 0.0),
        "temperature": getattr(module, "softmax_temperature", 1.0),
        "atten_emb_dim": getattr(module, "atten_emb_dim", 64),
        "kernel_type": getattr(module, "kernel_type", "rbf"),
    }

    fused = module.resolver_fn(**resolver_args)
    module.weight.data += T(fused).to(module.weight.dtype) * scaling

    if hasattr(module, "merged"):
        module.merged = True


@torch.no_grad()
def merge_lora_model_weights(model):
    """
    Merge all LoRA/CRLoMTL modules into their base weights in-place.
    """
    for module in model.modules():
        _merge_single_lora_module(module)

    return model


def make_merged_deployment_state_dict(model):
    """
    Create a merged deployment state_dict without modifying the training model.
    """
    merged_model = copy.deepcopy(model)
    merged_model.eval()
    merge_lora_model_weights(merged_model)
    return non_adapter_state_dict(merged_model)

def save_final_checkpoint(
    config,
    model,
    optimizer,
    scheduler,
    tag,
    adapter_enabled,
):
    """
    Save:
      1. normal full checkpoint for resuming/debugging
      2. adapter-only checkpoint
      3. merged deployment checkpoint with LoRA/CRLoMTL folded into weights
      4. optimizer and scheduler checkpoints
    """

    ckpt_save_dir = config.OUTPUT
    os.makedirs(ckpt_save_dir, exist_ok=True)

    adapter_save_path = None

    if adapter_enabled:
        adapter_save_path = os.path.normpath(
            f"{ckpt_save_dir}/adapter_ckpt_last.pth"
        )
        torch.save(
            {"model": adapter_state_dict(config, model)},
            adapter_save_path,
        )
        logging.info(f"Saved adapter at last epoch to {adapter_save_path}")

    model_save_path = os.path.normpath(
        f"{ckpt_save_dir}/ckpt_last.pth"
    )
    merged_model_save_path = os.path.normpath(
        f"{ckpt_save_dir}/ckpt_last_merged.pth"
    )
    optimizer_save_path = os.path.normpath(
        f"{ckpt_save_dir}/optimizer_last.pth"
    )
    scheduler_save_path = os.path.normpath(
        f"{ckpt_save_dir}/scheduler_last.pth"
    )

    # Normal checkpoint: keeps adapters. Use this for resume/debug.
    torch.save(
        {"model": model.state_dict()},
        model_save_path,
    )

    # Merged deployment checkpoint: folds fused LoRA update into base weights
    # and removes adapter-only keys.
    merged_model_save_success = False

    try:
        merged_state = make_merged_deployment_state_dict(model)

        torch.save(
            {"model": merged_state},
            merged_model_save_path,
        )

        merged_model_save_success = True

        logging.info(
            f"Saved final merged deployment model to {merged_model_save_path}"
        )

    except Exception as exc:
        merged_model_save_path = None

        logging.warning(
            f"Could not save final merged deployment checkpoint: {exc}"
        )

    torch.save(
        {"optimizer": optimizer.state_dict()},
        optimizer_save_path,
    )

    torch.save(
        {"scheduler": scheduler.state_dict()},
        scheduler_save_path,
    )

    logging.info(f"Saved final model to {model_save_path}")
    logging.info(f"Saved final optimizer to {optimizer_save_path}")
    logging.info(f"Saved final scheduler to {scheduler_save_path}")

    maybe_log_final_artifacts_to_wandb(
        config=config,
        tag=tag,
        model_save_path=model_save_path,
        optimizer_save_path=optimizer_save_path,
        scheduler_save_path=scheduler_save_path,
        adapter_save_path=adapter_save_path,
        adapter_enabled=adapter_enabled,
    )

    return {
        "model": model_save_path,
        "merged_model": merged_model_save_path if merged_model_save_success else None,
        "optimizer": optimizer_save_path,
        "scheduler": scheduler_save_path,
        "adapter": adapter_save_path,
    }

   
def maybe_log_epoch_artifacts_to_wandb(
    config,
    tag,
    epoch,
    model_save_path,
    optimizer_save_path,
    scheduler_save_path,
    adapter_save_path=None,
    adapter_enabled=False,
):
    """
    Log epoch checkpoints to wandb if enabled.
    """

    if not config.WANDB:
        return

    if not HAS_WANDB:
        logging.warning("WANDB=True but wandb is not installed.")
        return

    wandb_save_model = getattr(config, "WANDB_SAVE_MODEL", False)
    wandb_save_freq = getattr(
        config,
        "WANDB_SAVE_FREQ",
        config.EVAL_AND_SAVE_FREQ,
    )

    if not wandb_save_model:
        return

    if (epoch + 1) % wandb_save_freq != 0:
        return

    logging.info(
        f"Saving model checkpoint to wandb at epoch {epoch + 1}"
    )

    model_artifact = wandb.Artifact(
        name=f"{tag}_model_epoch_{epoch + 1}",
        type="model",
        description=f"Model checkpoint at epoch {epoch + 1}",
    )
    model_artifact.add_file(model_save_path)
    wandb.log_artifact(model_artifact)

    if adapter_enabled and adapter_save_path is not None:
        adapter_artifact = wandb.Artifact(
            name=f"{tag}_adapter_model_epoch_{epoch + 1}",
            type="model",
            description=f"Adapter model checkpoint at epoch {epoch + 1}",
        )
        adapter_artifact.add_file(adapter_save_path)
        wandb.log_artifact(adapter_artifact)

    optimizer_artifact = wandb.Artifact(
        name=f"{tag}_optimizer_epoch_{epoch + 1}",
        type="optimizer",
        description=f"Optimizer state at epoch {epoch + 1}",
    )
    optimizer_artifact.add_file(optimizer_save_path)
    wandb.log_artifact(optimizer_artifact)

    scheduler_artifact = wandb.Artifact(
        name=f"{tag}_scheduler_epoch_{epoch + 1}",
        type="scheduler",
        description=f"Scheduler state at epoch {epoch + 1}",
    )
    scheduler_artifact.add_file(scheduler_save_path)
    wandb.log_artifact(scheduler_artifact)


def maybe_log_final_artifacts_to_wandb(
    config,
    tag,
    model_save_path,
    optimizer_save_path,
    scheduler_save_path,
    adapter_save_path=None,
    adapter_enabled=False,
):
    """
    Log final checkpoints to wandb if enabled.
    """

    if not config.WANDB:
        return

    if not HAS_WANDB:
        logging.warning("WANDB=True but wandb is not installed.")
        return

    wandb_save_model = getattr(config, "WANDB_SAVE_MODEL", False)

    if not wandb_save_model:
        return

    logging.info("Saving final model checkpoint to wandb")

    model_artifact = wandb.Artifact(
        name=f"{tag}_model_final",
        type="model",
        description="Final model checkpoint",
    )
    model_artifact.add_file(model_save_path)
    wandb.log_artifact(model_artifact)

    if adapter_enabled and adapter_save_path is not None:
        adapter_artifact = wandb.Artifact(
            name=f"{tag}_adapter_model_final",
            type="model",
            description="Final Adapter model checkpoint",
        )
        adapter_artifact.add_file(adapter_save_path)
        wandb.log_artifact(adapter_artifact)

    optimizer_artifact = wandb.Artifact(
        name=f"{tag}_optimizer_final",
        type="optimizer",
        description="Final optimizer state",
    )
    optimizer_artifact.add_file(optimizer_save_path)
    wandb.log_artifact(optimizer_artifact)

    scheduler_artifact = wandb.Artifact(
        name=f"{tag}_scheduler_final",
        type="scheduler",
        description="Final scheduler state",
    )
    scheduler_artifact.add_file(scheduler_save_path)
    wandb.log_artifact(scheduler_artifact)

def save_checkpoint_bundle(
    config,
    model,
    optimizer,
    scheduler,
    epoch,
    tag,
    adapter_enabled,
):
    """
    Save model, adapter, optimizer, and scheduler checkpoints for a specific epoch.
    """

    ckpt_save_dir = config.OUTPUT
    os.makedirs(ckpt_save_dir, exist_ok=True)

    epoch_str = str(epoch + 1).zfill(len(str(config.TRAIN.EPOCHS)))

    adapter_save_path = None

    if adapter_enabled:
        adapter_save_path = os.path.normpath(
            f"{ckpt_save_dir}/adapter_ckpt_epoch_{epoch_str}.pth"
        )
        torch.save(
            {"model": adapter_state_dict(config, model)},
            adapter_save_path,
        )
        logging.info(
            f"Saved adapter at epoch {epoch} to {adapter_save_path}"
        )

    model_save_path = os.path.normpath(
        f"{ckpt_save_dir}/ckpt_epoch_{epoch_str}.pth"
    )
    optimizer_save_path = os.path.normpath(
        f"{ckpt_save_dir}/optimizer_epoch_{epoch_str}.pth"
    )
    scheduler_save_path = os.path.normpath(
        f"{ckpt_save_dir}/scheduler_epoch_{epoch_str}.pth"
    )

    torch.save({"model": model.state_dict()}, model_save_path)
    torch.save({"optimizer": optimizer.state_dict()}, optimizer_save_path)
    torch.save({"scheduler": scheduler.state_dict()}, scheduler_save_path)

    logging.info(f"Saved model at epoch {epoch} to {model_save_path}")
    logging.info(f"Saved optimizer at epoch {epoch} to {optimizer_save_path}")
    logging.info(f"Saved scheduler at epoch {epoch} to {scheduler_save_path}")

    maybe_log_epoch_artifacts_to_wandb(
        config=config,
        tag=tag,
        epoch=epoch,
        model_save_path=model_save_path,
        optimizer_save_path=optimizer_save_path,
        scheduler_save_path=scheduler_save_path,
        adapter_save_path=adapter_save_path,
        adapter_enabled=adapter_enabled,
    )

    return {
        "model": model_save_path,
        "optimizer": optimizer_save_path,
        "scheduler": scheduler_save_path,
        "adapter": adapter_save_path,
    }
import logging
import torch

from crlomtl.utils import mark_only_lora_as_trainable as mark_only_crlora_as_trainable
from models.swin.swin_mtl import SwinMTL
from models.swin.swin_celeba import SwinCelebA

MODELS = {
    "swin": SwinMTL,
    "swin_celeba": SwinCelebA,
}

def is_enabled(config, name):
    name = name.upper()
    return (
        hasattr(config.MODEL, name)
        and getattr(config.MODEL, name).ENABLED
    )


def get_adapter_name(config):
    # check shared LoRA first, CRLoMTL also sets LORA.ENABLED
    if is_enabled(config, "SHARED_LORA"):
        return "shared_lora"

    if config.LORA.ENABLED:
        return "lora"

    return "none"


def mark_only_shared_lora_trainable(model, logger=None):
    """
    Freeze the whole MTL model (backbone, decoders, downsampler, MTI) and
    train only the shared LoRA matrices. Used for the Shared-LoRA ablation.
    """

    logger = logger or logging.getLogger(__name__)

    # freeze everything
    for name, param in model.named_parameters():
        param.requires_grad = False

    # then unfreeze LoRA only
    num_lora_tensors = 0
    num_lora_params = 0

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            num_lora_tensors += 1
            num_lora_params += param.numel()

    # check that nothing else is trainable
    trainable_names = [
        name for name, param in model.named_parameters()
        if param.requires_grad
    ]

    non_lora_trainable = [
        name for name in trainable_names
        if "lora_A" not in name and "lora_B" not in name
    ]

    logger.info(
        "Frozen full MTL model including decoders; "
        f"training only shared LoRA tensors={num_lora_tensors}, "
        f"LoRA params={num_lora_params:,}"
    )

    logger.info(
        "First 20 trainable parameters: "
        f"{trainable_names[:20]}"
    )

    if len(non_lora_trainable) > 0:
        logger.warning(
            "Non-LoRA parameters are still trainable: "
            f"{non_lora_trainable[:50]}"
        )

    if num_lora_tensors == 0:
        logger.warning(
            "No LoRA parameters were found. "
            "Check that shared_lora model was actually built."
        )

    return model

def adapter_is_enabled(config):
    return get_adapter_name(config) != "none"

def build_lora_config(config):
    if not config.LORA.ENABLED:
        return None

    lora_config = {
        "params": {
            "r": config.LORA.RANK,
            "lora_alpha": config.LORA.ALPHA,
            "lora_dropout": config.LORA.DROPOUT,
            "task_r": config.LORA.TASK_RANK,
            "task_alpha": config.LORA.TASK_ALPHA,
            "task_dropout": config.LORA.TASK_DROPOUT,
            "noise_factor": config.LORA.NOISE_FACTOR,
            "softmax_temperature": config.CRLOMTL.SOFTMAX_TEMP,
            "kernel_type": config.CRLOMTL.KERNEL,
            "atten_emb_dim": config.CRLOMTL.ATTEN_EMB_DIM,
        },
        "freeze_patch_embed": config.LORA.FREEZE_PATCH_EMBED,
        "freeze_patch_merge": config.LORA.FREEZE_PATCH_MERGE,
    }

    if "crlomtl" in config.METHOD:
        lora_config["params"]["resolver"] = config.CRLOMTL.RESOLVER
        lora_config["params"]["full_rank_resolver"] = config.CRLOMTL.FULL_RANK_RESOLVER
        if config.CRLOMTL.SHARED_RANK_RATIO is not None:
            lora_config["params"]["shared_rank_ratio"] = config.CRLOMTL.SHARED_RANK_RATIO

    # layers that get the shared LoRA, e.g. ["qkv", "proj", "fc1", "fc2"]
    if is_enabled(config, "SHARED_LORA") and hasattr(config.MODEL.SHARED_LORA, "TARGET_MODULES"):
        lora_config["target_modules"] = list(config.MODEL.SHARED_LORA.TARGET_MODULES)

    return lora_config


def log_trainable_parameter_audit(model, logger=None):
    """
    Log how many LoRA, decoder and other parameters are trainable.
    """

    logger = logger or logging.getLogger(__name__)

    total_params = 0
    trainable_params = 0

    lora_trainable_params = 0
    decoder_trainable_params = 0
    backbone_non_lora_trainable_params = 0
    non_lora_trainable_params = 0

    trainable_names = []
    decoder_trainable_names = []
    non_lora_trainable_names = []

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n

        if not param.requires_grad:
            continue

        trainable_params += n
        trainable_names.append(name)

        is_lora = "lora_A" in name or "lora_B" in name
        is_decoder = (
            "decoder" in name.lower()
            or "decoders" in name.lower()
            or "head" in name.lower()
            or "heads" in name.lower()
            or "downsampler" in name.lower()
            or "mti" in name.lower()
            or "mti_net" in name.lower()
        )

        if is_lora:
            lora_trainable_params += n
        else:
            non_lora_trainable_params += n
            non_lora_trainable_names.append(name)

        if is_decoder:
            decoder_trainable_params += n
            decoder_trainable_names.append(name)

        if "backbone" in name and not is_lora:
            backbone_non_lora_trainable_params += n

    logger.info("========== [VERIFY TRAINABLE PARAMS] ==========")
    logger.info(f"total_params: {total_params:,}")
    logger.info(f"trainable_params: {trainable_params:,}")
    logger.info(f"lora_trainable_params: {lora_trainable_params:,}")
    logger.info(f"decoder_trainable_params: {decoder_trainable_params:,}")
    logger.info(
        f"backbone_non_lora_trainable_params: "
        f"{backbone_non_lora_trainable_params:,}"
    )
    logger.info(f"non_lora_trainable_params: {non_lora_trainable_params:,}")
    logger.info(f"num_trainable_tensors: {len(trainable_names)}")
    logger.info(f"first_30_trainable_tensors: {trainable_names[:30]}")
    logger.info(
        f"first_30_decoder_trainable_tensors: "
        f"{decoder_trainable_names[:30]}"
    )
    logger.info(
        f"first_30_non_lora_trainable_tensors: "
        f"{non_lora_trainable_names[:30]}"
    )
    logger.info("================================================")

def apply_adapter_freezing(config, model, logger, adapter_name,tasks=None):
    target_module = getattr(model, "backbone", model)

    # shared LoRA: the MTL model (backbone and decoders) stays frozen, only LoRA trains
    if adapter_name == "shared_lora":
        return mark_only_shared_lora_trainable(model, logger)

    if adapter_name == "lora":
        logger.info(
            f"Applying CRLoRA-style trainable parameter selection for {adapter_name}"
        )
        return mark_only_crlora_as_trainable(
            target_module,
            bias=getattr(config.LORA, "BIAS", "none"),
            #trainable_classifier=True,
            #trainable_scaling=False,
        )

    return model

def apply_backbone_freezing(model, config, logger=None):
    logger = logger or logging.getLogger(__name__)

    if not config.MODEL.FREEZE_BACKBONE:
        logger.info("MODEL.FREEZE_BACKBONE=False; skipping backbone freezing.")
        return model

    if not hasattr(model, "backbone"):
        logger.warning("MODEL.FREEZE_BACKBONE=True, but model has no backbone attribute.")
        return model

    logger.info("Freezing backbone.")

    for param in model.backbone.parameters():
        param.requires_grad = False

    if config.MODEL.UNFREEZE_PATCH_EMBED:
        logger.info("Unfreezing patch embedding.")

        if hasattr(model, "patch_embed"):
            for param in model.patch_embed.parameters():
                param.requires_grad = True
        elif hasattr(model.backbone, "patch_embed"):
            for param in model.backbone.patch_embed.parameters():
                param.requires_grad = True
        else:
            logger.warning("UNFREEZE_PATCH_EMBED=True, but no patch_embed found.")

    if config.MODEL.UNFREEZE_PATCH_MERGE:
        logger.info("Unfreezing patch merge layers.")

        for name, param in model.backbone.named_parameters():
            if "patch_merge" in name or "patch_merging" in name or "downsample" in name:
                param.requires_grad = True

    return model


def build_model(config, logger):

    device = torch.device(config.DEVICE)
    tasks = config.TASKS
    model_name = config.MODEL.TYPE
    assert model_name in MODELS, f"Model {model_name} not found"
    adapter_name = get_adapter_name(config)
    adapter_enabled = adapter_name != "none"
    lora_config = build_lora_config(config)
    model = MODELS[model_name](
        config=config,
        logger=logger,
        lora_config=lora_config,
        tasks=tasks,
    ).to(device, non_blocking=True)
    apply_backbone_freezing(model, config, logger)
    apply_adapter_freezing(
        model=model,
        config=config,
        tasks=tasks,
        adapter_name=adapter_name,
        logger=logger,
    )

    return model, device, tasks, adapter_enabled, adapter_name, lora_config

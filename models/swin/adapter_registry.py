from models.swin.swin_transformer import SwinTransformer
from models.swin.swin_transformer_shared_lora import SwinTransformerSharedLoRA


def is_enabled(config, name):
    name = name.upper()
    return (
        hasattr(config.MODEL, name)
        and getattr(config.MODEL, name).ENABLED
    )


def get_adapter_name(config):
    """
    Returns one adapter name. The order matters if several are enabled.
    """
    if is_enabled(config, "SHARED_LORA"):
        return "shared_lora"

    if config.LORA.ENABLED:
        return "crlomtl"

    return "none"


def get_backbone_class(adapter_name):
    backbone_map = {
        "none": SwinTransformer,
        "crlomtl": SwinTransformer,
        "shared_lora": SwinTransformerSharedLoRA,
    }

    if adapter_name not in backbone_map:
        raise ValueError(f"Unknown adapter_name: {adapter_name}")

    return backbone_map[adapter_name]


def build_backbone_kwargs(config, layernorm, tasks, lora_config, adapter_name):
    """
    Common Swin kwargs + adapter-specific kwargs.
    """
    kwargs = dict(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        norm_layer=layernorm,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        fused_window_process=config.FUSED_WINDOW_PROCESS,
        tasks=tasks,
    )

    if adapter_name in ["crlomtl", "shared_lora"]:
        kwargs["lora_config"] = lora_config

    return kwargs


def adapter_returns_task_stage_dict(adapter_name, shared_representation):
    # every backbone kept in this repo returns one tensor per stage
    return False

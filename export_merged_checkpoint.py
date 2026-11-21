import logging
import os

import torch

from config import get_config
from utils import create_cmd_parser, load_checkpoint
from setup import (
    DATASETS_LOADER,
    MODELS,
    build_model,
    setup_logging,
)

from engine import make_merged_deployment_state_dict


METHODS = [
    "multi",
    "nashmtl",
    "pcgrad",
    "cagrad",
    "crlomtl",
    "shared_lora",
    "mgda",
    "mti",
    "famo",
    "imtl",
    "uw",
    "ls",
    "consmtl",
    "go4align",
    "fairgrad",
]


def main():
    parser = create_cmd_parser(
        models=MODELS,
        methods=METHODS,
        dataset_loaders=DATASETS_LOADER,
    )

    args = parser.parse_args()
    config = get_config(args)

    logger, tag, logs_save_dir = setup_logging(config)

    logger.info("=====================================")
    logger.info("Exporting merged CRLoMTL checkpoint")
    logger.info(f"config: {args.cfg}")
    logger.info(f"resume: {config.MODEL.RESUME}")
    logger.info(f"output: {config.OUTPUT}")
    logger.info("=====================================")

    if not config.MODEL.RESUME:
        raise ValueError("Pass the trained checkpoint with --resume")

    os.makedirs(config.OUTPUT, exist_ok=True)

    model, device, tasks, adapter_enabled, adapter_name, lora_config = build_model(
        config=config,
        logger=logger,
    )

    logger.info(f"adapter_enabled: {adapter_enabled}")
    logger.info(f"adapter_name: {adapter_name}")
    logger.info(f"tasks: {tasks}")

    load_checkpoint(
        config=config,
        logger=logger,
        lora_enabled=adapter_enabled,
        model=model,
    )

    model.eval()

    merged_state = make_merged_deployment_state_dict(model)

    merged_save_path = os.path.join(config.OUTPUT, "ckpt_last_merged.pth")

    torch.save(
        {"model": merged_state},
        merged_save_path,
    )

    logger.info(f"Saved regenerated merged checkpoint to: {merged_save_path}")
    logger.info(f"Merged checkpoint tensors: {len(merged_state)}")

    # Quick sanity check for the patch embedding key fix.
    keys = list(merged_state.keys())

    has_vanilla_patch = (
        "backbone.patch_embed.proj.weight" in keys
        or "patch_embed.proj.weight" in keys
    )

    has_conv_patch = any(
        k.endswith("patch_embed.proj.conv.weight")
        for k in keys
    )

    logger.info(f"Has vanilla patch_embed.proj.weight: {has_vanilla_patch}")
    logger.info(f"Has old patch_embed.proj.conv.weight: {has_conv_patch}")

    if has_conv_patch:
        logger.warning(
            "Merged checkpoint still contains patch_embed.proj.conv.weight. "
            "Check non_adapter_state_dict() key mapping."
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
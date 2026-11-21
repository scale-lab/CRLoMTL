# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import re

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from data.mtl_dataset import (get_mtl_train_dataloader, get_mtl_train_dataset,
                              get_mtl_val_dataloader, get_mtl_val_dataset,
                              get_transformations)
#from data.celeba_dataset import CelebAMTLDataset
try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER
            if config.AUG.COLOR_JITTER > 0
            else None,
            auto_augment=config.AUG.AUTO_AUGMENT
            if config.AUG.AUTO_AUGMENT != "none"
            else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4
            )
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(
                    size, interpolation=_pil_interp(config.DATA.INTERPOLATION)
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_mtl(config, db_name="NYUD", val_only=False):
    config.defrost()
    if val_only:
        _, val_transforms = get_transformations(db_name, config.TASKS_CONFIG)
        dataset_val = get_mtl_val_dataset(db_name, config, val_transforms)
        data_loader_val = get_mtl_val_dataloader(config, dataset_val)
        return dataset_val, data_loader_val

    train_transforms, val_transforms = get_transformations(db_name, config.TASKS_CONFIG)
    dataset_train = get_mtl_train_dataset(db_name, config, train_transforms)
    dataset_val = get_mtl_val_dataset(db_name, config, val_transforms)
    data_loader_train = get_mtl_train_dataloader(config, dataset_train)
    data_loader_val = get_mtl_val_dataloader(config, dataset_val)
    mixup_fn = None

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def create_nyud_loader(config):
    _, _, data_loader_train, data_loader_val, _ = build_mtl(config, "NYUD")
    return data_loader_train, data_loader_val


def create_pascal_loader(config, val_only=False):
    _, _, data_loader_train, data_loader_val, _ = build_mtl(
        config, "PASCALContext", val_only=val_only
    )
    return data_loader_train, data_loader_val


def create_celeba_loader(config, val_only=False):

    if val_only:
        dataset_val, data_loader_val = build_mtl(
            config,
            "CelebA",
            val_only=True,

        )

        return dataset_val, data_loader_val
    _, _, data_loader_train, data_loader_val, _ = build_mtl(
        config,
        "CelebA",
        val_only=False,

    )

    return data_loader_train, data_loader_val

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import re

import torch
import yaml
from yacs.config import CfgNode as CN

from crlomtl.crlora import resolvers
from data.mtl_dataset import get_tasks_config

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split(".")[0])




_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
_C.DATA.EVAL_BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = "./dataset"
# Dataset name
_C.DATA.DATASET = "PASCALContext"
# Input image size
_C.DATA.IMG_SIZE = 448
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = "part"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "swin"
# Model name
_C.MODEL.NAME = "swin_tiny_patch4_window7_224"
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ""
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ""
_C.MODEL.RESUME_OPTIMIZER = ""
_C.MODEL.RESUME_SCHEDULER = ""
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.0
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.0
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = None
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.0
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# [SimMIM] Norm target during training
_C.MODEL.SIMMIM = CN()
_C.MODEL.SIMMIM.NORM_TARGET = CN()
_C.MODEL.SIMMIM.NORM_TARGET.ENABLE = False
_C.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 500
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False


# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# Extra freezing options
_C.TRAIN.FREEZE_PATCH_EMBED = False
_C.TRAIN.FREEZE_LAYER_NORM = False
_C.TRAIN.FREEZE_RELATIVE_POSITION_BIAS = False
_C.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION = False

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = "pixel"
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ""
# Path to output folder, overwritten by command line argument
_C.OUTPUT = "output"
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.EVAL_AND_SAVE_FREQ = 25
# Frequency to logging info
_C.PRINT_FREQ = 25
# Fixed random seed
_C.SEED = 42
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
_C.EVAL_TASK_SPECIFIC = False

# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False
#####
_C.LOGGING_TRAINING_TIME=False
# CRLoMTL realted configuration
_C.METHOD = "multi"
_C.TASKS = None
_C.LORA = CN()
_C.LORA.ENABLED = False
_C.LORA.RANK = 8
_C.LORA.ALPHA = 16
_C.LORA.DROPOUT = 0.0
_C.LORA.TASK_RANK = CN(new_allowed=True)
_C.LORA.TASK_ALPHA = CN(new_allowed=True)
_C.LORA.TASK_DROPOUT = CN(new_allowed=True)
_C.LORA.FREEZE_PATCH_EMBED = False
_C.LORA.FREEZE_PATCH_MERGE = False
_C.LORA.NO_FREEZE = False
_C.LORA.NOISE_FACTOR = 0.0
_C.TRAIN.DISABLE_PER_TASK_TRAINING = False
_C.CRLOMTL = CN()
_C.CRLOMTL.MODE = "alternating_with_interm"
# r_s / r for the shared_subspace resolvers, None means r/2 (paper default)
_C.CRLOMTL.SHARED_RANK_RATIO = None
_C.CRLOMTL.RESOLVER = "mean"
_C.CRLOMTL.FULL_RANK_RESOLVER = False
_C.CRLOMTL.SOFTMAX_TEMP = 1.0
_C.CRLOMTL.KERNEL = "rbf"
_C.CRLOMTL.ATTEN_EMB_DIM = 64
_C.CRLOMTL.UNFREEZE_BACKBONE = False
_C.CRLOMTL.COSINE_LOSS = False
_C.CRLOMTL.COSINE_LOSS_LAMBDA = 5.0
_C.CRLOMTL.FEATURE_REGULARIZATION = None
_C.CRLOMTL.FEATURE_REGULARIZATION_WEIGHT = 1.0
_C.CRLOMTL.FEATURE_REGULARIZATION_CHANNELWISE_TEMP = 4.0
_C.CRLOMTL.FEATURE_REGULARIZATION_NORM = None
_C.CRLOMTL.MTL_LOSS_LAMBDA = 1.0
_C.CRLOMTL.LORA_TASKS = []
_C.CRLOMTL.NUM_PASS_TASKS_PER_BATCH = 0
"""_C.CRLOMTL.LORA_TASKS = [
    "hair",
    "facial_hair",
    "expression",
    "makeup",
    "accessories",
    "face_shape",
    "identity_style",
]"""

_C.TRAIN.METHOD_LR = 0.001
_C.LOGDIR = "logs"
_C.MODEL.RESUME_LORA = None
_C.MODEL.RESUME_BACKBONE = False
_C.MODEL.UNFREEZE_ALL = False
_C.MODEL.MTI = False
_C.WANDB = False
_C.DISABLE_PROGRESS = False
_C.DEVICE = "cuda"


# Multi task deocders
_C.MODEL.DECODER_DOWNSAMPLER = True
_C.MODEL.PER_TASK_DOWNSAMPLER = True
_C.MODEL.DECODER_HEAD = CN()
_C.MODEL.DECODER_HEAD["semseg"] = "hrnet"
_C.MODEL.DECODER_HEAD["normals"] = "hrnet"
_C.MODEL.DECODER_HEAD["sal"] = "hrnet"
_C.MODEL.DECODER_HEAD["human_parts"] = "hrnet"
_C.MODEL.DECODER_HEAD["edge"] = "hrnet"
_C.MODEL.DECODER_HEAD["depth"] = "hrnet"
_C.MODEL.DECODER_CHANNELS = [18, 36, 72, 144]
_C.MODEL.FREEZE_BACKBONE = False
_C.MODEL.UNFREEZE_PATCH_EMBED = False
_C.MODEL.UNFREEZE_PATCH_MERGE = False


_C.MODEL.SEGFORMER_CHANNELS = 256
_C.MODEL.MULTISCALE = True
_C.PRINT_LAYERS = False

_C.CRLOMTL.ADAPTIVE_LAMBDA = False
_C.DRAW_COSINE_MATRIX = False
_C.CRLOMTL.EPSILON = 1e-8


# -----------------------------------------------------------------------------
# Shared LoRA configuration
# -----------------------------------------------------------------------------
# This is separate from CRLoMTL.
# Use this only when you want ONE shared LoRA on the Swin backbone.
# Registry will select SwinTransformerSharedLoRA when this is True.
_C.MODEL.SHARED_LORA = CN()
_C.MODEL.SHARED_LORA.ENABLED = False

# Which shared backbone Linear layers receive LoRA.
# Default: attention only. Add "fc1", "fc2" if you want MLP LoRA too.
_C.MODEL.SHARED_LORA.TARGET_MODULES = ["qkv", "proj","fc1", "fc2"]

# Freezing policy for your intended setup:
# freeze original backbone, train shared LoRA + decoders/downsampler.
_C.MODEL.SHARED_LORA.FREEZE_BACKBONE = True
_C.MODEL.SHARED_LORA.TRAIN_DECODERS = True


_C.SVD_ANALYSIS = CN()
_C.SVD_ANALYSIS.ENABLED = False
_C.SVD_ANALYSIS.FREQ = 5
#####################CELEBA CONFIG#####################
CELEBA_TASKS = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",

]

CELEBA_GROUPS = {
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

DEFAULT_CELEBA_GROUPS_4 = {

    "hair_facial": [
        "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Gray_Hair",
        "Receding_Hairline",
        "Straight_Hair",
        "Wavy_Hair",
        "5_o_Clock_Shadow",
        "Goatee",
        "Mustache",
        "No_Beard",
        "Sideburns",

    ],

    "expression_makeup": [
        "Smiling",
        "Mouth_Slightly_Open",
        "Narrow_Eyes",
        "Arched_Eyebrows",
        "Bags_Under_Eyes",
        "Heavy_Makeup",
        "Rosy_Cheeks",
        "Wearing_Lipstick",
        "Bushy_Eyebrows",

    ],

    "accessories_style": [
        "Eyeglasses",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Blurry",
        "Male",
        "Young",

    ],

    "face_shape_skin": [
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

}
_C.DATA.CELEBA_ROOT = "./dataset/celeba"
_C.DATA.CELEBA_IMAGE_SIZE = 224
_C.MODEL.CELEBA = CN()
_C.MODEL.CELEBA.HEAD_DROPOUT = 0.0
_C.MODEL.CELEBA.USE_GROUPED_LORA = True
############ DATASET TASKS ############
DATASET_TASKS = {
    "NYUD": ["depth", "semseg", "edge"],
    # 'PASCALContext': ['semseg', 'normals', 'human_parts', 'edge', 'sal'],
    "PASCALContext": ["semseg", "normals", "human_parts", "sal"],
     "CelebA": CELEBA_TASKS,
}
def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f"args.{name}"):
            return True
        return False

    # merge from specific arguments
    if _check_args("batch_size"):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args("eval_batch_size"):
        config.DATA.EVAL_BATCH_SIZE = args.eval_batch_size
    if _check_args("data_path"):
        config.DATA.DATA_PATH = args.data_path
    if _check_args("zip"):
        config.DATA.ZIP_MODE = True
    if _check_args("cache_mode"):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args("pretrained"):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args("resume"):
        config.MODEL.RESUME = args.resume

    if _check_args("resume_optimizer"):
        config.MODEL.RESUME_OPTIMIZER = args.resume_optimizer
    if _check_args("resume_scheduler"):
        config.MODEL.RESUME_SCHEDULER = args.resume_scheduler
    if _check_args("accumulation_steps"):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args("use_checkpoint"):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args("amp_opt_level"):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == "O0":
            config.AMP_ENABLE = False
    if _check_args("disable_amp"):
        config.AMP_ENABLE = False
    if _check_args("output"):
        config.OUTPUT = args.output
    if _check_args("tag"):
        config.TAG = args.tag
    if _check_args("eval"):
        config.EVAL_MODE = True
    if _check_args("throughput"):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args("enable_amp"):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args("fused_window_process"):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args("fused_layernorm"):
        config.FUSED_LAYERNORM = True
    # Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args("optim"):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    # handle crlomtl related configurations
    if _check_args("method"):
        config.METHOD = args.method
    #is_nyud = _check_args("dataset") and args.dataset == "NYUD"
    #if is_nyud:
    #    config.DATA.DATASET = "NYUD"
    if _check_args("dataset"):
        config.DATA.DATASET = args.dataset
    if _check_args("tasks"):
        config.TASKS = re.split(r"\s*,\s*", args.tasks)
    elif config.TASKS is None:
        config.TASKS = DATASET_TASKS[config.DATA.DATASET]
    if config.METHOD == "mti":
        config.MODEL.MTI = True

    if _check_args("epochs"):
        config.TRAIN.EPOCHS = args.epochs
    if _check_args("eval_and_save_freq"):
        config.EVAL_AND_SAVE_FREQ = args.eval_and_save_freq
    if _check_args("model"):
        config.MODEL.TYPE = args.model
    if config.DATA.DATASET == "CelebA" and config.MODEL.TYPE == "swin":
        config.MODEL.TYPE = "swin_celeba"    
    #if _check_args("dataset"):
    #    config.DATA.DATASET = args.dataset
    if _check_args("lora_r"):
        config.LORA.ENABLED = True
        config.LORA.RANK = args.lora_r
    # compare with None so that 0 can be passed, e.g. --lora_alpha 0
    if getattr(args, "lora_alpha", None) is not None:
        config.LORA.ALPHA = args.lora_alpha
    if getattr(args, "lora_dropout", None) is not None:
        config.LORA.DROPOUT = args.lora_dropout
    if getattr(args, "seed", None) is not None:
        config.SEED = args.seed
    if getattr(args, "lr", None) is not None:
        config.TRAIN.BASE_LR = args.lr
    if _check_args("disable_per_task_training"):
        config.TRAIN.DISABLE_PER_TASK_TRAINING = True
    if _check_args("crlomtl_mode"):
        config.CRLOMTL.MODE = args.crlomtl_mode
    if _check_args("crlomtl_resolver"):
        config.CRLOMTL.RESOLVER = args.crlomtl_resolver
    if _check_args("crlomtl_unfreeze_backbone"):
        config.CRLOMTL.UNFREEZE_BACKBONE = True
    if _check_args("crlomtl_cosine_loss"):
        config.CRLOMTL.COSINE_LOSS = True
    if _check_args("crlomtl_full_rank_resolver"):
        config.CRLOMTL.FULL_RANK_RESOLVER = True
    if _check_args("method_lr"):
        config.TRAIN.METHOD_LR = args.method_lr
    if _check_args("log_dir"):
        config.LOGDIR = args.log_dir
    if _check_args("resume_lora"):
        config.MODEL.RESUME_LORA = args.resume_lora
    if _check_args("resume_backbone"):
        config.MODEL.RESUME_BACKBONE = True
    if _check_args("wandb"):
        config.WANDB = True
    if _check_args("disable_progress"):
        config.DISABLE_PROGRESS = True
    if _check_args("device"):
        config.DEVICE = args.device
    if _check_args("print_layers"):
        config.PRINT_LAYERS = True
    if _check_args("unfreeze_all"):
        config.MODEL.UNFREEZE_ALL = True
    if _check_args("eval_task_specific"):
        config.EVAL_TASK_SPECIFIC = True
    if _check_args("freeze_backbone"):
        config.MODEL.FREEZE_BACKBONE = True
    if _check_args("adaptive_lambda"):
        config.CRLOMTL.ADAPTIVE_LAMBDA = True
    if _check_args("draw_cosine_matrix"):
        config.DRAW_COSINE_MATRIX = True
    
    if _check_args("svd_analysis"):
        config.SVD_ANALYSIS.ENABLED = True
    if _check_args("svd_analysis_freq"):
        config.SVD_ANALYSIS.FREQ = args.svd_analysis_freq
    
    if config.DATA.DATASET == "CelebA" and config.MODEL.CELEBA.USE_GROUPED_LORA:
        config.CRLOMTL.LORA_TASKS = list(CELEBA_GROUPS.keys())
    else:
        config.CRLOMTL.LORA_TASKS = list(config.TASKS)
    
        
    #lora_config_tasks = (
    #config.CRLOMTL.LORA_TASKS
    #if config.DATA.DATASET == "CelebA" and config.MODEL.CELEBA.USE_GROUPED_LORA 
    #else config.TASKS
    #    )
    lora_config_tasks = config.CRLOMTL.LORA_TASKS    
    for t in lora_config_tasks: #config.TASKS:
        if t not in config.LORA.TASK_RANK:
            config.LORA.TASK_RANK[t] = config.LORA.RANK
            config.LORA.TASK_ALPHA[t] = config.LORA.ALPHA
            config.LORA.TASK_DROPOUT[t] = config.LORA.DROPOUT
    
    if _check_args("logging_training_time"):
        config.LOGGING_TRAINING_TIME=args.logging_training_time    

    tsk_config, _ = get_tasks_config(
        config.DATA.DATASET, config.TASKS, config.DATA.IMG_SIZE
    )
    tsk_config = dict(tsk_config)
    config.TASKS_CONFIG = CN(tsk_config)
    config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT = CN(
        dict(config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT)
    )
    config.TASKS_CONFIG.ALL_TASKS.FLAGVALS = CN(
        dict(config.TASKS_CONFIG.ALL_TASKS.FLAGVALS)
    )
    config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS = CN(
        dict(config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS)
    )
    config.MODEL.NUM_CLASSES = 0

    all_loss_weights = {
        "depth": 1.0,
        "semseg": 1.0,
        "human_parts": 2.0,
        "sal": 5.0,
        "edge": 50.0,
        "normals": 10.0,
        "task_L": 1.0,
        "task_R": 1.0,
    }
    if config.DATA.DATASET == "CelebA":
        for t in config.TASKS:
            all_loss_weights[t] = 1.0

    loss_weights = {}
    for t in config.TASKS:
        loss_weights[t] = all_loss_weights[t]
    config.LOSS_WEIGHTS = CN(loss_weights)

    assert config.CRLOMTL.MODE in [
        "pretrain",
        "alternating_with_interm",
        "frozen",
    ]

    assert config.CRLOMTL.RESOLVER in resolvers.keys()

    # NativeScalerWithGradNormCount overwrites p.grad on every call (the task
    # passes rely on this), so gradients are not accumulated across steps
    assert config.TRAIN.ACCUMULATION_STEPS == 1, "gradient accumulation is not supported"

    world_size = 1

    # Following Swin linear scaling of the learning rate according to batch size
    linear_scaled_lr = (
        config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * world_size / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * world_size / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * world_size / 512.0
    )
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = (
            linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

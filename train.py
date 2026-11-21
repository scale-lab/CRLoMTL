import logging
import random
import os

import numpy as np
import torch
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import json
try:
    import wandb
except ImportError:
    wandb = None

from config import get_config
from utils import create_cmd_parser, load_checkpoint
from analysis_utils import (
    load_single_task_performance,
    calculate_multi_task_performance,
    extract_eval_dict_for_mtl_performance,
    enable_svd_analysis,
    disable_svd_analysis,
)
from setup import (
    DATASETS_LOADER,
    MODELS,
    build_dataloaders,
    build_model,
    build_optimizer_scheduler_criterion,
    setup_logging,
    setup_wandb,
)

from engine import (
    train_one_epoch,
    evaluate,
    save_checkpoint_bundle,
    save_final_checkpoint,
)

from gradient_conflict import GradientConflictMonitor


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

def save_epoch_profiling_summary(
    epoch_profile_logs,
    config,
    method,
    tag,
    logger,
    exclude_first_epoch=True,
):
    if len(epoch_profile_logs) == 0:
        logger.warning("[Profiling Summary] No profiling logs were collected.")
        return None

    profiling_dir = os.path.join(config.OUTPUT, "profiling_logs")
    os.makedirs(profiling_dir, exist_ok=True)

    profile_path = os.path.join(
        profiling_dir,
        f"{method}_{tag}_profile.json"
    )

    with open(profile_path, "w") as f:
        json.dump(epoch_profile_logs, f, indent=2)

    if exclude_first_epoch and len(epoch_profile_logs) > 1:
        valid_logs = epoch_profile_logs[1:]
        epoch_note = "excluding epoch 0"
    else:
        valid_logs = epoch_profile_logs
        epoch_note = "including all epochs"

    epoch_times = [x["train_time_sec"] for x in valid_logs]
    allocated_mems = [x["peak_allocated_mem_mb"] for x in valid_logs]
    reserved_mems = [x["peak_reserved_mem_mb"] for x in valid_logs]

    avg_time = np.mean(epoch_times)
    std_time = np.std(epoch_times)

    avg_alloc_mem = np.mean(allocated_mems)
    std_alloc_mem = np.std(allocated_mems)
    max_alloc_mem = np.max(allocated_mems)

    avg_reserved_mem = np.mean(reserved_mems)
    std_reserved_mem = np.std(reserved_mems)
    max_reserved_mem = np.max(reserved_mems)

    summary = {
        "method": method,
        "tag": tag,
        "num_epochs_logged": len(epoch_profile_logs),
        "num_epochs_used_for_average": len(valid_logs),
        "exclude_first_epoch": exclude_first_epoch,

        "avg_epoch_time_sec": float(avg_time),
        "std_epoch_time_sec": float(std_time),

        "avg_peak_allocated_mem_mb": float(avg_alloc_mem),
        "std_peak_allocated_mem_mb": float(std_alloc_mem),
        "max_peak_allocated_mem_mb": float(max_alloc_mem),

        "avg_peak_reserved_mem_mb": float(avg_reserved_mem),
        "std_peak_reserved_mem_mb": float(std_reserved_mem),
        "max_peak_reserved_mem_mb": float(max_reserved_mem),

        "profile_path": profile_path,
    }

    summary_path = os.path.join(
        profiling_dir,
        f"{method}_{tag}_profile_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"[Profiling Summary] Method: {method} | "
        f"{epoch_note} | "
        f"Avg epoch time: {avg_time:.2f} ± {std_time:.2f} sec | "
        f"Avg peak allocated memory: {avg_alloc_mem:.2f} ± {std_alloc_mem:.2f} MB | "
        f"Max peak allocated memory: {max_alloc_mem:.2f} MB | "
        f"Avg peak reserved memory: {avg_reserved_mem:.2f} ± {std_reserved_mem:.2f} MB | "
        f"Max peak reserved memory: {max_reserved_mem:.2f} MB"
    )

    logger.info(f"[Profiling Summary] Per-epoch profiling saved to: {profile_path}")
    logger.info(f"[Profiling Summary] Summary saved to: {summary_path}")

    return summary
 

def log_experiment_summary(
    logger,
    config,
    tag,
    model,
    model_name,
    dataset,
    method,
    device,
    tasks,
    adapter_enabled,
    adapter_name,
    crlomtl_mode,
    single_pass_training,
    single_pass_eval,
    eval_task_specific,
):
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # count LoRA, decoder/head and other trainable parameters separately
    lora_trainable_params = 0
    decoder_trainable_params = 0
    non_lora_trainable_params = 0
    backbone_non_lora_trainable_params = 0

    trainable_names = []
    lora_trainable_names = []
    decoder_trainable_names = []
    non_lora_trainable_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        n = p.numel()
        trainable_names.append(name)

        is_lora = (
            "lora_A" in name
            or "lora_B" in name
            or "lora" in name.lower()
        )

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
            lora_trainable_names.append(name)
        else:
            non_lora_trainable_params += n
            non_lora_trainable_names.append(name)

        if is_decoder:
            decoder_trainable_params += n
            decoder_trainable_names.append(name)

        if "backbone" in name and not is_lora:
            backbone_non_lora_trainable_params += n

    logger.info("=====================================")
    logger.info(f"tag: {tag}")
    logger.info(f"method: {method}")
    logger.info(f"model: {model_name}")
    logger.info(f"dataset: {dataset}")
    logger.info(f"batch_size: {config.DATA.BATCH_SIZE}")
    logger.info(f"learning rate: {config.TRAIN.BASE_LR}")
    logger.info(f"epochs: {config.TRAIN.EPOCHS}")
    logger.info(f"device: {device}")
    logger.info(f"seed: {config.SEED}")

    if adapter_enabled:
        logger.info(f"adapter: {adapter_name}")
    else:
        logger.info("No adapter")

    if config.MODEL.RESUME:
        logger.info(f"resume: {config.MODEL.RESUME}")

    if adapter_name in ["lora", "shared_lora"]:
        logger.info(f"lora_r: {config.LORA.RANK}")
        logger.info(f"lora_alpha: {config.LORA.ALPHA}")
        logger.info(f"lora_dropout: {config.LORA.DROPOUT}")
        logger.info(f"lora_task_r: {config.LORA.TASK_RANK}")
        logger.info(f"lora_task_alpha: {config.LORA.TASK_ALPHA}")
        logger.info(f"lora_task_dropout: {config.LORA.TASK_DROPOUT}")
        logger.info(f"lora noise factor: {config.LORA.NOISE_FACTOR}")


        if config.MODEL.RESUME_LORA:
            logger.info(f"resume_lora: {config.MODEL.RESUME_LORA}")

    else:
        logger.info("No LoRA")

    if "crlomtl" in method:
        logger.info(f"crlomtl_mode: {crlomtl_mode}")
        logger.info(f"crlomtl_resolver: {config.CRLOMTL.RESOLVER}")
        logger.info(f"crlomtl_cosine_loss: {config.CRLOMTL.COSINE_LOSS}")
        logger.info(
            f"crlomtl_cosine_loss_lambda: {config.CRLOMTL.COSINE_LOSS_LAMBDA}"
        )
        logger.info(
            f"crlomtl_mtl_loss_lambda: {config.CRLOMTL.MTL_LOSS_LAMBDA}"
        )
        logger.info(f"crlomtl_freeze_patch_embed: {config.LORA.FREEZE_PATCH_EMBED}")
        logger.info(f"crlomtl_freeze_patch_merge: {config.LORA.FREEZE_PATCH_MERGE}")
        logger.info(
            f"crlomtl_full_rank_resolver: {config.CRLOMTL.FULL_RANK_RESOLVER}"
        )
        logger.info(f"crlomtl_softmax_temperature: {config.CRLOMTL.SOFTMAX_TEMP}")
        logger.info(f"crlomtl_kernel: {config.CRLOMTL.KERNEL}")
        logger.info(f"crlomtl_atten_emb_dim: {config.CRLOMTL.ATTEN_EMB_DIM}")

    if config.EVAL_MODE:
        logger.info("Evaluation only")

    logger.info(f"single_pass_training: {single_pass_training}")
    logger.info(f"single_pass_eval: {single_pass_eval}")
    logger.info(f"eval_task_specific: {eval_task_specific}")
    logger.info(f"tasks: {tasks}")
    logger.info(f"number of parameters: {num_params:,}")
    logger.info(
        f"trainable parameters: {num_trainable_params:,} "
        f"({num_trainable_params / num_params * 100:.2f}%)"
    )

    # for shared LoRA (frozen MTL model) both of these should be 0
    logger.info("========== [TRAINABLE PARAMETER AUDIT] ==========")
    logger.info(
        f"LoRA trainable parameters: {lora_trainable_params:,} "
        f"({lora_trainable_params / num_params * 100:.4f}%)"
    )
    logger.info(
        f"Decoder/head/downsampler/MTI trainable parameters: "
        f"{decoder_trainable_params:,} "
        f"({decoder_trainable_params / num_params * 100:.4f}%)"
    )
    logger.info(
        f"Backbone non-LoRA trainable parameters: "
        f"{backbone_non_lora_trainable_params:,} "
        f"({backbone_non_lora_trainable_params / num_params * 100:.4f}%)"
    )
    logger.info(
        f"Non-LoRA trainable parameters: {non_lora_trainable_params:,} "
        f"({non_lora_trainable_params / num_params * 100:.4f}%)"
    )
    logger.info(f"Num trainable tensors: {len(trainable_names)}")
    logger.info(f"First 30 trainable tensors: {trainable_names[:30]}")
    logger.info(f"First 30 LoRA trainable tensors: {lora_trainable_names[:30]}")
    logger.info(
        f"First 30 decoder/head/downsampler/MTI trainable tensors: "
        f"{decoder_trainable_names[:30]}"
    )
    logger.info(
        f"First 30 non-LoRA trainable tensors: "
        f"{non_lora_trainable_names[:30]}"
    )
    logger.info("=================================================")

    if adapter_enabled:
        adapter_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad and "lora" in name.lower()
        )

        logger.info(
            f"Adapter parameters: {adapter_params:,} "
            f"({adapter_params / num_params * 100:.2f}%)"
        ) 
def log_trainable_layers(logger, config, model, adapter_enabled):
    if not (adapter_enabled and config.PRINT_LAYERS):
        return

    logger.info("trainable adapter parameters list:")

    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name:
            logger.info(name)

    logger.info("=================")
    logger.info("trainable non-LoRA parameters list:")

    for name, param in model.named_parameters():
        if param.requires_grad and "lora" not in name:
            logger.info(name)


def log_to_wandb_epoch(
    config,
    epoch,
    avg_loss,
    custom_losses,
    scores=None,
    loss_metrics=None,
):
    if not config.WANDB:
        return

    if wandb is None:
        return

    log_dict = {
        "train/epoch": epoch + 1,
        "train/loss": avg_loss,
        "train/cos_loss": custom_losses.get("cos_loss", None),
        "train/ft_reg_loss": custom_losses.get("ft_reg_loss", None),
    }

    if scores is not None:
        log_dict.update(scores)

    if loss_metrics is not None:
        log_dict.update(loss_metrics)

    wandb.log(log_dict)

def set_random_seed(seed):
    # python random is used by the data augmentation in data/custom_transforms.py
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_and_collect_metrics(
    config,
    model,
    val_loader,
    tasks,
    device,
    criterion,
    epoch,
    single_pass_eval,
    eval_task_specific,
    disable_progress,
    logger
):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    eval_start_time = time.time()
    meter, task_losses = evaluate(
        config=config,
        model=model,
        val_loader=val_loader,
        tasks=tasks,
        device=device,
        criterion=criterion,
        disable_progress=disable_progress,
        single_pass=single_pass_eval,
        eval_task_specific=eval_task_specific,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    eval_time_sec = time.time() - eval_start_time
    num_samples = len(val_loader.dataset)
    avg_time_per_sample_ms = (eval_time_sec / num_samples) * 1000.0
    throughput = num_samples / eval_time_sec
    print(
    f"[Inference Time] "
    f"Total eval time: {eval_time_sec:.4f} sec | "
    f"Num samples: {num_samples} | "
    f"Avg time per sample: {avg_time_per_sample_ms:.4f} ms | "
    f"Throughput: {throughput:.2f} samples/sec")

    scores = meter.get_task_scores(epoch=epoch + 1, verbose=True)

    overall_mtl_performance = None

    if config.DATA.DATASET == "CelebA":
        mean_f1 = scores.get("val/tasks/CelebA/mean_f1", 0.0)
        mean_acc = scores.get("val/tasks/CelebA/mean_acc", 0.0)

        logger.info(
            f"CelebA performance at epoch {epoch + 1}: "
            f"mean_f1={mean_f1:.6f}, "
            f"mean_acc={mean_acc:.6f}"
        )

        logger.info("========== CelebA per-task validation metrics ==========")

        for task in tasks:
            task_f1 = scores.get(f"val/tasks/{task}/f1", None)
            task_acc = scores.get(f"val/tasks/{task}/acc", None)
            task_precision = scores.get(f"val/tasks/{task}/precision", None)
            task_recall = scores.get(f"val/tasks/{task}/recall", None)

            if task_acc is None:
                logger.warning(f"[CelebA] Missing accuracy for task: {task}")
                continue

            logger.info(
                f"[CelebA task] {task}: "
                f"acc={task_acc if task_acc is not None else 0.0:.6f}, "
                f"f1={task_f1 if task_f1 is not None else 0.0:.6f}, "
                f"precision={task_precision if task_precision is not None else 0.0:.6f}, "
                f"recall={task_recall if task_recall is not None else 0.0:.6f}"
            )

        logger.info("=======================================================")

        if len(tasks) == 1:
            task = tasks[0]
            task_f1 = scores.get(f"val/tasks/{task}/f1", 0.0)
            task_acc = scores.get(f"val/tasks/{task}/acc", 0.0)

            logger.info(
                f"CelebA task {task} at epoch {epoch + 1}: "
                f"f1={task_f1:.6f}, "
                f"acc={task_acc:.6f}"
            )

    elif (single_task_dict := load_single_task_performance(config.DATA.DATASET)) is not None:
        eval_dict = extract_eval_dict_for_mtl_performance(
            scores=scores,
            tasks=tasks,
        )

        common_tasks = set(eval_dict.keys()) & set(single_task_dict.keys())

        eval_dict = {
            task: eval_dict[task]
            for task in common_tasks
        }

        single_task_eval_ref = {
            task: single_task_dict[task]
            for task in common_tasks
        }

        if len(common_tasks) > 0:
            overall_mtl_performance = calculate_multi_task_performance(
                eval_dict,
                single_task_eval_ref,
            )

            logger.info(
                f"Overall MTL performance at epoch {epoch + 1}: "
                f"{overall_mtl_performance:.6f}"
            )

        else:
            logger.warning(
                "Could not calculate overall MTL performance because "
                "no common tasks were found between eval scores and single-task baseline."
            )
            logger.warning(f"eval_dict keys: {list(eval_dict.keys())}")
            logger.warning(f"single_task_dict keys: {list(single_task_dict.keys())}")

    loss_metrics = {}

    if task_losses is not None:
        for task, loss_meter in task_losses.items():
            loss_metrics[f"val/tasks/{task}/loss"] = loss_meter.avg

    if overall_mtl_performance is not None:
        scores["val/overall_mtl_performance"] = overall_mtl_performance

    return scores, loss_metrics

def main():
    parser = create_cmd_parser(
        models=MODELS,
        methods=METHODS,
        dataset_loaders=DATASETS_LOADER,
    )

    args = parser.parse_args()
    config = get_config(args)

    dataset = config.DATA.DATASET
    method = config.METHOD
    model_name = config.MODEL.TYPE
    logging_results=config.LOGGING_TRAINING_TIME

    logger, tag, logs_save_dir = setup_logging(config)

    setup_wandb(config, tag)
    set_random_seed(config.SEED)

    assert method in METHODS, f"Method {method} not found"
    assert model_name in MODELS, f"Model {model_name} not found"
   
    os.makedirs(config.OUTPUT, exist_ok=True)
    

    train_loader, val_loader = build_dataloaders(config)


    model, device, tasks, adapter_enabled, adapter_name, lora_config = build_model(
    config=config,
    logger=logger,
    )
    assert (config.MODEL.RESUME_LORA is None or adapter_enabled), "Resume adapter without adapter enabled"

    optimizer, scheduler, criterion, loss_scaler, wmethod, use_weight_method = (
        build_optimizer_scheduler_criterion(
            config=config,
            model=model,
            train_loader=train_loader,
            tasks=tasks,
        )
    )
    if config.MODEL.RESUME:
        load_checkpoint(
        config=config,
        logger=logger,
        lora_enabled=adapter_enabled,
        model=model,
        )

    if config.MODEL.RESUME_OPTIMIZER:
        optimizer_ckp_path = config.MODEL.RESUME_OPTIMIZER
        logger.info(f"Loading optimizer state from {optimizer_ckp_path}")
        optimizer.load_state_dict(torch.load(optimizer_ckp_path)["optimizer"])

    if config.MODEL.RESUME_SCHEDULER:
        scheduler_ckp_path = config.MODEL.RESUME_SCHEDULER
        logger.info(f"Loading scheduler state from {scheduler_ckp_path}")
        scheduler.load_state_dict(torch.load(scheduler_ckp_path)["scheduler"])

    is_crlomtl = "crlomtl" in method
    crlomtl_mode = config.CRLOMTL.MODE if is_crlomtl else None

    if crlomtl_mode == "frozen":
        assert config.CRLOMTL.RESOLVER in [
            "projection",
            "projection_hold",
            "coeff",
            "coeff_soft",
            "coeff_soft_noisy",
            "matrix",
            "conv",
            "shared_subspace_b",
            "shared_subspace_norm",
        ], "Only projection & coeff resolver is supported in frozen mode"

    single_pass_training = crlomtl_mode is None or crlomtl_mode == "frozen"
    single_pass_eval = crlomtl_mode is None or crlomtl_mode != "pretrain"
    eval_task_specific = config.EVAL_TASK_SPECIFIC
    disable_progress = config.DISABLE_PROGRESS

    gradient_monitor = None
    if config.DRAW_COSINE_MATRIX:
        gradient_monitor = GradientConflictMonitor(config.TASKS)

    log_trainable_layers(
        logger=logger,
        config=config,
        model=model,
        adapter_enabled=adapter_enabled,
    )

    log_experiment_summary(
        logger=logger,
        config=config,
        tag=tag,
        model=model,
        model_name=model_name,
        dataset=dataset,
        method=method,
        device=device,
        tasks=tasks,
        adapter_enabled=adapter_enabled,
        crlomtl_mode=crlomtl_mode,
        single_pass_training=single_pass_training,
        single_pass_eval=single_pass_eval,
        eval_task_specific=eval_task_specific,
        adapter_name=adapter_name
    )

    if config.EVAL_MODE:
        logger.info("Evaluation only")

        scores, loss_metrics = evaluate_and_collect_metrics(
            config=config,
            model=model,
            val_loader=val_loader,
            tasks=tasks,
            device=device,
            criterion=criterion,
            epoch=-1,
            single_pass_eval=single_pass_eval,
            eval_task_specific=eval_task_specific,
            disable_progress=disable_progress,
            logger=logger,
        )

        if config.WANDB and wandb is not None:
            wandb.log(
                {
                    "train/epoch": 0,
                    **scores,
                    **loss_metrics,
                }
            )

        return

    epoch_profile_logs = []
    
          
    with logging_redirect_tqdm():
        for epoch in tqdm(range(config.TRAIN.EPOCHS), desc="Epochs"):
           
            epoch_peak_mem_mb = 0.0
            epoch_peak_reserved_mb = 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            epoch_start_time = time.time()  
            
            collect_svd_analysis = (
            config.SVD_ANALYSIS.ENABLED
            and is_crlomtl
            and ((epoch) % config.SVD_ANALYSIS.FREQ == 0))
            if collect_svd_analysis:  
                svd_analysis_path = os.path.join(
                    config.OUTPUT,
                    "svd_analysis",
                    f"{tag}_train_epoch_{epoch + 1}.jsonl",)

                logger.info(f"[SVD Analysis] Enabled for actual training epoch {epoch + 1}")
                logger.info(f"[SVD Analysis] Saving diagnostics to: {svd_analysis_path}")

                enable_svd_analysis(svd_analysis_path)
            try:    
                avg_loss, custom_losses = train_one_epoch(
                config=config,
                model=model,
                trainable_params=[
                    p for p in model.parameters() if p.requires_grad
                ],
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                tasks=tasks,
                method=method,
                criterion=criterion,
                epoch=epoch,
                wmethod=wmethod,
                use_weight_method=use_weight_method,
                disable_progress=disable_progress,
                crlomtl_mode=crlomtl_mode,
                single_pass=single_pass_training,
                loss_scaler=loss_scaler,
                gradient_monitor=gradient_monitor,
                logging_results=logging_results,
            )
            finally:
                if collect_svd_analysis:
                    disable_svd_analysis()
                    logger.info(f"[SVD Analysis] Disabled after training epoch {epoch + 1}")
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                epoch_peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                epoch_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            epoch_train_time_sec = time.time() - epoch_start_time

            logger.info(
            f"Epoch {epoch} training time: {epoch_train_time_sec:.2f} sec | "
            f"peak allocated memory: {epoch_peak_mem_mb:.2f} MB | "
            f"peak reserved memory: {epoch_peak_reserved_mb:.2f} MB")  
           
            epoch_profile_logs.append({
                "epoch": epoch,
                "train_time_sec": epoch_train_time_sec,
                "peak_allocated_mem_mb": epoch_peak_mem_mb,
                "peak_reserved_mem_mb": epoch_peak_reserved_mb,
                    })
            cos_loss_str = (
                ""
                if "cos_loss" not in custom_losses
                else f' | Cosine Loss {custom_losses["cos_loss"]:.3f}'
            )

            ft_reg_loss_str = (
                ""
                if "ft_reg_loss" not in custom_losses
                else f' | Feature Reg. Loss {custom_losses["ft_reg_loss"]:.3f}'
            )

            do_eval_and_save = (
                (epoch + 1) % config.EVAL_AND_SAVE_FREQ == 0
            )

            if do_eval_and_save:
                save_checkpoint_bundle(
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    tag=tag,
                    adapter_enabled=adapter_enabled,
                )

                scores, loss_metrics = evaluate_and_collect_metrics(
                    config=config,
                    model=model,
                    val_loader=val_loader,
                    tasks=tasks,
                    device=device,
                    criterion=criterion,
                    epoch=epoch,
                    single_pass_eval=single_pass_eval,
                    eval_task_specific=eval_task_specific,
                    disable_progress=disable_progress,
                    logger=logger,
                )

                logger.info(
                    f"Epoch {epoch} | Loss {avg_loss:.3f}"
                    f"{cos_loss_str}{ft_reg_loss_str}"
                )

                log_to_wandb_epoch(
                    config=config,
                    epoch=epoch,
                    avg_loss=avg_loss,
                    custom_losses=custom_losses,
                    scores=scores,
                    loss_metrics=loss_metrics,
                )

            else:
                logger.info(
                    f"Epoch {epoch} | Loss {avg_loss:.3f}"
                    f"{cos_loss_str}{ft_reg_loss_str}"
                )

                log_to_wandb_epoch(
                    config=config,
                    epoch=epoch,
                    avg_loss=avg_loss,
                    custom_losses=custom_losses,
                )
    save_epoch_profiling_summary(
            epoch_profile_logs=epoch_profile_logs,
            config=config,
            method=method,
            tag=tag,
            logger=logger,
            exclude_first_epoch=True,
            )
    # back to train mode so layers that merge LoRA in eval mode (shared LoRA)
    # are saved unmerged, otherwise ckpt_last.pth adds the LoRA update twice on reload
    model.train()
    save_final_checkpoint(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tag=tag,
        adapter_enabled=adapter_enabled,
    )

    logger.info("Training finished")


if __name__ == "__main__":
    main()
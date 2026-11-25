import math

import torch
from tqdm import tqdm

from analysis_utils import unpack_batch
from methods.cwd import CriterionCWD
from gradient_conflict import (
    compute_conflict_lambdas,
    calculate_tasks_gradient,
    draw_gradient_conflict,
)
from utils import (
    MMDLoss,
    l1_loss,
    l2_loss,
    channel_wise_normalize,
    weighted_feature_aggregation,
    normalize_feature_maps,
    ampscaler_get_grad_norm,
)
import time
import os
import json
from collections import defaultdict
from crlomtl.resolver import (
    enable_svd_timing,
    disable_svd_timing,
    get_svd_timing_records,
    clear_svd_timing_records,
)
def save_epoch_timing_json(save_path, epoch, timing_stats):
    """
    Save timing statistics for one epoch to a JSON file.

    The JSON file stores:
    - per-epoch summary
    - raw per-batch timing values
    - all previous epochs
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epoch_record = {
        "epoch": int(epoch),
        "metrics": {},
    }

    for key, values in timing_stats.items():
        if len(values) == 0:
            continue

        values = [float(v) for v in values]
        total_ms = sum(values)
        avg_ms = total_ms / len(values)

        epoch_record["metrics"][key] = {
            "avg_ms": avg_ms,
            "total_ms": total_ms,
            "total_sec": total_ms / 1000.0,
            "num_measurements": len(values),
            "values_ms": values,
        }

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "epochs": []
        }

    data["epochs"].append(epoch_record)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    return save_path





def train_one_epoch(
    config,
    model,
    trainable_params,
    train_loader,
    optimizer,
    scheduler,
    device,
    tasks,
    method,
    criterion,
    epoch,
    wmethod,
    use_weight_method,
    loss_scaler,
    disable_progress=False,
    crlomtl_mode=None,
    single_pass=True,
    gradient_monitor=None,
    logging_results=False,
):
    model.train()
    optimizer.zero_grad()

    # -------------------------------------------------------------------------
    # Timing helpers.
    # These only measure time; they do not change training logic.
    # CUDA events are used because GPU execution is asynchronous.
    # -------------------------------------------------------------------------
    timing_stats = defaultdict(list)

    # time each torch.linalg.svd call inside the shared-subspace resolver
    profile_svd = (
        hasattr(config, "CRLOMTL")
        and hasattr(config.CRLOMTL, "RESOLVER")
        and "shared_subspace" in str(config.CRLOMTL.RESOLVER)
    )

    if profile_svd:
        clear_svd_timing_records()
        enable_svd_timing()

    def cuda_timer_start():
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return start_event, end_event
        else:
            return time.perf_counter(), None

    def cuda_timer_end(timer):
        start_event, end_event = timer

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event)  # milliseconds
        else:
            return (time.perf_counter() - start_event) * 1000.0

    epoch_timer = cuda_timer_start()

    losses = []
    task_losses = []

    num_steps = len(train_loader)

    # -------------------------------------------------------------------------
    # Default behavior:
    #   NYUD/PASCAL/toy:
    #       pass_tasks == tasks
    #
    #   CelebA grouped LoRA:
    #       tasks      = 40 output attributes
    #       pass_tasks = LoRA groups, e.g. hair/facial_hair/expression/...
    # -------------------------------------------------------------------------
    pass_tasks = tasks

    if (
        config.DATA.DATASET == "CelebA"
        and hasattr(config.MODEL, "CELEBA")
        and config.MODEL.CELEBA.USE_GROUPED_LORA
    ):
        pass_tasks = list(config.CRLOMTL.LORA_TASKS)

    cos_loss_enabled = config.CRLOMTL.COSINE_LOSS
    cos_lambda = config.CRLOMTL.COSINE_LOSS_LAMBDA
    mtl_loss_lambda = config.CRLOMTL.MTL_LOSS_LAMBDA

    cos_loss = 0.0
    ft_loss = 0.0

    ft_reg = config.CRLOMTL.FEATURE_REGULARIZATION
    ft_reg_enabled = ft_reg is not None

    tasks_grads = []
    tasks_conflict_lambdas = None
    cosine_matrices_over_time = []

    assert not (
        ft_reg_enabled and cos_loss_enabled
    ), "Cannot do feature regularization with cosine loss"

    ft_reg_weight = config.CRLOMTL.FEATURE_REGULARIZATION_WEIGHT
    ft_reg_channelwise_temp = config.CRLOMTL.FEATURE_REGULARIZATION_CHANNELWISE_TEMP
    ft_reg_norm = config.CRLOMTL.FEATURE_REGULARIZATION_NORM

    ft_loss_fn = None
    unwrap_window = False
    unwrap_fn_defined = False

    def unwrap_fn(ft, ind):
        return ft

    def ft_reg_norm_fn(ft_maps):
        return ft_maps

    is_lora_reg = False
    is_last_only = False

    if ft_reg_enabled:
        if ft_reg == "l1":
            ft_loss_fn = l1_loss

        elif ft_reg == "l2" or ft_reg == "mse":
            ft_loss_fn = l2_loss

        elif ft_reg == "l2l":
            ft_loss_fn = l2_loss
            is_last_only = True

        elif ft_reg == "l2lora":
            ft_loss_fn = l2_loss
            is_lora_reg = True

            norm_task_repr = {
                task: [
                    p
                    for name, p in model.named_parameters()
                    if f"lora_A.{task}" in name or f"lora_B.{task}" in name
                ]
                for task in pass_tasks
            }

        elif ft_reg == "mmd":
            ft_loss_fn = MMDLoss()

        elif ft_reg == "cwd":
            ft_loss_fn = CriterionCWD(
                temperature=ft_reg_channelwise_temp
            )
            unwrap_window = True

        else:
            raise ValueError(f"Unknown feature regularization {ft_reg}")

        if ft_reg_norm is not None:
            if ft_reg_norm == "mean":
                ft_reg_norm_fn = channel_wise_normalize

            elif ft_reg_norm == "weighted":
                ft_reg_norm_fn = weighted_feature_aggregation

            elif ft_reg_norm == "norm":
                ft_reg_norm_fn = normalize_feature_maps

            else:
                raise ValueError(
                    f"Unknown feature regularization norm {ft_reg_norm}"
                )

    custom_losses = {}

    for idx, batch in enumerate(
        tqdm(
            train_loader,
            desc="Batches",
            leave=False,
            disable=disable_progress,
        )
    ):
        batch_timer = cuda_timer_start()

        img, labels = unpack_batch(batch, config)

        img = img.to(device, non_blocking=True)
        labels = {
            k: v.to(device, non_blocking=True)
            for k, v in labels.items()
        }

        is_update_step = (
            (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        )

        if single_pass:
            single_pass_timer = cuda_timer_start()

            loss = run_single_pass_step(
                config=config,
                model=model,
                img=img,
                labels=labels,
                tasks=tasks,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                idx=idx,
                num_steps=num_steps,
                trainable_params=trainable_params,
                loss_scaler=loss_scaler,
                wmethod=wmethod,
                use_weight_method=use_weight_method,
                is_update_step=is_update_step,
            )

            single_pass_ms = cuda_timer_end(single_pass_timer)
            timing_stats["single_pass_step_ms"].append(single_pass_ms)

            if idx == 0:
                print(f"[TIME] single-pass step: {single_pass_ms:.3f} ms")

            losses.append(loss.detach().item())

        else:
            if cos_loss_enabled or ft_reg_enabled:
                task_repr = {}

                def compute_cos_loss(ft1, ft2):
                    return (
                        1
                        - torch.nn.functional.cosine_similarity(
                            ft1,
                            ft2,
                            dim=-1,
                        ).mean()
                    )

            if crlomtl_mode is None or crlomtl_mode == "pretrain":
                pretrain_timer = cuda_timer_start()

                epoch_loss = run_task_specific_pretrain_step(
                    config=config,
                    model=model,
                    img=img,
                    labels=labels,
                    tasks=pass_tasks,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    idx=idx,
                    num_steps=num_steps,
                    trainable_params=trainable_params,
                    loss_scaler=loss_scaler,
                    is_update_step=is_update_step,
                )

                pretrain_ms = cuda_timer_end(pretrain_timer)
                timing_stats["crlomtl_pretrain_step_ms"].append(pretrain_ms)

                if idx == 0:
                    print(f"[TIME] CRLoMTL pretrain step: {pretrain_ms:.3f} ms")

                losses.append(epoch_loss)

            elif crlomtl_mode == "alternating_with_interm":
                crlomtl_step_timer = cuda_timer_start()

                task_losses = []
                tasks_grads = []

                # -------------------------------------------------------------
                # Select pass tasks for this batch.
                # -------------------------------------------------------------
                num_pass_tasks_per_batch = 0

                if hasattr(config.CRLOMTL, "NUM_PASS_TASKS_PER_BATCH"):
                    num_pass_tasks_per_batch = int(
                        config.CRLOMTL.NUM_PASS_TASKS_PER_BATCH
                    )

                use_rotating_pass_tasks = (
                    config.DATA.DATASET == "CelebA"
                    and hasattr(config.MODEL, "CELEBA")
                    and config.MODEL.CELEBA.USE_GROUPED_LORA
                    and num_pass_tasks_per_batch > 0
                    and num_pass_tasks_per_batch < len(pass_tasks)
                )

                if use_rotating_pass_tasks:
                    start = idx % len(pass_tasks)
                    selected_pass_tasks = [
                        pass_tasks[(start + j) % len(pass_tasks)]
                        for j in range(num_pass_tasks_per_batch)
                    ]
                else:
                    selected_pass_tasks = pass_tasks

                task_loop_timer = cuda_timer_start()

                for task in selected_pass_tasks:
                    if is_update_step:
                        optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                        predictions, features = model(
                            img,
                            return_representation=True,
                            task=task,
                        )

                        if cos_loss_enabled or ft_reg_enabled:
                            task_repr[task] = [
                                f.detach() for f in features
                            ]

                        task_loss, loss_dict = criterion(
                            predictions,
                            labels,
                        )

                        task_loss = (
                            task_loss
                            / config.TRAIN.ACCUMULATION_STEPS
                        )

                    if (
                        config.CRLOMTL.ADAPTIVE_LAMBDA
                        or config.DRAW_COSINE_MATRIX
                    ):
                        norm, grads_vec = loss_scaler(
                            task_loss,
                            optimizer,
                            clip_grad=config.TRAIN.CLIP_GRAD,
                            parameters=trainable_params,
                            create_graph=False,
                            update_grad=is_update_step,
                            adaptive_lambda=config.CRLOMTL.ADAPTIVE_LAMBDA,
                            draw_cosine_matrix=config.DRAW_COSINE_MATRIX,
                        )
                        tasks_grads.append(grads_vec)

                    else:
                        loss_scaler(
                            task_loss,
                            optimizer,
                            clip_grad=config.TRAIN.CLIP_GRAD,
                            parameters=trainable_params,
                            create_graph=False,
                            update_grad=is_update_step,
                        )

                    task_losses.append(task_loss.detach().item())

                task_loop_ms = cuda_timer_end(task_loop_timer)
                timing_stats["crlomtl_task_specific_loop_ms"].append(task_loop_ms)

                if idx == 0:
                    print(
                        f"[TIME] CRLoMTL task-specific loop: "
                        f"{task_loop_ms:.3f} ms "
                        f"for {len(selected_pass_tasks)} task/group passes"
                    )

                if (
                    config.CRLOMTL.ADAPTIVE_LAMBDA
                    or config.DRAW_COSINE_MATRIX
                ):
                    if tasks_grads:
                        adaptive_lambda_timer = cuda_timer_start()

                        tasks_conflict_lambdas = compute_conflict_lambdas(
                            tasks_grads,
                            eps=config.CRLOMTL.EPSILON,
                        )

                        adaptive_lambda_ms = cuda_timer_end(adaptive_lambda_timer)
                        timing_stats["crlomtl_adaptive_lambda_ms"].append(
                            adaptive_lambda_ms
                        )

                        if idx == 0:
                            print(
                                f"[TIME] CRLoMTL adaptive lambda computation: "
                                f"{adaptive_lambda_ms:.3f} ms"
                            )

                if is_update_step:
                    optimizer.zero_grad()

                global_pass_timer = cuda_timer_start()

                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    predictions, shared_ft = model(
                        img,
                        return_representation=True,
                    )

                    combined_pass_loss, loss_dict = criterion(
                        predictions,
                        labels,
                    )

                    combined_pass_loss = (
                        combined_pass_loss
                        / config.TRAIN.ACCUMULATION_STEPS
                    )

                    if cos_loss_enabled:
                        num_levels = len(shared_ft)

                        cos_loss = (
                            torch.mean(
                                torch.cat(
                                    [
                                        torch.stack(
                                            [
                                                compute_cos_loss(
                                                    task_repr[task][ind],
                                                    shared_ft[ind],
                                                )
                                                for ind in range(num_levels)
                                            ]
                                        )
                                        for task in selected_pass_tasks
                                    ]
                                )
                            )
                            / config.TRAIN.ACCUMULATION_STEPS
                        )

                    elif ft_reg_enabled:
                        num_levels = len(shared_ft)

                        if unwrap_window and not unwrap_fn_defined:
                            window_dims = [
                                int(math.sqrt(m.shape[1]))
                                for m in shared_ft
                            ]

                            def unwrap_fn(ft, ind):
                                return ft.transpose(1, 2).reshape(
                                    ft.shape[0],
                                    -1,
                                    window_dims[ind],
                                    window_dims[ind],
                                )

                            unwrap_fn_defined = True

                        if not is_lora_reg:
                            norm_shared_ft = ft_reg_norm_fn(shared_ft)

                            norm_task_repr = {
                                task: ft_reg_norm_fn(task_repr[task])
                                for task in selected_pass_tasks
                            }

                            if not is_last_only:
                                ft_loss = (
                                    torch.mean(
                                        torch.cat(
                                            [
                                                torch.stack(
                                                    [
                                                        ft_loss_fn(
                                                            unwrap_fn(
                                                                norm_task_repr[
                                                                    task
                                                                ][ind],
                                                                ind,
                                                            ),
                                                            unwrap_fn(
                                                                norm_shared_ft[
                                                                    ind
                                                                ],
                                                                ind,
                                                            ),
                                                        )
                                                        for ind in range(
                                                            num_levels
                                                        )
                                                    ]
                                                )
                                                for task in selected_pass_tasks
                                            ]
                                        )
                                    )
                                    / config.TRAIN.ACCUMULATION_STEPS
                                )

                            else:
                                ft_loss = (
                                    torch.mean(
                                        torch.cat(
                                            [
                                                torch.stack(
                                                    [
                                                        ft_loss_fn(
                                                            unwrap_fn(
                                                                norm_task_repr[
                                                                    task
                                                                ][ind],
                                                                ind,
                                                            ),
                                                            unwrap_fn(
                                                                norm_shared_ft[
                                                                    ind
                                                                ],
                                                                ind,
                                                            ),
                                                        )
                                                        for ind in [-1]
                                                    ]
                                                )
                                                for task in selected_pass_tasks
                                            ]
                                        )
                                    )
                                    / config.TRAIN.ACCUMULATION_STEPS
                                )

                        else:
                            per_task_losses = []

                            for task1 in selected_pass_tasks:
                                pairwise_losses = []

                                for task2 in selected_pass_tasks:
                                    if task1 == task2:
                                        continue

                                    loss_levels = []

                                    for ind in range(num_levels):
                                        loss_levels.append(
                                            ft_loss_fn(
                                                unwrap_fn(
                                                    norm_task_repr[task1][ind],
                                                    ind,
                                                ),
                                                unwrap_fn(
                                                    norm_task_repr[task2][ind],
                                                    ind,
                                                ),
                                            )
                                        )

                                    if len(loss_levels) > 0:
                                        pairwise_losses.append(
                                            torch.stack(loss_levels).mean()
                                        )

                                if len(pairwise_losses) > 0:
                                    per_task_losses.append(
                                        torch.stack(pairwise_losses).mean()
                                    )

                            if len(per_task_losses) > 0:
                                per_task_losses = torch.stack(per_task_losses)

                                if config.CRLOMTL.ADAPTIVE_LAMBDA:
                                    lambda_tensor = torch.tensor(
                                        tasks_conflict_lambdas,
                                        device=per_task_losses.device,
                                        dtype=per_task_losses.dtype,
                                    )

                                    weighted_loss = (
                                        lambda_tensor * per_task_losses
                                    ).sum()

                                else:
                                    weighted_loss = (
                                        per_task_losses.mean()
                                        * len(selected_pass_tasks)
                                    )

                                ft_loss = (
                                    weighted_loss
                                    / config.TRAIN.ACCUMULATION_STEPS
                                )
                            else:
                                ft_loss = torch.tensor(
                                    0.0,
                                    device=combined_pass_loss.device,
                                    dtype=combined_pass_loss.dtype,
                                )

                global_pass_ms = cuda_timer_end(global_pass_timer)
                timing_stats["crlomtl_global_pass_ms"].append(global_pass_ms)

                if idx == 0:
                    print(
                        f"[TIME] CRLoMTL global pass "
                        f"(forward + loss + reg): "
                        f"{global_pass_ms:.3f} ms"
                    )

                if config.CRLOMTL.ADAPTIVE_LAMBDA:
                    total_loss = (
                        mtl_loss_lambda * combined_pass_loss
                        + ft_loss
                    )
                else:
                    total_loss = (
                        mtl_loss_lambda * combined_pass_loss
                        + ft_reg_weight * ft_loss
                    )

                global_backward_timer = cuda_timer_start()

                if config.CRLOMTL.ADAPTIVE_LAMBDA:
                    norm, grad_vec = loss_scaler(
                        total_loss,
                        optimizer,
                        clip_grad=config.TRAIN.CLIP_GRAD,
                        parameters=trainable_params,
                        create_graph=False,
                        update_grad=is_update_step,
                        adaptive_lambda=config.CRLOMTL.ADAPTIVE_LAMBDA,
                        draw_cosine_matrix=config.DRAW_COSINE_MATRIX,
                    )
                else:
                    loss_scaler(
                        total_loss,
                        optimizer,
                        clip_grad=config.TRAIN.CLIP_GRAD,
                        parameters=trainable_params,
                        create_graph=False,
                        update_grad=is_update_step,
                    )

                global_backward_ms = cuda_timer_end(global_backward_timer)
                timing_stats["crlomtl_global_backward_ms"].append(
                    global_backward_ms
                )

                if idx == 0:
                    print(
                        f"[TIME] CRLoMTL global backward/scaler: "
                        f"{global_backward_ms:.3f} ms"
                    )

                losses.append(total_loss.detach().item())

                if is_update_step:
                    optimizer.zero_grad()
                    scheduler.step_update(epoch * num_steps + idx)

                    if ft_reg_enabled:
                        for p in trainable_params:
                            p.grad = None

                if config.DRAW_COSINE_MATRIX:
                    cosine_matrix_timer = cuda_timer_start()

                    task_grads = []

                    for task in selected_pass_tasks:
                        norm, grads_vec = calculate_tasks_gradient(
                            img,
                            model,
                            task,
                            predictions,
                            labels,
                            optimizer,
                            criterion,
                            config,
                            loss_scaler,
                            is_update_step,
                            trainable_params,
                        )

                        task_grads.append(grads_vec)

                    cosine_matrix = gradient_monitor.compute_cosine_matrix(
                        task_grads
                    )

                    cosine_matrices_over_time.append(cosine_matrix)

                    cosine_matrix_ms = cuda_timer_end(cosine_matrix_timer)
                    timing_stats["crlomtl_cosine_matrix_extra_ms"].append(
                        cosine_matrix_ms
                    )

                    if idx == 0:
                        print(
                            f"[TIME] CRLoMTL extra cosine-matrix computation: "
                            f"{cosine_matrix_ms:.3f} ms"
                        )

                crlomtl_step_ms = cuda_timer_end(crlomtl_step_timer)
                timing_stats["crlomtl_total_step_ms"].append(crlomtl_step_ms)

                if idx == 0:
                    print(
                        f"[TIME] CRLoMTL total alternating step: "
                        f"{crlomtl_step_ms:.3f} ms"
                    )

        batch_ms = cuda_timer_end(batch_timer)
        timing_stats["batch_total_ms"].append(batch_ms)

        if idx == 0:
            print(
                f"[TIME] total batch time including unpack/data transfer: "
                f"{batch_ms:.3f} ms"
            )

    if cos_loss_enabled:
        custom_losses["cos_loss"] = cos_loss

    if ft_reg_enabled:
        custom_losses["ft_reg_loss"] = ft_loss

    if config.DRAW_COSINE_MATRIX:
        draw_gradient_conflict(
            cosine_matrices_over_time,
            None,
            gradient_monitor,
            pass_tasks,
            epoch,
        )

    # collect SVD timing before the epoch summary
    svd_records = []

    if profile_svd:
        svd_records = get_svd_timing_records()
        disable_svd_timing()

        for record in svd_records:
            timing_stats["svd_ms"].append(record["elapsed_ms"])

        if len(svd_records) > 0:
            svd_by_resolver = defaultdict(list)
            svd_by_shape = defaultdict(list)

            for record in svd_records:
                svd_by_resolver[record["resolver"]].append(
                    record["elapsed_ms"]
                )

                shape_key = tuple(record["matrix_shape"])
                svd_by_shape[shape_key].append(record["elapsed_ms"])

            timing_stats["_svd_records_count"].append(len(svd_records))

            print("")
            print("---------- SVD timing details ----------")
            print(f"SVD calls this epoch: {len(svd_records)}")
            print(f"num_lora_tasks: {len(config.CRLOMTL.LORA_TASKS)}")
            print(f"resolver: {config.CRLOMTL.RESOLVER}")

            for resolver_name, values in svd_by_resolver.items():
                total = sum(values)
                avg = total / len(values)

                print(
                    f"SVD resolver={resolver_name}: "
                    f"avg={avg:.3f} ms, "
                    f"total={total:.3f} ms, "
                    f"max={max(values):.3f} ms, "
                    f"n={len(values)}"
                )

            print("Top SVD shapes by total time:")

            shape_summaries = []

            for shape, values in svd_by_shape.items():
                total = sum(values)
                avg = total / len(values)
                max_v = max(values)
                n = len(values)

                shape_summaries.append(
                    (
                        total,
                        shape,
                        n,
                        avg,
                        max_v,
                    )
                )

            shape_summaries = sorted(shape_summaries, reverse=True)[:10]

            for total, shape, n, avg, max_v in shape_summaries:
                print(
                    f"  shape={shape}: "
                    f"total={total:.3f} ms, "
                    f"avg={avg:.3f} ms, "
                    f"max={max_v:.3f} ms, "
                    f"n={n}"
                )

    # -------------------------------------------------------------------------
    # Epoch-level timing summary.
    # -------------------------------------------------------------------------
    epoch_total_ms = cuda_timer_end(epoch_timer)

    print("")
    print("========== [TIME SUMMARY] ==========")
    print(f"epoch_total_time: {epoch_total_ms:.3f} ms")
    print(f"epoch_total_time: {epoch_total_ms / 1000.0:.3f} sec")
    print("")

    for key, values in timing_stats.items():
        # Internal bookkeeping keys should not be printed as timing values.
        if key.startswith("_"):
            continue

        if len(values) == 0:
            continue

        total_ms = sum(values)
        avg_ms = total_ms / len(values)
        min_ms = min(values)
        max_ms = max(values)
        epoch_percent = 100.0 * total_ms / max(epoch_total_ms, 1e-8)

        print(
            f"{key}: "
            f"avg={avg_ms:.3f} ms, "
            f"total={total_ms:.3f} ms, "
            f"epoch%={epoch_percent:.2f}%, "
            f"min={min_ms:.3f} ms, "
            f"max={max_ms:.3f} ms, "
            f"n={len(values)}"
        )

    if (
        "crlomtl_task_specific_loop_ms" in timing_stats
        and "crlomtl_global_pass_ms" in timing_stats
    ):
        total_task_loop_ms = sum(
            timing_stats["crlomtl_task_specific_loop_ms"]
        )
        total_global_pass_ms = sum(
            timing_stats["crlomtl_global_pass_ms"]
        )

        avg_task_loop_ms = (
            total_task_loop_ms
            / len(timing_stats["crlomtl_task_specific_loop_ms"])
        )
        avg_global_pass_ms = (
            total_global_pass_ms
            / len(timing_stats["crlomtl_global_pass_ms"])
        )

        core_ms = total_task_loop_ms + total_global_pass_ms

        print("")
        print("---------- CRLoMTL core bottleneck breakdown ----------")
        print(f"Avg task-specific loop / batch: {avg_task_loop_ms:.3f} ms")
        print(f"Avg global pass / batch:        {avg_global_pass_ms:.3f} ms")
        print(
            f"Task-loop/global-pass ratio:    "
            f"{avg_task_loop_ms / max(avg_global_pass_ms, 1e-8):.3f}x"
        )
        print("")
        print(f"Total task-specific loop time:  {total_task_loop_ms:.3f} ms")
        print(f"Total global pass time:         {total_global_pass_ms:.3f} ms")
        print(
            f"Task-loop share of core:        "
            f"{100.0 * total_task_loop_ms / max(core_ms, 1e-8):.2f}%"
        )
        print(
            f"Global-pass share of core:      "
            f"{100.0 * total_global_pass_ms / max(core_ms, 1e-8):.2f}%"
        )

    if "crlomtl_total_step_ms" in timing_stats:
        total_crlomtl_step_ms = sum(timing_stats["crlomtl_total_step_ms"])
        avg_crlomtl_step_ms = (
            total_crlomtl_step_ms
            / len(timing_stats["crlomtl_total_step_ms"])
        )

        print("")
        print("---------- CRLoMTL total step ----------")
        print(f"Avg CRLoMTL total step / batch: {avg_crlomtl_step_ms:.3f} ms")
        print(f"Total CRLoMTL step time:        {total_crlomtl_step_ms:.3f} ms")

        if (
            "crlomtl_task_specific_loop_ms" in timing_stats
            and "crlomtl_global_pass_ms" in timing_stats
            and "crlomtl_global_backward_ms" in timing_stats
        ):
            total_task_loop_ms = sum(
                timing_stats["crlomtl_task_specific_loop_ms"]
            )
            total_global_pass_ms = sum(
                timing_stats["crlomtl_global_pass_ms"]
            )
            total_global_backward_ms = sum(
                timing_stats["crlomtl_global_backward_ms"]
            )

            other_ms = (
                total_crlomtl_step_ms
                - total_task_loop_ms
                - total_global_pass_ms
                - total_global_backward_ms
            )

            print("")
            print("---------- CRLoMTL total-step composition ----------")
            print(
                f"Task-specific loop percentage:  "
                f"{100.0 * total_task_loop_ms / max(total_crlomtl_step_ms, 1e-8):.2f}%"
            )
            print(
                f"Global pass percentage:         "
                f"{100.0 * total_global_pass_ms / max(total_crlomtl_step_ms, 1e-8):.2f}%"
            )
            print(
                f"Global backward percentage:     "
                f"{100.0 * total_global_backward_ms / max(total_crlomtl_step_ms, 1e-8):.2f}%"
            )
            print(
                f"Other overhead percentage:      "
                f"{100.0 * other_ms / max(total_crlomtl_step_ms, 1e-8):.2f}%"
            )

    if "single_pass_step_ms" in timing_stats:
        total_single_pass_ms = sum(timing_stats["single_pass_step_ms"])
        avg_single_pass_ms = (
            total_single_pass_ms
            / len(timing_stats["single_pass_step_ms"])
        )

        print("")
        print("---------- Single-pass timing ----------")
        print(f"Avg single-pass step / batch:   {avg_single_pass_ms:.3f} ms")
        print(f"Total single-pass time:         {total_single_pass_ms:.3f} ms")
        print(
            f"Single-pass share of epoch:     "
            f"{100.0 * total_single_pass_ms / max(epoch_total_ms, 1e-8):.2f}%"
        )

    # share of the epoch time spent in SVD
    if "svd_ms" in timing_stats:
        total_svd_ms = sum(timing_stats["svd_ms"])

        print("")
        print("---------- SVD bottleneck conclusion ----------")
        print(f"Total SVD time:                 {total_svd_ms:.3f} ms")
        print(
            f"SVD share of epoch:             "
            f"{100.0 * total_svd_ms / max(epoch_total_ms, 1e-8):.2f}%"
        )

    print("========================================")
    print("")

    if logging_results:
        timing_save_path = os.path.join(
            config.OUTPUT,
            "crlomtl_timing_per_epoch.json",
        )

        save_epoch_timing_json(
            save_path=timing_save_path,
            epoch=epoch,
            timing_stats=timing_stats,
        )

        print(f"[TIME] Saved epoch timing to: {timing_save_path}")

        # save every SVD call with its matrix shape
        if profile_svd:
            svd_records_path = os.path.join(
                config.OUTPUT,
                "svd_timing_records.json",
            )

            if os.path.exists(svd_records_path):
                with open(svd_records_path, "r") as f:
                    svd_data = json.load(f)
            else:
                svd_data = {"epochs": []}

            svd_data["epochs"].append(
                {
                    "epoch": int(epoch),
                    "resolver": str(config.CRLOMTL.RESOLVER),
                    "num_lora_tasks": int(len(config.CRLOMTL.LORA_TASKS)),
                    "records": svd_records,
                }
            )

            with open(svd_records_path, "w") as f:
                json.dump(svd_data, f, indent=2)

            print(f"[TIME] Saved SVD timing records to: {svd_records_path}")

    return sum(losses) / len(losses), custom_losses

 

# weight methods that the original main.py already had; they keep its exact
# GradScaler path so the released code gives the published baseline numbers
LEGACY_WEIGHT_METHODS = {"pcgrad", "cagrad", "nashmtl", "mgda"}


def run_single_pass_step(
    config,
    model,
    img,
    labels,
    tasks,
    criterion,
    optimizer,
    scheduler,
    epoch,
    idx,
    num_steps,
    trainable_params,
    loss_scaler,
    wmethod,
    use_weight_method,
    is_update_step,
):
    """
    One single-pass MTL training step.

    FairGrad-style behavior:
    - pass torch.stack task losses
    - call wmethod.backward(...)
    - use optimizer.step(), not GradScaler step
    - if FairGrad returns None, keep original criterion loss for logging
    """

    method_name = str(config.METHOD).lower()

    amp_sensitive_methods = {
        "fairgrad",
        "imtl",
        "consmtl",
        "famo",
        "go4align",
    }

    amp_enabled = bool(
        config.AMP_ENABLE
        and method_name not in amp_sensitive_methods
    )

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        predictions, features = model(
            img,
            return_representation=True,
        )

        loss, loss_dict = criterion(predictions, labels)

        loss = loss / config.TRAIN.ACCUMULATION_STEPS

    if use_weight_method and method_name in LEGACY_WEIGHT_METHODS:
        # same steps as the original main.py used for the paper baselines:
        # GradScaler-scaled task losses go into the weight method, then unscale,
        # clip and step through the scaler
        scaler = loss_scaler._scaler
        wmethod.backward(
            losses=[scaler.scale(loss_dict[task]) for task in tasks],
            shared_parameters=[p for p in model.shared_parameters() if p.requires_grad],
            task_specific_parameters=[p for p in model.task_specific_parameters() if p.requires_grad],
            last_shared_parameters=model.last_shared_parameters(),
            representation=features,
        )
        if is_update_step:
            scaler.unscale_(optimizer)
            if config.TRAIN.CLIP_GRAD is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.TRAIN.CLIP_GRAD)
            else:
                ampscaler_get_grad_norm(trainable_params)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step_update(epoch * num_steps + idx)

    elif use_weight_method:
        if method_name == "consmtl":
            task_specific_params = model.task_specific_parameters_grouped(tasks)
        else:
            task_specific_params = [
                p for p in model.task_specific_parameters()
                if p.requires_grad
            ]

        shared_params = [
            p for p in model.shared_parameters()
            if p.requires_grad
        ]

        wm_losses = torch.stack([
            loss_dict[task] / config.TRAIN.ACCUMULATION_STEPS
            for task in tasks
        ])

        new_loss, extra_outputs = wmethod.backward(
            losses=wm_losses,
            shared_parameters=shared_params,
            task_specific_parameters=task_specific_params,
            last_shared_parameters=model.last_shared_parameters(),
            representation=features,
        )

        # Some methods, especially FairGrad, may return None
        # because they directly overwrite gradients.
        if new_loss is not None:
            loss = new_loss

        if is_update_step:
            clip_grad = config.TRAIN.CLIP_GRAD

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    clip_grad,
                )
            else:
                ampscaler_get_grad_norm(trainable_params)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step_update(epoch * num_steps + idx)

    else:
        loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=trainable_params,
            create_graph=False,
            update_grad=is_update_step,
        )

        if is_update_step:
            optimizer.zero_grad()
            scheduler.step_update(epoch * num_steps + idx)

    return loss

def run_task_specific_pretrain_step(
    config,
    model,
    img,
    labels,
    tasks,
    criterion,
    optimizer,
    scheduler,
    epoch,
    idx,
    num_steps,
    trainable_params,
    loss_scaler,
    is_update_step,
):
    """
    Task-specific pretraining step used when crlomtl_mode is None or 'pretrain'.
    """

    task_losses = []

    for task in tasks:
        if is_update_step:
            optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            predictions, features = model(
                img,
                return_representation=True,
                task=task,
            )

            loss, loss_dict = criterion(predictions, labels)

            loss = loss / config.TRAIN.ACCUMULATION_STEPS

        loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=trainable_params,
            create_graph=False,
            update_grad=is_update_step,
        )

        task_losses.append(loss.detach().item())

    if is_update_step:
        optimizer.zero_grad()
        scheduler.step_update(epoch * num_steps + idx)

    return sum(task_losses) / len(tasks)
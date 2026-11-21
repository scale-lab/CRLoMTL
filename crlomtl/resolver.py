import torch
import torch.nn.functional as F

from analysis_utils import (
    is_svd_analysis_enabled,
    next_svd_call_id,
    cosine_stats,
    dump_svd_record,
    per_task_cosine_stats
)

import time
import torch

SVD_TIMING = {
    "enabled": False,
    "records": [],
}


def enable_svd_timing():
    SVD_TIMING["enabled"] = True
    SVD_TIMING["records"] = []


def disable_svd_timing():
    SVD_TIMING["enabled"] = False


def get_svd_timing_records():
    return list(SVD_TIMING["records"])


def clear_svd_timing_records():
    SVD_TIMING["records"] = []


def _svd_sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def record_svd_timing(
    resolver_name,
    matrix_shape,
    num_tasks,
    rank,
    elapsed_ms,
    device,
):
    if not SVD_TIMING["enabled"]:
        return

    SVD_TIMING["records"].append(
        {
            "resolver": str(resolver_name),
            "matrix_shape": list(matrix_shape),
            "num_tasks": int(num_tasks),
            "rank": int(rank),
            "elapsed_ms": float(elapsed_ms),
            "device": str(device),
        }
    )

def project_tensors_svd(matrices, direction_matrix):
    u, _, _ = torch.svd(direction_matrix)
    norm_direction_matrix = u
    combined_projection = torch.zeros_like(matrices[0])

    for matrix in matrices:
        projection = torch.mm(matrix, norm_direction_matrix)
        combined_projection += projection
    return combined_projection / len(matrices)


def project_tensors_serial(matrices, direction_matrix):
    projected_sets = []
    for mat in matrices:
        projections = (
            torch.sum(mat * direction_matrix, dim=1, keepdim=True)
            / torch.sum(direction_matrix * direction_matrix, dim=1, keepdim=True)
            * direction_matrix
        )
        projected_sets.append(mat - projections)
    average_projection = sum(projected_sets) / len(projected_sets)
    return average_projection


def project_tensors(matrices, direction_matrix):
    matrices_stack = torch.stack(matrices)
    numerator = torch.sum(
        matrices_stack * direction_matrix, dim=2, keepdim=True)
    denominator = torch.sum(
        direction_matrix * direction_matrix, dim=1, keepdim=True)
    projections = (numerator / denominator) * direction_matrix.unsqueeze(0)
    result = matrices_stack - projections
    average_projection = torch.mean(result, dim=0)
    return average_projection


def svd_merge_v1(input_mats, low_r=2):
    num_tasks = len(input_mats)
    m, n = input_mats[0].shape
    stacked = torch.stack(input_mats).reshape(num_tasks, -1).t()
    U, S, V = torch.svd_lowrank(stacked, q=num_tasks)
    U_reduced = U[:, :low_r]
    S_reduced = torch.diag_embed(S[:low_r])
    V_reduced = V[:, :low_r].t()
    combined = U_reduced @ S_reduced @ V_reduced
    return combined.mean(dim=1).reshape(m, n)


def svd_merge_v2(mats_A, mats_B, low_r=2):
    num_tasks = len(mats_A)
    r1, m = mats_A[0].shape
    n, r2 = mats_B[0].shape
    assert r1 == r2
    r = len(mats_A)
    reconstr = [B @ A for A, B in zip(mats_A, mats_B)]
    stacked = torch.stack(reconstr).reshape(num_tasks, -1).t()
    U, S, V = torch.svd_lowrank(stacked, q=r)
    U_reduced = U[:, :low_r]
    S_reduced = torch.diag_embed(S[:low_r])
    V_reduced = V[:, :low_r].t()

    return (U_reduced @ S_reduced @ V_reduced).mean(dim=1).reshape(m, n)


def svd_merge_v3(input_mats, low_r=2):
    num_tasks = len(input_mats)
    U_mean = torch.zeros(
        *input_mats[0].shape[:-1], low_r).to(input_mats[0].device)
    Sigma_mean = torch.zeros(*input_mats[0].shape[:-2], low_r, low_r).to(
        input_mats[0].device
    )
    V_mean = (
        torch.zeros(input_mats[0].shape[1], low_r).to(input_mats[0].device)
        if len(input_mats[0].shape) == 2
        else torch.zeros(input_mats[0].shape[0], input_mats[0].shape[2], low_r).to(
            input_mats[0].device
        )
    )

    for A in input_mats:
        U, S, V = torch.svd_lowrank(A, q=low_r)
        U_mean += U
        Sigma_mean += torch.diag_embed(S)
        V_mean += V

    U_mean /= num_tasks
    Sigma_mean /= num_tasks
    V_mean /= num_tasks

    if len(U_mean.shape) == 2:
        result = U_mean[:, :low_r] @ Sigma_mean[:low_r,
                                                :low_r] @ V_mean[:low_r, :]
    else:
        result = torch.bmm(
            U_mean[:, :, :low_r],
            torch.bmm(
                Sigma_mean[:, :low_r, :low_r], V_mean[:,
                                                      :, :low_r].transpose(1, 2)
            ),
        )
    return result


def svd_resolver_v1(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    assert not full_rank, "SVD resolver v1 does not support full rank"
    lora_A = [lora_A[task] for task in tasks]
    lora_B = [lora_B[task] for task in tasks]
    with torch.cuda.amp.autocast(enabled=False):
        result = svd_merge_v1(lora_B) @ svd_merge_v1(lora_A)
    return result


def svd_resolver_v2(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    assert not full_rank, "SVD resolver v2 does not support full rank"
    lora_A = [lora_A[task] for task in tasks]
    lora_B = [lora_B[task] for task in tasks]
    with torch.cuda.amp.autocast(enabled=False):
        result = svd_merge_v2(lora_A, lora_B).t()
    return result


def svd_mean_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    if not full_rank:
        lora_A = [lora_A[task].t() for task in tasks]
        lora_B = [lora_B[task] for task in tasks]
        with torch.cuda.amp.autocast(enabled=False):
            result = svd_merge_v3(lora_B) @ svd_merge_v3(lora_A).t()
        return result
    else:
        with torch.cuda.amp.autocast(enabled=False):
            result = svd_merge_v3(
                [
                    x.to(lora_A[tasks[0]].dtype)
                    @ lora_A[task].transpose(0, 1)
                    @ lora_B[task].transpose(0, 1)
                    for task in tasks
                ],
                low_r=8,
            )
        return result


def projection_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    if not full_rank:
        with torch.cuda.amp.autocast(enabled=False):
            result = (
                project_tensors([lora_B[task] for task in tasks], direction_B)
                @ project_tensors([lora_A[task].T for task in tasks], direction_A.T).T
            )
        return result
    else:
        with torch.cuda.amp.autocast(enabled=False):
            x = x.to(lora_A[tasks[0]].dtype)
            result = project_tensors(
                [x @ lora_A[task].T @ lora_B[task].T for task in tasks], direction_A
            )
        return result


def projection_hold_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    if not full_rank:
        with torch.cuda.amp.autocast(enabled=False):
            result = (
                project_tensors([lora_B[task] for task in tasks], direction_B)
                @ project_tensors([lora_A[task].T for task in tasks], direction_A.T).T
            )
        return result
    else:
        with torch.cuda.amp.autocast(enabled=False):
            x = x.to(lora_A[tasks[0]].dtype)
            result = project_tensors(
                [x @ lora_A[task].T @ lora_B[task].T for task in tasks], direction_A
            )
        return result


def matrix_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    assert not full_rank, "Matrix resolver does not support full rank"
    return direction_B @ direction_A


def conv_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    assert not full_rank, "conv resolver does not support full rank"
    fused_A = direction_A(torch.stack([lora_A[t] for t in tasks]))
    fused_B = direction_B(torch.stack(
        [lora_B[t] for t in tasks]).transpose(1, 2)).t()
    return fused_B @ fused_A


def rbf_v1_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    assert not full_rank, "rbf_v1 resolver does not support full rank"

    # gamma = torch.nn.Parameter(torch.tensor(1.0)).to(x.device)
    def _rbf_kernel(x, y):
        return torch.exp(-1.0 * torch.norm(x - y, dim=-1)**2)

    similarities = _rbf_kernel(
        direction_B.unsqueeze(0),
        direction_A
    )

    weights = F.softmax(similarities, dim=0)

    combined_delta = torch.zeros(
        lora_A[tasks[0]].shape[1], lora_B[tasks[0]].shape[0]).to(x.device)
    for i, task in enumerate(tasks):
        delta = lora_A[task].T @ lora_B[task].T
        combined_delta += weights[i] * delta
    return combined_delta.T


def mean_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
  
    if not full_rank:
        lora_A = torch.stack([lora_A[task] for task in tasks]).mean(dim=0)
        lora_B = torch.stack([lora_B[task] for task in tasks]).mean(dim=0)
        return lora_B @ lora_A
    else:
        return torch.stack(
            [
                x @ lora_A[task].transpose(0, 1) @ lora_B[task].transpose(0, 1)
                for task in tasks
            ]
        ).mean(dim=0)


def coeff_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    if not full_rank:
        lora_A = torch.stack([lora_A[task] * coeff_A[task] for task in tasks])
        lora_B = torch.stack([lora_B[task] * coeff_B[task] for task in tasks])
        return lora_B.mean(dim=0) @ lora_A.mean(dim=0)
    else:
        return torch.stack(
            [
                x
                @ lora_A[task].transpose(0, 1)
                @ lora_B[task].transpose(0, 1)
                * coeff_A[task]
                for task in tasks
            ]
        ).mean(dim=0)


def coeff_soft_noisy_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type
):
    noise_A = (
        {task: torch.randn_like(coeff_A[task])
         * noise_factor for task in tasks}
        if training
        else {task: 0 for task in tasks}
    )
    coeff_A = F.softmax(
        torch.stack([coeff_A[task] + noise_A[task] for task in tasks]), dim=0
    ).unsqueeze(-1)
    if not full_rank:
        noise_B = (
            {task: torch.randn_like(
                coeff_B[task]) * noise_factor for task in tasks}
            if training
            else {task: 0 for task in tasks}
        )
        coeff_B = F.softmax(
            torch.stack([coeff_B[task] + noise_B[task] for task in tasks]), dim=0
        ).unsqueeze(-1)
        lora_A = torch.stack([lora_A[task] for task in tasks])
        lora_B = torch.stack([lora_B[task] for task in tasks])
        return (lora_B * coeff_B).sum(dim=0) @ (lora_A * coeff_A).sum(dim=0)
    else:
        return (
            torch.stack(
                [
                    x @ lora_A[task].transpose(0,1) @ lora_B[task].transpose(0, 1)
                    for task in tasks
                ]
            )
            * coeff_A.unsqueeze(-1)
        ).mean(dim=0)


def coeff_soft_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    coeff_A,
    coeff_B,
    direction_A=None,
    direction_B=None,
    training=True,
    noise_factor=0.0,
    scalers=None,
    temperature=1.0,
    atten_emb_dim=64,
    kernel_type='rbf'

):
    if scalers is None:
        scalers = {task: 1.0 for task in tasks}

    scaled_coeff_A = torch.stack(
        [coeff_A[task] * scalers.get(task, 1.0) for task in tasks])
    coeff_A_normalized = F.softmax(
        scaled_coeff_A / temperature, dim=0).unsqueeze(-1)

    if not full_rank:
        scaled_coeff_B = torch.stack(
            [coeff_B[task] * scalers.get(task, 1.0) for task in tasks])
        coeff_B_normalized = F.softmax(
            scaled_coeff_B / temperature, dim=0).unsqueeze(-1)

        stacked_lora_A = torch.stack([lora_A[task] for task in tasks])
        stacked_lora_B = torch.stack([lora_B[task] for task in tasks])

        weighted_A = (stacked_lora_A * coeff_A_normalized).sum(dim=0)
        weighted_B = (stacked_lora_B * coeff_B_normalized).sum(dim=0)

        return weighted_B @ weighted_A
    else:
        adaptations = []
        for i, task in enumerate(tasks):
            task_adaptation = x @ lora_A[task].transpose(
                0, 1) @ lora_B[task].transpose(0, 1)
            adaptations.append(task_adaptation * coeff_A_normalized[i])

        return torch.stack(adaptations).sum(dim=0)


# Kernel-based LoRA combination approach
def kernel_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    coeff_A,
    coeff_B,
    direction_A=None,
    direction_B=None,
    training=True,
    noise_factor=0.0,
    kernel_type="rbf",
    kernel_param=1.0,
    temperature=1.0,
    atten_emb_dim=64,
):
    stacked_coeff_A = torch.stack([coeff_A[task] for task in tasks])
    normalized_coeff_A = F.softmax(stacked_coeff_A, dim=0)

    stacked_lora_A = torch.stack([lora_A[task] for task in tasks])
    stacked_lora_B = torch.stack([lora_B[task] for task in tasks])
    num_tasks = len(tasks)
    task_similarities = torch.zeros(
        (num_tasks, num_tasks), device=stacked_lora_A.device)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if kernel_type == "rbf":
                diff_A = lora_A[tasks[i]] - lora_A[tasks[j]]
                diff_B = lora_B[tasks[i]] - lora_B[tasks[j]]
                dist_A = torch.sum(diff_A ** 2)
                dist_B = torch.sum(diff_B ** 2)
                similarity = torch.exp(-kernel_param * (dist_A + dist_B))
            elif kernel_type == "poly":
                dot_A = torch.sum(lora_A[tasks[i]] * lora_A[tasks[j]])
                dot_B = torch.sum(lora_B[tasks[i]] * lora_B[tasks[j]])
                similarity = (kernel_param * (dot_A + dot_B) + 1.0) ** 2
            elif kernel_type == "sigmoid":
                dot_A = torch.sum(lora_A[tasks[i]] * lora_A[tasks[j]])
                dot_B = torch.sum(lora_B[tasks[i]] * lora_B[tasks[j]])
                similarity = torch.tanh(kernel_param * (dot_A + dot_B))
            else:
                raise ValueError(f"Unsupported kernel type: {kernel_type}")

            task_similarities[i, j] = similarity

    kernel_weights = F.softmax(task_similarities.mean(dim=1), dim=0)

    combined_weights = (normalized_coeff_A.squeeze() *
                        kernel_weights).unsqueeze(-1).unsqueeze(-1)

    if not full_rank:
        stacked_coeff_B = torch.stack([coeff_B[task] for task in tasks])
        normalized_coeff_B = F.softmax(stacked_coeff_B, dim=0)

        combined_weights_B = (normalized_coeff_B.squeeze()
                              * kernel_weights).unsqueeze(-1).unsqueeze(-1)

        weighted_A = (stacked_lora_A * combined_weights).sum(dim=0)
        weighted_B = (stacked_lora_B * combined_weights_B).sum(dim=0)

        return weighted_B @ weighted_A
    else:
        adaptations = []
        weights = combined_weights.squeeze()

        for i, task in enumerate(tasks):
            task_adaptation = x @ lora_A[task].transpose(
                0, 1) @ lora_B[task].transpose(0, 1)
            adaptations.append(task_adaptation * weights[i])

        return torch.stack(adaptations).sum(dim=0)



class MoEFusion(torch.nn.Module):
    def __init__(self, d_model: int, num_tasks: int, noise_factor: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_tasks = num_tasks
        self.router = torch.nn.Linear(d_model, num_tasks, bias=False)
        self.noise_factor = noise_factor

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x.mean(dim=(2, 3))
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError("Unsupported input rank for MoEFusion")

    def forward(
        self,
        x: torch.Tensor,
        lora_A,
        lora_B,
        tasks,
        *,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if len(lora_A) != self.num_tasks or len(lora_B) != self.num_tasks:
            raise ValueError(
                "Router num_tasks mismatch with provided adapters")

        pooled = self._pool(x)
        logits = self.router(pooled)
        if self.training and self.noise_factor > 0:
            logits = logits + torch.randn_like(logits) * self.noise_factor
        gate = F.softmax(logits, dim=-1)
        w = gate.mean(dim=0)


        A_stack = torch.stack([lora_A[task] for task in tasks])
        B_stack = torch.stack([lora_B[task] for task in tasks])

        w_exp = w[:, None, None]
        A_fused = torch.sum(A_stack * w_exp, dim=0)
        B_fused = torch.sum(B_stack * w_exp, dim=0)

        W_fused = B_fused @ A_fused
        if alpha != 1.0:
            W_fused = alpha * W_fused
        return W_fused


def moe_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    moe_module,
    **kwargs
):
    assert not full_rank
    assert moe_module is not None, "moe_module must be provided for moe_resolver"
    return moe_module(x, lora_A, lora_B, tasks)



def shared_subspace_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    rank=None,
    rank_ratio=None,
    debug_analysis=False,
):
    assert not full_rank
    do_analysis = debug_analysis or is_svd_analysis_enabled()
    base_A = lora_A[tasks[0]]
    r_task, d_in = base_A.shape
    # r_s = r/2 by default, CRLOMTL.SHARED_RANK_RATIO changes it (r_s ablation)
    if rank is None and rank_ratio is not None:
        rank = max(1, int(r_task * rank_ratio))
    if rank is None:
        rank = max(1, r_task//2)
    rank = min(rank, d_in)

    A_list = [lora_A[t] for t in tasks]
    B_list = [lora_B[t] for t in tasks]
    out_dim = B_list[0].shape[0]
    A_cat = torch.cat([A.T for A in A_list], dim=1).float()

    # S is kept for the singular value / energy analysis
    t0 = _svd_sync_time()
    U, S, _ = torch.linalg.svd(A_cat, full_matrices=False)
    t1 = _svd_sync_time()
    record_svd_timing(
    resolver_name="shared_subspace_A",
    matrix_shape=A_cat.shape,
    num_tasks=len(tasks),
    rank=rank,
    elapsed_ms=(t1 - t0) * 1000.0,
    device=A_cat.device)
    A_shared = U[:, :rank].T

    target_dtype = base_A.dtype
    if A_shared.dtype != target_dtype:
        A_shared = A_shared.to(target_dtype)

    B_acc = torch.zeros(
        out_dim,
        rank,
        dtype=target_dtype,
        device=A_shared.device,
    )

    if do_analysis:
        Delta_list = []
        Kept_list = []
        Residual_list = []

    for A_t, B_t in zip(A_list, B_list):
        full_update = B_t @ A_t
        full_update = full_update.to(target_dtype)

        X = torch.linalg.lstsq(
            A_shared.T,
            full_update.T,
        ).solution.T

        if X.shape[1] < rank:
            pad_rank = rank - X.shape[1]
            X = torch.cat(
                [X,
                torch.zeros(
                        out_dim,
                        pad_rank,
                        dtype=target_dtype,
                        device=X.device,
                    ),
                ],
                dim=1,
            )

        B_acc += X

        if do_analysis:
            kept_update = X @ A_shared
            residual_update = full_update - kept_update
            Delta_list.append(full_update.detach().flatten().float())
            Kept_list.append(kept_update.detach().flatten().float())
            Residual_list.append(residual_update.detach().flatten().float())

    B_shared = B_acc / len(B_list)
    fused = B_shared @ A_shared

    if do_analysis:
        Delta = torch.stack(Delta_list, dim=0)
        Kept = torch.stack(Kept_list, dim=0)
        Residual = torch.stack(Residual_list, dim=0)

        avg_delta, npr_delta, cos_delta = cosine_stats(Delta)
        avg_kept, npr_kept, cos_kept = cosine_stats(Kept)
        avg_res, npr_res, cos_res = cosine_stats(Residual)

        per_task_original = per_task_cosine_stats(cos_delta, tasks)
        per_task_kept = per_task_cosine_stats(cos_kept, tasks)
        per_task_residual = per_task_cosine_stats(cos_res, tasks)

        total_energy = torch.norm(Delta, p="fro") ** 2
        kept_energy = torch.norm(Kept, p="fro") ** 2
        residual_energy = torch.norm(Residual, p="fro") ** 2

        sv_energy = S.detach().float() ** 2
        cumulative_energy = torch.cumsum(sv_energy, dim=0) / sv_energy.sum()

        # -----------------------------------------------------
        # Merged LoRA diagnostics
        # -----------------------------------------------------
        fused_vec = fused.detach().flatten().float().unsqueeze(0)  # [1, D]

        Delta_norm = torch.nn.functional.normalize(Delta, dim=1)
        Kept_norm = torch.nn.functional.normalize(Kept, dim=1)
        Residual_norm = torch.nn.functional.normalize(Residual, dim=1)
        fused_norm = torch.nn.functional.normalize(fused_vec, dim=1)

        # Cosine between merged LoRA and each task update
        cos_fused_original = (Delta_norm @ fused_norm.T).squeeze(1)
        cos_fused_kept = (Kept_norm @ fused_norm.T).squeeze(1)
        cos_fused_residual = (Residual_norm @ fused_norm.T).squeeze(1)

        # Distances between merged LoRA and each task update
        dist_fused_original = torch.norm(Delta - fused_vec, dim=1)
        dist_fused_kept = torch.norm(Kept - fused_vec, dim=1)

        record = {
            "call_id": next_svd_call_id(),
            "rank": int(rank),
            "num_tasks": int(len(tasks)),
            "tasks": list(tasks),

            # Pairwise task-update alignment
            "avg_cos_original": float(avg_delta),
            "avg_cos_kept": float(avg_kept),
            "avg_cos_residual": float(avg_res),

            "npr_original": float(npr_delta),
            "npr_kept": float(npr_kept),
            "npr_residual": float(npr_res),

            "per_task_original": per_task_original,
            "per_task_kept": per_task_kept,
            "per_task_residual": per_task_residual,

            # Energy retained/removed
            "energy_original": 1.0,
            "energy_kept": float((kept_energy / total_energy).item()),
            "energy_residual": float((residual_energy / total_energy).item()),

            # SVD spectrum
            "singular_values": S.detach().float().cpu().tolist(),
            "cumulative_energy": cumulative_energy.detach().cpu().tolist(),

            # Heatmaps for task-task cosine
            "cos_original": cos_delta.detach().float().cpu().tolist(),
            "cos_kept": cos_kept.detach().float().cpu().tolist(),
            "cos_residual": cos_res.detach().float().cpu().tolist(),

            # Merged LoRA alignment
            "cos_fused_to_original_per_task": cos_fused_original.detach().cpu().tolist(),
            "cos_fused_to_kept_per_task": cos_fused_kept.detach().cpu().tolist(),
            "cos_fused_to_residual_per_task": cos_fused_residual.detach().cpu().tolist(),

            "avg_cos_fused_to_original": float(cos_fused_original.mean().item()),
            "avg_cos_fused_to_kept": float(cos_fused_kept.mean().item()),
            "avg_cos_fused_to_residual": float(cos_fused_residual.mean().item()),

            "avg_dist_fused_to_original": float(dist_fused_original.mean().item()),
            "avg_dist_fused_to_kept": float(dist_fused_kept.mean().item()),

            # Norms
            "fused_norm": float(torch.norm(fused_vec).item()),
            "avg_original_norm": float(torch.norm(Delta, dim=1).mean().item()),
            "avg_kept_norm": float(torch.norm(Kept, dim=1).mean().item()),
            "avg_residual_norm": float(torch.norm(Residual, dim=1).mean().item()),
        }

        dump_svd_record(record)

    return fused

def shared_subspace_normalized_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    rank=None,
    rank_ratio=None,
    debug_analysis=False,
):
    """
    Shared-subspace resolver with per-task normalization before SVD.

    Difference from shared_subspace_resolver:
        - Before SVD, each task's A matrix is normalized by its Frobenius norm.
        - This prevents tasks with larger adapter magnitude from dominating
          the SVD basis.
        - The original unnormalized A_t and B_t are still used for least-squares
          reconstruction and for the final fused LoRA update.
    """

    assert not full_rank

    eps = 1e-8
    do_analysis = debug_analysis or is_svd_analysis_enabled()

    base_A = lora_A[tasks[0]]
    r_task, d_in = base_A.shape

    if rank is None and rank_ratio is not None:
        rank = max(1, int(r_task * rank_ratio))
    if rank is None:
        rank = max(1, r_task // 2)

    rank = min(rank, d_in)

    A_list = [lora_A[t] for t in tasks]
    B_list = [lora_B[t] for t in tasks]

    out_dim = B_list[0].shape[0]

    # normalize each A_t (Frobenius norm) before the SVD, A_t.T is [d_in, r_task]
    A_cat = torch.cat(
        [
            A.T.float() / (A.T.float().norm(p="fro") + eps)
            for A in A_list
        ],
        dim=1,
    )

    # SVD on normalized A_cat.
    U, S, _ = torch.linalg.svd(A_cat, full_matrices=False)

    A_shared = U[:, :rank].T

    target_dtype = base_A.dtype

    if A_shared.dtype != target_dtype:
        A_shared = A_shared.to(target_dtype)

    B_acc = torch.zeros(
        out_dim,
        rank,
        dtype=target_dtype,
        device=A_shared.device,
    )

    if do_analysis:
        Delta_list = []
        Kept_list = []
        Residual_list = []
        A_norms = []

    for A_t, B_t in zip(A_list, B_list):
        full_update = B_t @ A_t
        full_update = full_update.to(target_dtype)

        # Project the ORIGINAL unnormalized full update into the shared A basis.
        X = torch.linalg.lstsq(
            A_shared.T,
            full_update.T,
        ).solution.T

        if X.shape[1] < rank:
            pad_rank = rank - X.shape[1]
            X = torch.cat(
                [
                    X,
                    torch.zeros(
                        out_dim,
                        pad_rank,
                        dtype=target_dtype,
                        device=X.device,
                    ),
                ],
                dim=1,
            )

        B_acc += X

        if do_analysis:
            kept_update = X @ A_shared
            residual_update = full_update - kept_update

            Delta_list.append(full_update.detach().flatten().float())
            Kept_list.append(kept_update.detach().flatten().float())
            Residual_list.append(residual_update.detach().flatten().float())
            A_norms.append(float(A_t.detach().float().norm(p="fro").item()))

    B_shared = B_acc / len(B_list)

    fused = B_shared @ A_shared

    if do_analysis:
        Delta = torch.stack(Delta_list, dim=0)
        Kept = torch.stack(Kept_list, dim=0)
        Residual = torch.stack(Residual_list, dim=0)

        avg_delta, npr_delta, cos_delta = cosine_stats(Delta)
        avg_kept, npr_kept, cos_kept = cosine_stats(Kept)
        avg_res, npr_res, cos_res = cosine_stats(Residual)

        per_task_original = per_task_cosine_stats(cos_delta, tasks)
        per_task_kept = per_task_cosine_stats(cos_kept, tasks)
        per_task_residual = per_task_cosine_stats(cos_res, tasks)

        total_energy = torch.norm(Delta, p="fro") ** 2
        kept_energy = torch.norm(Kept, p="fro") ** 2
        residual_energy = torch.norm(Residual, p="fro") ** 2

        sv_energy = S.detach().float() ** 2
        cumulative_energy = torch.cumsum(sv_energy, dim=0) / (sv_energy.sum() + eps)

        fused_vec = fused.detach().flatten().float().unsqueeze(0)

        Delta_norm = torch.nn.functional.normalize(Delta, dim=1)
        Kept_norm = torch.nn.functional.normalize(Kept, dim=1)
        Residual_norm = torch.nn.functional.normalize(Residual, dim=1)
        fused_norm = torch.nn.functional.normalize(fused_vec, dim=1)

        cos_fused_original = (Delta_norm @ fused_norm.T).squeeze(1)
        cos_fused_kept = (Kept_norm @ fused_norm.T).squeeze(1)
        cos_fused_residual = (Residual_norm @ fused_norm.T).squeeze(1)

        dist_fused_original = torch.norm(Delta - fused_vec, dim=1)
        dist_fused_kept = torch.norm(Kept - fused_vec, dim=1)

        record = {
            "call_id": next_svd_call_id(),
            "resolver": "shared_subspace_normalized",
            "normalization": "per_task_A_frobenius_before_svd",
            "rank": int(rank),
            "num_tasks": int(len(tasks)),
            "tasks": list(tasks),

            # Useful to verify whether task magnitudes were very different.
            "A_fro_norms_per_task": dict(zip(tasks, A_norms)),

            # Pairwise task-update alignment
            "avg_cos_original": float(avg_delta),
            "avg_cos_kept": float(avg_kept),
            "avg_cos_residual": float(avg_res),

            "npr_original": float(npr_delta),
            "npr_kept": float(npr_kept),
            "npr_residual": float(npr_res),

            "per_task_original": per_task_original,
            "per_task_kept": per_task_kept,
            "per_task_residual": per_task_residual,

            # Energy retained/removed
            "energy_original": 1.0,
            "energy_kept": float((kept_energy / (total_energy + eps)).item()),
            "energy_residual": float((residual_energy / (total_energy + eps)).item()),

            # SVD spectrum of normalized A_cat
            "singular_values": S.detach().float().cpu().tolist(),
            "cumulative_energy": cumulative_energy.detach().cpu().tolist(),

            # Heatmaps
            "cos_original": cos_delta.detach().float().cpu().tolist(),
            "cos_kept": cos_kept.detach().float().cpu().tolist(),
            "cos_residual": cos_res.detach().float().cpu().tolist(),

            # Merged LoRA alignment
            "cos_fused_to_original_per_task": cos_fused_original.detach().cpu().tolist(),
            "cos_fused_to_kept_per_task": cos_fused_kept.detach().cpu().tolist(),
            "cos_fused_to_residual_per_task": cos_fused_residual.detach().cpu().tolist(),

            "avg_cos_fused_to_original": float(cos_fused_original.mean().item()),
            "avg_cos_fused_to_kept": float(cos_fused_kept.mean().item()),
            "avg_cos_fused_to_residual": float(cos_fused_residual.mean().item()),

            "avg_dist_fused_to_original": float(dist_fused_original.mean().item()),
            "avg_dist_fused_to_kept": float(dist_fused_kept.mean().item()),

            # Norms
            "fused_norm": float(torch.norm(fused_vec).item()),
            "avg_original_norm": float(torch.norm(Delta, dim=1).mean().item()),
            "avg_kept_norm": float(torch.norm(Kept, dim=1).mean().item()),
            "avg_residual_norm": float(torch.norm(Residual, dim=1).mean().item()),
        }

        dump_svd_record(record)

    return fused

def shared_subspace_B_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    rank=None,
    rank_ratio=None,
    debug_analysis=False,
):
    """
    Shared-subspace resolver using SVD on B instead of A.

    Original A-SVD version:
        learn shared A basis, average projected B.

    This B-SVD version:
        learn shared B basis, average projected A.

    For each task:
        full_update_t = B_t @ A_t

    We build:
        B_shared from SVD over all B_t

    Then solve:
        full_update_t ~= B_shared @ A_t_projected

    Finally:
        A_shared = average_t A_t_projected
        fused = B_shared @ A_shared
    """

    assert not full_rank

    do_analysis = debug_analysis or is_svd_analysis_enabled()

    base_A = lora_A[tasks[0]]
    base_B = lora_B[tasks[0]]

    r_task, d_in = base_A.shape
    out_dim, _ = base_B.shape

    if rank is None and rank_ratio is not None:
        rank = max(1, int(r_task * rank_ratio))
    if rank is None:
        rank = max(1, r_task // 2)

    # For B-SVD, rank cannot exceed out_dim.
    rank = min(rank, out_dim)

    A_list = [lora_A[t] for t in tasks]
    B_list = [lora_B[t] for t in tasks]

    target_dtype = base_B.dtype

    # ---------------------------------------------------------
    # SVD on B
    # ---------------------------------------------------------
    # Each B_t: [out_dim, r_task]
    # B_cat:    [out_dim, num_tasks * r_task]
    # U:        [out_dim, min(out_dim, num_tasks * r_task)]
    # B_shared: [out_dim, rank]
    # ---------------------------------------------------------
    B_cat = torch.cat([B for B in B_list], dim=1).float()

    U, S, _ = torch.linalg.svd(B_cat, full_matrices=False)

    B_shared = U[:, :rank]

    if B_shared.dtype != target_dtype:
        B_shared = B_shared.to(target_dtype)

    # Accumulate projected A matrices.
    # A_projected has shape [rank, d_in]
    A_acc = torch.zeros(
        rank,
        d_in,
        dtype=target_dtype,
        device=B_shared.device,
    )

    if do_analysis:
        Delta_list = []
        Kept_list = []
        Residual_list = []

    for A_t, B_t in zip(A_list, B_list):
        full_update = B_t @ A_t
        full_update = full_update.to(target_dtype)

        # -----------------------------------------------------
        # Solve:
        #     B_shared @ X ~= full_update
        #
        # B_shared:    [out_dim, rank]
        # full_update: [out_dim, d_in]
        # X:           [rank, d_in]
        # -----------------------------------------------------
        X = torch.linalg.lstsq(
            B_shared,
            full_update,
        ).solution

        if X.shape[0] < rank:
            pad_rank = rank - X.shape[0]
            X = torch.cat(
                [
                    X,
                    torch.zeros(
                        pad_rank,
                        d_in,
                        dtype=target_dtype,
                        device=X.device,
                    ),
                ],
                dim=0,
            )

        A_acc += X

        if do_analysis:
            kept_update = B_shared @ X
            residual_update = full_update - kept_update

            Delta_list.append(full_update.detach().flatten().float())
            Kept_list.append(kept_update.detach().flatten().float())
            Residual_list.append(residual_update.detach().flatten().float())

    A_shared = A_acc / len(A_list)

    fused = B_shared @ A_shared

    if do_analysis:
        Delta = torch.stack(Delta_list, dim=0)
        Kept = torch.stack(Kept_list, dim=0)
        Residual = torch.stack(Residual_list, dim=0)

        avg_delta, npr_delta, cos_delta = cosine_stats(Delta)
        avg_kept, npr_kept, cos_kept = cosine_stats(Kept)
        avg_res, npr_res, cos_res = cosine_stats(Residual)

        per_task_original = per_task_cosine_stats(cos_delta, tasks)
        per_task_kept = per_task_cosine_stats(cos_kept, tasks)
        per_task_residual = per_task_cosine_stats(cos_res, tasks)

        total_energy = torch.norm(Delta, p="fro") ** 2
        kept_energy = torch.norm(Kept, p="fro") ** 2
        residual_energy = torch.norm(Residual, p="fro") ** 2

        sv_energy = S.detach().float() ** 2
        cumulative_energy = torch.cumsum(sv_energy, dim=0) / sv_energy.sum()

        # -----------------------------------------------------
        # Merged LoRA diagnostics
        # -----------------------------------------------------
        fused_vec = fused.detach().flatten().float().unsqueeze(0)

        Delta_norm = torch.nn.functional.normalize(Delta, dim=1)
        Kept_norm = torch.nn.functional.normalize(Kept, dim=1)
        Residual_norm = torch.nn.functional.normalize(Residual, dim=1)
        fused_norm = torch.nn.functional.normalize(fused_vec, dim=1)

        cos_fused_original = (Delta_norm @ fused_norm.T).squeeze(1)
        cos_fused_kept = (Kept_norm @ fused_norm.T).squeeze(1)
        cos_fused_residual = (Residual_norm @ fused_norm.T).squeeze(1)

        dist_fused_original = torch.norm(Delta - fused_vec, dim=1)
        dist_fused_kept = torch.norm(Kept - fused_vec, dim=1)

        record = {
            "call_id": next_svd_call_id(),
            "resolver": "shared_subspace_B_resolver",
            "svd_on": "B",
            "rank": int(rank),
            "num_tasks": int(len(tasks)),
            "tasks": list(tasks),

            # Pairwise task-update alignment
            "avg_cos_original": float(avg_delta),
            "avg_cos_kept": float(avg_kept),
            "avg_cos_residual": float(avg_res),

            "npr_original": float(npr_delta),
            "npr_kept": float(npr_kept),
            "npr_residual": float(npr_res),

            "per_task_original": per_task_original,
            "per_task_kept": per_task_kept,
            "per_task_residual": per_task_residual,

            # Energy retained/removed
            "energy_original": 1.0,
            "energy_kept": float((kept_energy / total_energy).item()),
            "energy_residual": float((residual_energy / total_energy).item()),

            # SVD spectrum from B_cat
            "singular_values": S.detach().float().cpu().tolist(),
            "cumulative_energy": cumulative_energy.detach().cpu().tolist(),

            # Heatmaps for task-task cosine
            "cos_original": cos_delta.detach().float().cpu().tolist(),
            "cos_kept": cos_kept.detach().float().cpu().tolist(),
            "cos_residual": cos_res.detach().float().cpu().tolist(),

            # Merged LoRA alignment
            "cos_fused_to_original_per_task": cos_fused_original.detach().cpu().tolist(),
            "cos_fused_to_kept_per_task": cos_fused_kept.detach().cpu().tolist(),
            "cos_fused_to_residual_per_task": cos_fused_residual.detach().cpu().tolist(),

            "avg_cos_fused_to_original": float(cos_fused_original.mean().item()),
            "avg_cos_fused_to_kept": float(cos_fused_kept.mean().item()),
            "avg_cos_fused_to_residual": float(cos_fused_residual.mean().item()),

            "avg_dist_fused_to_original": float(dist_fused_original.mean().item()),
            "avg_dist_fused_to_kept": float(dist_fused_kept.mean().item()),

            # Norms
            "fused_norm": float(torch.norm(fused_vec).item()),
            "avg_original_norm": float(torch.norm(Delta, dim=1).mean().item()),
            "avg_kept_norm": float(torch.norm(Kept, dim=1).mean().item()),
            "avg_residual_norm": float(torch.norm(Residual, dim=1).mean().item()),
        }

        dump_svd_record(record)

    return fused
def rank_avg_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    rank=None
):

    assert not full_rank

    if rank is None:
        rank = max(1, lora_A[tasks[0]].shape[0] // 2)

    out_dim = lora_B[tasks[0]].shape[0]
    in_dim = lora_A[tasks[0]].shape[1]

    A_avg = torch.zeros(rank, in_dim, device=lora_A[tasks[0]].device)
    B_avg = torch.zeros(out_dim, rank, device=lora_B[tasks[0]].device)

    for task in tasks:
        A_t = lora_A[task]
        B_t = lora_B[task]
        A_reduced = A_t[:rank, :]
        B_reduced = B_t[:, :rank]
        A_avg += A_reduced
        B_avg += B_reduced

    A_shared = A_avg / len(tasks)
    B_shared = B_avg / len(tasks)

    return B_shared @ A_shared

def coeff_shared_subspace_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    rank=None,

):

    assert not full_rank
    base_A = lora_A[tasks[0]]
    r_task, d_in = base_A.shape
    if rank is None:
        rank = max(1, r_task // 2)

    rank = min(rank, d_in)
    A_list = [lora_A[t] for t in tasks]
    B_list = [lora_B[t] for t in tasks]
    out_dim = B_list[0].shape[0]
    A_cat = torch.cat([A.T for A in A_list], dim=1).float()
    U, _, _ = torch.linalg.svd(A_cat, full_matrices=False)
    A_shared = U[:, :rank].T
    target_dtype = base_A.dtype
    A_shared = A_shared.to(target_dtype)

    if coeff_A is not None:
        raw_coeff = torch.stack(
            [coeff_A[t] for t in tasks],
            dim=0,
        ).view(-1)
        alpha = F.softmax(raw_coeff / temperature, dim=0).to(target_dtype)

    else:

        alpha = torch.full(
            (len(tasks),),
            1.0 / len(tasks),
            dtype=target_dtype,
            device=A_shared.device,
        )

    B_acc = torch.zeros(
        out_dim,
        rank,
        dtype=target_dtype,
        device=A_shared.device,
    )

    for i, (A_t, B_t) in enumerate(zip(A_list, B_list)):
        full_update = B_t @ A_t
        full_update = full_update.to(target_dtype)
        X = torch.linalg.lstsq(
            A_shared.T,
            full_update.T,
        ).solution.T
        if X.shape[1] < rank:
            pad_rank = rank - X.shape[1]
            X = torch.cat(
                [X,
                    torch.zeros(
                        out_dim,
                        pad_rank,
                        dtype=target_dtype,
                        device=X.device,),],
                dim=1,
            )
        B_acc += alpha[i] * X
    fused = B_acc @ A_shared
    return fused

def soft_coeff_then_shared_subspace_svd_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    rank=None,
):
    assert not full_rank
    #print(f"Running soft_coeff_then_shared_subspace_svd_resolver with rank={rank}")

    base_A = lora_A[tasks[0]]
    r_task, d_in = base_A.shape

    if rank is None:
        rank = max(1, r_task // 2)
    rank = min(rank, d_in)

    target_dtype = base_A.dtype
    device = base_A.device

    # [T, r, in_features]
    A_stack = torch.stack(
        [lora_A[t] for t in tasks],
        dim=0,
    ).to(dtype=target_dtype, device=device)

    # [T, out_features, r]
    B_stack = torch.stack(
        [lora_B[t] for t in tasks],
        dim=0,
    ).to(dtype=target_dtype, device=device)

    # [T]
    raw_coeff = torch.stack(
        [coeff_A[t] for t in tasks],
        dim=0,
    ).view(-1)

    alpha = F.softmax(
        raw_coeff / temperature,
        dim=0,
    ).to(dtype=target_dtype, device=device)

    # Delta_soft = sum_t alpha_t * (B_t @ A_t)
    # B_stack: [T, out, r]
    # A_stack: [T, r, in]
    # output:  [out, in]
    Delta_soft = torch.einsum(
        "t,tor,tri->oi",
        alpha,
        B_stack,
        A_stack,
    )

    # A_cat = [A_1^T | A_2^T | ... | A_T^T]
    # shape: [in_features, T * r]
    A_cat = A_stack.permute(2, 0, 1).reshape(d_in, -1).float()

    # Exact/economy SVD
    with torch.cuda.amp.autocast(enabled=False):
        U, _, _ = torch.linalg.svd(
            A_cat,
            full_matrices=False,
        )

    A_shared = U[:, :rank].T.to(dtype=target_dtype, device=device)
    # [rank, in_features]

    # Closed-form projection because A_shared is orthonormal
    X = Delta_soft @ A_shared.T
    # [out_features, rank]

    fused = X @ A_shared
    # [out_features, in_features]

    return fused


def soft_coeff_then_shared_subspace_lowrank_svd_resolver(
    full_rank,
    x,
    lora_A,
    lora_B,
    tasks,
    direction_A,
    direction_B,
    coeff_A,
    coeff_B,
    training,
    noise_factor,
    temperature,
    atten_emb_dim,
    kernel_type,
    rank=None,
    oversample=2,
    niter=1,
):
    assert not full_rank
    #print(f"Running soft_coeff_then_shared_subspace_lowrank_svd_resolver with rank={rank}, oversample={oversample}, niter={niter}")

    base_A = lora_A[tasks[0]]
    r_task, d_in = base_A.shape

    if rank is None:
        rank = max(1, r_task // 2)
    rank = min(rank, d_in)

    target_dtype = base_A.dtype
    device = base_A.device

    A_stack = torch.stack(
        [lora_A[t] for t in tasks],
        dim=0,
    ).to(dtype=target_dtype, device=device)

    B_stack = torch.stack(
        [lora_B[t] for t in tasks],
        dim=0,
    ).to(dtype=target_dtype, device=device)

    raw_coeff = torch.stack(
        [coeff_A[t] for t in tasks],
        dim=0,
    ).view(-1)

    alpha = F.softmax(
        raw_coeff / temperature,
        dim=0,
    ).to(dtype=target_dtype, device=device)

    # Delta_soft = sum_t alpha_t * (B_t @ A_t)
    Delta_soft = torch.einsum(
        "t,tor,tri->oi",
        alpha,
        B_stack,
        A_stack,
    )

    A_cat = A_stack.permute(2, 0, 1).reshape(d_in, -1).float()

    # q should be a little larger than rank for better approximation.
    q = min(rank + oversample, min(A_cat.shape))

    with torch.cuda.amp.autocast(enabled=False):
        U, _, _ = torch.svd_lowrank(
            A_cat,
            q=q,
            niter=niter,
        )

    A_shared = U[:, :rank].T.to(dtype=target_dtype, device=device)

    # Closed-form projection
    X = Delta_soft @ A_shared.T

    fused = X @ A_shared

    return fused
import os
import json
import torch
import torch.nn.functional as F


SVD_ANALYSIS_PATH = None
SVD_ANALYSIS_COUNTER = 0
SVD_DEBUG_ANALYSIS = False


def set_svd_analysis_path(path):
    global SVD_ANALYSIS_PATH, SVD_ANALYSIS_COUNTER

    SVD_ANALYSIS_PATH = path
    SVD_ANALYSIS_COUNTER = 0

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Clear old file
    with open(path, "w") as f:
        pass


def enable_svd_analysis(path):
    global SVD_DEBUG_ANALYSIS

    set_svd_analysis_path(path)
    SVD_DEBUG_ANALYSIS = True


def disable_svd_analysis():
    global SVD_DEBUG_ANALYSIS

    SVD_DEBUG_ANALYSIS = False


def is_svd_analysis_enabled():
    return SVD_DEBUG_ANALYSIS


def next_svd_call_id():
    global SVD_ANALYSIS_COUNTER

    call_id = SVD_ANALYSIS_COUNTER
    SVD_ANALYSIS_COUNTER += 1
    return call_id


def cosine_stats(X):
    X = F.normalize(X, dim=1)
    cos = X @ X.T

    T = X.shape[0]
    mask = ~torch.eye(T, dtype=torch.bool, device=X.device)
    vals = cos[mask]

    avg_cos = vals.mean().item()
    npr = (vals < 0).float().mean().item()

    return avg_cos, npr, cos


def dump_svd_record(record):
    global SVD_ANALYSIS_PATH

    if SVD_ANALYSIS_PATH is None:
        return

    with open(SVD_ANALYSIS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def per_task_cosine_stats(cos_mat, tasks):

    """

    cos_mat: [T, T] cosine similarity matrix

    tasks: list of task names

    """

    T = len(tasks)

    stats = {}

    for i, task in enumerate(tasks):
        mask = torch.ones(T, dtype=torch.bool, device=cos_mat.device)
        mask[i] = False
        vals = cos_mat[i][mask]
        stats[task] = {
            "avg_cos": float(vals.mean().item()),
            "npr": float((vals < 0).float().mean().item()),
            "min_cos": float(vals.min().item()),
            "max_cos": float(vals.max().item()),
        }

    return stats
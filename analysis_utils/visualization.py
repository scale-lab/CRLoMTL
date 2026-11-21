import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from .batch_utils import unpack_batch

import os
import re
import json
def plot_tsne_per_task(model, dataloader, device, config):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x, y_dict = unpack_batch(batch, config)
            x, y_dict = x.to(device, non_blocking=True), {
                k: v.to(device, non_blocking=True) for k, v in y_dict.items()
            }

            _, shared_ft = model(x, return_shared_ft=True)

            for task in model.tasks:
                task_feature = shared_ft[task]

                if isinstance(task_feature, list):
                    task_feature = task_feature[-1]

                if isinstance(task_feature, torch.Tensor) and task_feature.ndim > 2:
                    task_feature = torch.mean(task_feature, dim=[2, 3])

                task_feature_np = task_feature.cpu().numpy()

                all_features.append(task_feature_np)
                all_labels.extend([task] * task_feature_np.shape[0])

            if idx == 2:
                break

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    unique_labels, numeric_labels = np.unique(all_labels, return_inverse=True)

    # Calculate silhouette score in t-SNE space
    sil_score = silhouette_score(tsne_results, numeric_labels)

    plt.figure(figsize=(3.5, 3), dpi=400)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=numeric_labels, cmap="tab10", alpha=0.6, s=5)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=plt.cm.tab10(idx / max(len(unique_labels) - 1, 1)),
                   label=label, markersize=6)
        for idx, label in enumerate(unique_labels)
    ]

    plt.legend(handles=handles, fontsize=5)

    # Add silhouette score to title
    plt.title(f"t-SNE of Features (Silhouette Score = {sil_score:.2f})", fontsize=7)

    #plt.savefig("tsne_features_all_tasks.png", dpi=350, bbox_inches='tight')
    plt.savefig("tsne_features_all_tasks.pdf", format='pdf', bbox_inches='tight')
    plt.show()   



def plot_avg_cosine_over_epochs(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(5.5, 3.6),dpi=400)
    plt.plot(df["epoch"], df["avg_cos_original"], marker="o", label=r"Original $\Delta_t$")
    plt.plot(df["epoch"], df["avg_cos_kept"], marker="o", label=r"Retained $K_t$")
    plt.plot(df["epoch"], df["avg_cos_residual"], marker="o", label=r"Residual $R_t$")

    plt.xlabel("Epoch", labelpad=8)
    plt.ylabel("Average pairwise cosine")
    plt.title("Task-update alignment over training")
    plt.grid(True, alpha=0.3)
    #plt.legend()
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=True,
    fancybox=False,
    fontsize=9)
    plt.tight_layout()

    path = os.path.join(save_dir, "avg_cosine_over_epochs.pdf")
    path2 = os.path.join(save_dir, "avg_cosine_over_epochs.png")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path2, bbox_inches="tight")
    plt.show()
    print("Saved:", path)

def plot_energy_over_epochs(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(5.5, 3.6), dpi=400)
    plt.plot(df["epoch"], df["energy_kept"], marker="o", label=r"Retained $K_t$")
    plt.plot(df["epoch"], df["energy_residual"], marker="o", label=r"Residual $R_t$")

    plt.xlabel("Epoch", labelpad=8)
    plt.ylabel("Energy ratio")
    plt.title("SVD retained vs. residual energy")
    plt.grid(True, alpha=0.3)
    #plt.legend()
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=True,
    fancybox=False,
    fontsize=9)
    plt.tight_layout()

    path = os.path.join(save_dir, "energy_over_epochs.pdf")
    path2 = os.path.join(save_dir, "energy_over_epochs.png")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path2, bbox_inches="tight")
    plt.show()
    print("Saved:", path)


def plot_fused_alignment_over_epochs(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(5.5, 3.6), dpi=400)
    plt.plot(df["epoch"], df["avg_cos_fused_to_original"], marker="o", label=r"Fused vs. original $\Delta_t$")
    plt.plot(df["epoch"], df["avg_cos_fused_to_kept"], marker="o", label=r"Fused vs. retained $K_t$")
    plt.plot(df["epoch"], df["avg_cos_fused_to_residual"], marker="o", label=r"Fused vs. residual $R_t$")

    plt.xlabel("Epoch", labelpad=8)
    plt.ylabel("Average cosine to fused update")
    plt.title("Merged LoRA alignment over training")
    plt.grid(True, alpha=0.3)
    #plt.legend()
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=True,
    fancybox=False,
    fontsize=9
)
    plt.tight_layout()

    path = os.path.join(save_dir, "fused_alignment_over_epochs.pdf")
    path2 = os.path.join(save_dir, "fused_alignment_over_epochs.png")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path2, bbox_inches="tight")
    plt.show()
    print("Saved:", path)


def average_cosine_matrices(records):
    cos_original = np.array([r["cos_original"] for r in records])
    cos_kept = np.array([r["cos_kept"] for r in records])
    cos_residual = np.array([r["cos_residual"] for r in records])

    return {
        "tasks": records[0]["tasks"],
        "cos_original": cos_original.mean(axis=0),
        "cos_kept": cos_kept.mean(axis=0),
        "cos_residual": cos_residual.mean(axis=0),
    }



def load_jsonl(path):

    records = []

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if line:

                records.append(json.loads(line))

    return records

def extract_epoch(path):
    name = os.path.basename(path)
    match = re.search(r"epoch_(\d+)", name)
    if match is None:
        raise ValueError(f"Could not extract epoch from filename: {name}")

    return int(match.group(1))
def plot_heatmap(mat, tasks, title, save_path):
    plt.figure(figsize=(4.0, 3.6), dpi=400)
    plt.imshow(mat, vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")

    plt.xticks(range(len(tasks)), tasks, rotation=45, ha="right")
    plt.yticks(range(len(tasks)), tasks)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print("Saved:", save_path)


def plot_epoch_heatmaps(jsonl_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epoch = extract_epoch(jsonl_path)
    records = load_jsonl(jsonl_path)
    avg = average_cosine_matrices(records)

    tasks = avg["tasks"]

    plot_heatmap(
        avg["cos_original"],
        tasks,
        rf"Original $\Delta_t$ - Epoch {epoch}",
        os.path.join(save_dir, f"epoch_{epoch}_heatmap_original.pdf"),
    )

    plot_heatmap(
        avg["cos_kept"],
        tasks,
        rf"Retained $K_t$ - Epoch {epoch}",
        os.path.join(save_dir, f"epoch_{epoch}_heatmap_kept.pdf"),
    )

    plot_heatmap(
        avg["cos_residual"],
        tasks,
        rf"Residual $R_t$ - Epoch {epoch}",
        os.path.join(save_dir, f"epoch_{epoch}_heatmap_residual.pdf"),
    )


def plot_npr_over_epochs(df, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(5.5, 3.6),dpi=400)
    plt.plot(df["epoch"], df["npr_original"], marker="o", label=r"Original $\Delta_t$")
    plt.plot(df["epoch"], df["npr_kept"], marker="o", label=r"Retained $K_t$")
    plt.plot(df["epoch"], df["npr_residual"], marker="o", label=r"Residual $R_t$")
    plt.xlabel("Epoch", labelpad=8)
    plt.ylabel("Negative-pair rate")
    plt.title("Task conflict rate over training")
    plt.grid(True, alpha=0.3)
    #plt.legend()
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=True,
    fancybox=False,
    fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, "npr_over_epochs.pdf")
    path2 = os.path.join(save_dir, "npr_over_epochs.png")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path2, bbox_inches="tight")
    plt.show()
    print("Saved:", path)

def plot_combined_svd_analysis2(df, save_dir):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), dpi=300)

    # --------------------------------------------------
    # (a) Average cosine
    # --------------------------------------------------
    ax = axes[0]
    ax.plot(df["epoch"], df["avg_cos_original"], marker="o", linewidth=1.8, markersize=4, label=r"Original $\Delta_t$")
    ax.plot(df["epoch"], df["avg_cos_kept"], marker="s", linewidth=1.8, markersize=4, label=r"Retained $K_t$")
    ax.plot(df["epoch"], df["avg_cos_residual"], marker="^", linewidth=1.8, markersize=4, label=r"Residual $R_t$")
    ax.set_title("(a) Alignment")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg. pairwise cosine")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=True, fancybox=False)

    # --------------------------------------------------
    # (b) NPR
    # --------------------------------------------------
    ax = axes[1]
    ax.plot(df["epoch"], df["npr_original"], marker="o", linewidth=1.8, markersize=4, label=r"Original $\Delta_t$")
    ax.plot(df["epoch"], df["npr_kept"], marker="s", linewidth=1.8, markersize=4, label=r"Retained $K_t$")
    ax.plot(df["epoch"], df["npr_residual"], marker="^", linewidth=1.8, markersize=4, label=r"Residual $R_t$")
    ax.set_title("(b) Conflict rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NPR")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=True, fancybox=False)

    # --------------------------------------------------
    # (c) Fused alignment
    # --------------------------------------------------
    ax = axes[2]
    ax.plot(df["epoch"], df["avg_cos_fused_to_original"], marker="o", linewidth=1.8, markersize=4, label=r"Original $\Delta_t$")
    ax.plot(df["epoch"], df["avg_cos_fused_to_kept"], marker="s", linewidth=1.8, markersize=4, label=r"Retained $K_t$")
    ax.plot(df["epoch"], df["avg_cos_fused_to_residual"], marker="^", linewidth=1.8, markersize=4, label=r"Residual $R_t$")
    ax.set_title("(c) Fused alignment")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg. cosine to fused")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=True, fancybox=False)

    plt.subplots_adjust(bottom=0.32, wspace=0.35)

    path_pdf = os.path.join(save_dir, "combined_svd_resolver_analysis.pdf")
    path_png = os.path.join(save_dir, "combined_svd_resolver_analysis.png")

    plt.savefig(path_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(path_png, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved:", path_pdf)
    print("Saved:", path_png)    

def plot_combined_svd_analysis(df, save_dir):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Slightly larger figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4), dpi=300)

    # --------------------------------------------------
    # (a) Average cosine
    # --------------------------------------------------
    ax = axes[0]
    ax.plot(
        df["epoch"],
        df["avg_cos_original"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label=r"Original $\Delta_t$",
    )
    ax.plot(
        df["epoch"],
        df["avg_cos_kept"],
        marker="s",
        linewidth=1.8,
        markersize=4,
        label=r"Retained $K_t$",
    )
    ax.plot(
        df["epoch"],
        df["avg_cos_residual"],
        marker="^",
        linewidth=1.8,
        markersize=4,
        label=r"Residual $R_t$",
    )
    ax.set_title("(a) Alignment")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg. pairwise cosine")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # (b) NPR
    # --------------------------------------------------
    ax = axes[1]
    ax.plot(
        df["epoch"],
        df["npr_original"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label=r"Original $\Delta_t$",
    )
    ax.plot(
        df["epoch"],
        df["npr_kept"],
        marker="s",
        linewidth=1.8,
        markersize=4,
        label=r"Retained $K_t$",
    )
    ax.plot(
        df["epoch"],
        df["npr_residual"],
        marker="^",
        linewidth=1.8,
        markersize=4,
        label=r"Residual $R_t$",
    )
    ax.set_title("(b) Conflict rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NPR")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # (c) Fused alignment
    # --------------------------------------------------
    ax = axes[2]
    ax.plot(
        df["epoch"],
        df["avg_cos_fused_to_original"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label=r"Original $\Delta_t$",
    )
    ax.plot(
        df["epoch"],
        df["avg_cos_fused_to_kept"],
        marker="s",
        linewidth=1.8,
        markersize=4,
        label=r"Retained $K_t$",
    )
    ax.plot(
        df["epoch"],
        df["avg_cos_fused_to_residual"],
        marker="^",
        linewidth=1.8,
        markersize=4,
        label=r"Residual $R_t$",
    )
    ax.set_title("(c) Fused alignment")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg. cosine to fused")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # One shared legend for all three subplots
    # --------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
        fancybox=False,
    )

    # --------------------------------------------------
    # Layout adjustment
    # --------------------------------------------------
    plt.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.25,
        top=0.88,
        wspace=0.25,
    )

    path_pdf = os.path.join(save_dir, "combined_svd_resolver_analysis.pdf")
    path_png = os.path.join(save_dir, "combined_svd_resolver_analysis.png")

    plt.savefig(path_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(path_png, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved:", path_pdf)
    print("Saved:", path_png)
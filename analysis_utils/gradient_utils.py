import torch


def log_gradient_statistics(task, grads_vec, norm, tasks_grads, grad_norm_log, epoch, idx, train_loader, config):
    """
    Logs gradient statistics including mean, max, norm, and smallest singular value of G^T G.

    Args:
        task (str): Task name.
        grads_vec (torch.Tensor): Flattened gradient vector for the current task.
        norm (float): Pre-computed gradient norm for the current task.
        tasks_grads (list of torch.Tensor): List collecting all task gradients.
        grad_norm_log (list): List collecting gradient stats for logging.
        epoch (int): Current epoch.
        idx (int): Current batch index.
        train_loader (DataLoader): Training data loader (used for step calculation).
        config (object): Configuration object.
    """
    step = epoch * len(train_loader) + idx
   
   

    grad_mean = grads_vec.mean().item()
    grad_max = grads_vec.abs().max().item()

    grad_norm_log.append({
        task: {
            'step': step,
            'grad_norm': norm,
            'grad_mean': grad_mean,
            'grad_max': grad_max,
        }
    })

    if getattr(config, "WANDB", False):
        import wandb
        wandb.log({
            f"{task}_grad_norm": norm,
            f"{task}_grad_mean": grad_mean,
            f"{task}_grad_max": grad_max,
        })

def compute_and_log_singular_value(tasks_grads, epoch_step=None, config=None):
  
    n = len(tasks_grads)
    device = tasks_grads[0].device
    dtype = tasks_grads[0].dtype
    max_len = max(g.numel() for g in tasks_grads)
    padded_grads = []
    for i, g in enumerate(tasks_grads):
            if torch.isnan(g).any() or torch.isinf(g).any():
                #print(f"[Warning] Task {i} gradient has NaN or Inf — replacing with zeros.")
                g = torch.zeros_like(g)

            if g.numel() < max_len:
                pad = torch.zeros(max_len - g.numel(), device=device, dtype=dtype)
                g = torch.cat([g, pad], dim=0)
            padded_grads.append(g)


    G = torch.stack(padded_grads, dim=1)  # Shape: [gradient_dim, num_tasks]
    GtG = G.T @ G  # Shape: [num_tasks, num_tasks]

    singular_values = torch.linalg.svdvals(GtG)
    smallest_singular_value = singular_values[-1].item()

    if config and getattr(config, "WANDB", False):
        log_dict = {
            "smallest_singular_value_GtG": smallest_singular_value
        }
        if epoch_step is not None:
            log_dict["step"] = epoch_step
        import wandb
        wandb.log(log_dict)

    return smallest_singular_value

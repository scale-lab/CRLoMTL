import torch
import numpy as np
import matplotlib.pyplot as plt
def calculate_tasks_gradient(img,model,task,predictions,labels,optimizer,criterion,config,loss_scaler,is_update_step,trainable_params):
    predictions, features = model(
                            img, return_representation=True, task=task
                        )
    
    task_loss, loss_dict = criterion(predictions, labels)
    task_loss /= config.TRAIN.ACCUMULATION_STEPS
                    
    norm, grads_vec=loss_scaler(
                        task_loss,
                        optimizer,
                        clip_grad=config.TRAIN.CLIP_GRAD,
                        parameters=trainable_params,
                        create_graph=False,
                        update_grad=is_update_step,
                        adaptive_lambda=config.CRLOMTL.ADAPTIVE_LAMBDA,
                        draw_cosine_matrix=config.DRAW_COSINE_MATRIX
                        )   
    return norm, grads_vec                    
def draw_gradient_conflict(cosine_matrices_over_time,cancellation_ratios_over_time,monitor,task_names,epoch,path=""):
    all_cosines = torch.stack(cosine_matrices_over_time, dim=0).cpu().numpy()
   


    last_cos = torch.from_numpy(all_cosines[-1])
    monitor.plot_cosine_heatmap(
        last_cos,
        task_names=task_names,
        title=f"Epoch {epoch} — Grad Cosine (Last Batch)",
        save_path=f"heatmap_epoch{epoch}.png"
    )

    
    avg_cos = torch.from_numpy(all_cosines.mean(axis=0))
    monitor.plot_cosine_heatmap(
        avg_cos,
        task_names=task_names,
        title=f"Epoch {epoch} — Avg Grad Cosine",
        save_path=f"heatmap_epoch{epoch}_avg.png"
    )
    if cancellation_ratios_over_time is not None:
        all_cancellations = np.array(cancellation_ratios_over_time)
        monitor.plot_cancellation_ratios(
        all_cancellations.tolist(),
        epochs=list(range(1, len(all_cancellations) + 1)),
        title=f"Epoch {epoch} — Cancellation Ratio per Batch",
        save_path=f"cancellation_epoch{epoch}.png"
    )
def collect_grads(model,loss_dict,tasks):
    grad_hooks = {}
    task_grads = []
    for t in tasks:
        grad_hooks[t] = GradCaptureHook(model, t)
        g_t = grad_hooks[t].capture(loss_dict[t]) 
        task_grads.append(g_t)
    return task_grads, grad_hooks    
class GradCaptureHook:
    def __init__(self, model, task_name):
        self.model = model
        self.task_name = task_name
        self.grads = None

    def capture(self, loss):
     
        for p in self.model.parameters():
            p.grad = None

       
        loss.backward(retain_graph=True)

   
        grads = []
        for p in self.model.parameters():
            if p.requires_grad:
                grads.append(p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1))
        self.grads = torch.cat(grads)

   
        for p in self.model.parameters():
            p.grad = None

        return self.grads

def compute_conflict_lambdas(task_grads, eps=1e-8):
   
        """n = len(task_grads)
        device = task_grads[0].device

 
        g_mean = sum(task_grads) / n
        norm_mean = torch.norm(g_mean) + eps"""
        n = len(task_grads)
        device = task_grads[0].device
        dtype = task_grads[0].dtype
        max_len = max(g.numel() for g in task_grads)
        padded_grads = []

        for i, g in enumerate(task_grads):
            if torch.isnan(g).any() or torch.isinf(g).any():
                #print(f"[Warning] Task {i} gradient has NaN or Inf — replacing with zeros.")
                g = torch.zeros_like(g)

            if g.numel() < max_len:
                pad = torch.zeros(max_len - g.numel(), device=device, dtype=dtype)
                g = torch.cat([g, pad], dim=0)
            padded_grads.append(g)


        """g_mean = sum(padded_grads) / n
        norm_mean = torch.norm(g_mean) + eps

  
        lambdas = torch.zeros(n, device=device)
        for i, g_i in enumerate(task_grads):
            norm_i = torch.norm(g_i) + eps
            a_i = torch.dot(g_i, g_mean) / (norm_i * norm_mean)
            lambdas[i] = (1 - a_i) / (1 + a_i + eps)"""
        padded_grads = torch.stack(padded_grads) 

        g_mean = padded_grads.mean(dim=0)  
        norm_mean = g_mean.norm() + eps

        lambdas = torch.zeros(n, device=device, dtype=dtype)
        for i in range(n):
            g_i = padded_grads[i]
            #print("g_i ",g_i)
            norm_i = g_i.norm() + eps
            #print("norm_i ",norm_i)
            #print("g_mean ",g_mean)
            #print("norm_mean ",norm_mean)
            cos_sim = torch.dot(g_i, g_mean) / (norm_i * norm_mean)
            #print("cos_sim ",cos_sim)
            lambdas[i] = (1 - cos_sim) / (1 + cos_sim + eps)    

        return lambdas 
class GradientConflictMonitor:
    """
    A class to compute and visualize gradient conflict metrics:
      - Pairwise cosine similarity matrix (heatmap).
      - Cancellation ratio.

    Methods
    -------
    compute_cosine_matrix(task_grads: List[torch.Tensor]) -> torch.Tensor
        Compute the pairwise cosine similarity matrix for given task gradient vectors.

    compute_cancellation_ratio(task_grads: List[torch.Tensor]) -> float
        Compute the cancellation ratio: ||sum_i g_i|| / sum_i ||g_i||.

    plot_cosine_heatmap(
        cosine_matrix: Union[torch.Tensor, np.ndarray],
        task_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (6, 5),
        cmap: str = 'viridis',
        vmin: float = -1.0,
        vmax: float = 1.0,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None
        Plot a heatmap of the cosine similarity matrix.

    plot_cancellation_ratios(
        ratios: List[float],
        epochs: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (6, 4),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None
        Plot cancellation ratios over epochs (or arbitrary steps).
    """

    def __init__(self, task_names=None):
      
        if task_names is not None:
            self.task_names = list(task_names)
        else:
            self.task_names = None
     

    @staticmethod
    def compute_cosine_matrix(task_grads):
        """
        Compute the pairwise cosine similarity matrix between task gradients.

        Parameters:
        task_grads :
            A list of 1D tensors, each representing the flattened gradient vector for one task.

        Returns:
            A (n_tasks, n_tasks) tensor where entry (i, j) is the cosine similarity between g_i and g_j.
        """
        """
        n = len(task_grads)
        cosine_matrix = torch.zeros((n, n), dtype=torch.float32, device=task_grads[0].device)

        for i in range(n):
            g_i = task_grads[i]
            norm_i = torch.norm(g_i) + 1e-8
            for j in range(n):
                g_j = task_grads[j]
                norm_j = torch.norm(g_j) + 1e-8
                cosine_matrix[i, j] = torch.dot(g_i, g_j) / (norm_i * norm_j)
        """
        n = len(task_grads)
        device = task_grads[0].device
        dtype = task_grads[0].dtype
        max_len = max(g.numel() for g in task_grads)
        padded_grads = []

        for g in task_grads:
            if g.numel() < max_len:
                pad = torch.zeros(max_len - g.numel(), device=device, dtype=dtype)
                g = torch.cat([g, pad], dim=0)
            padded_grads.append(g)

        grad_matrix = torch.stack(padded_grads)  
        norm = grad_matrix.norm(dim=1, keepdim=True) + 1e-8
        grad_matrix_normalized = grad_matrix / norm

   
        cosine_matrix = grad_matrix_normalized @ grad_matrix_normalized.T
        cosine_matrix = cosine_matrix.clamp(min=-1.0, max=1.0)
        return cosine_matrix

    @staticmethod
    def compute_cancellation_ratio(task_grads):
        """
        Compute the cancellation ratio: ||sum_i g_i|| / sum_i ||g_i||.

        Parameters:
        task_grads :
            A list of 1D tensors, each representing the flattened gradient vector for one task.

        Returns
            The cancellation ratio (a scalar between 0 and 1).
        """
       
        sum_grad = torch.zeros_like(task_grads[0])
        sum_norms = 0.0
        for g in task_grads:
            sum_grad += g
            sum_norms += torch.norm(g).item()

        norm_sum_grad = torch.norm(sum_grad).item()
        cancellation_ratio = norm_sum_grad / (sum_norms + 1e-8)
        return cancellation_ratio

    """def plot_cosine_heatmap(
        self,
        cosine_matrix,
        task_names=None,
        figsize=(6, 5),
        cmap='viridis',
        vmin=-1.0,
        vmax=1.0,
        title=None,
        save_path=None
    ):
        if isinstance(cosine_matrix, torch.Tensor):
            cm = cosine_matrix.detach().cpu().numpy()
        else:
            cm = np.array(cosine_matrix)

        n = cm.shape[0]
        if task_names is None:
            if self.task_names is not None:
                labels = self.task_names
            else:
                labels = [f'T{i}' for i in range(n)]
        else:
            labels = task_names

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Task')
        ax.set_ylabel('Task')
        if title:
            ax.set_title(title)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()"""

    def plot_cancellation_ratios(
        self,
        ratios,
        epochs=None,
        figsize=(6, 4),
        title=None,
        save_path=None
    ):
       
        if epochs is None:
            x = list(range(1, len(ratios) + 1))
            xlabel = 'Epoch'
        else:
            x = epochs
            xlabel = 'Step'

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, ratios, marker='o', linestyle='-')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cancellation Ratio')
        if title:
            ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_cosine_heatmap(
    self,
    cosine_matrix,
    task_names=None,
    figsize=(6, 5),
    cmap='coolwarm',  
    vmin=-1.0,
    vmax=1.0,
    title=None,
    save_path=None
):
 
        if isinstance(cosine_matrix, torch.Tensor):
            cm = cosine_matrix.detach().cpu().numpy()
        else:
            cm = np.array(cosine_matrix)

        n = cm.shape[0]
        if task_names is None:
            labels = self.task_names if getattr(self, 'task_names', None) else [f'T{i}' for i in range(n)]
        else:
            labels = task_names

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)


        for i in range(n):
            for j in range(n):
                value = cm[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Task')
        ax.set_ylabel('Task')
        if title:
            ax.set_title(title)

 
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
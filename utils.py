from argparse import ArgumentParser

import torch
from torch import inf


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device)
                 for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def l1_loss(output, target):
    return torch.nn.functional.l1_loss(output, target)


def l2_loss(output, target):
    return torch.nn.functional.mse_loss(output, target)


def channel_wise_normalize(feature_maps):
    return [
        (feature_map - feature_map.mean(dim=1, keepdim=True))
        / (feature_map.std(dim=1, keepdim=True) + 1e-5)
        for feature_map in feature_maps
    ]


def weighted_feature_aggregation(feature_maps):
    weights = [1 / f.shape[1] for f in feature_maps]
    normalized_weights = [w / sum(weights) for w in weights]
    return [w * f for w, f in zip(normalized_weights, feature_maps)]


def normalize_feature_maps(feature_maps):
    return [f / f.norm() for f in feature_maps]


class MMDLoss(torch.nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / \
                (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        source = source.view(source.size(0), -1)
        target = target.view(target.size(0), -1)
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        adaptive_lambda=False,
        draw_cosine_matrix=False,
    ):
        parameters = list(parameters)
        grads = torch.autograd.grad(
            outputs=self._scaler.scale(loss),
            inputs=parameters,
            create_graph=create_graph,
            allow_unused=True,
        )
        grads_list=[]
        for p, g in zip(parameters, grads):
            p.grad = g
            if g is not None:
                grads_list.append(g.detach().contiguous().view(-1))
        grad_vec = torch.cat(grads_list if grads_list else None)       
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        if adaptive_lambda or draw_cosine_matrix:
            return norm, grad_vec    
        else: return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def load_checkpoint(config, logger, lora_enabled, model):
    model_state = torch.load(config.MODEL.RESUME, map_location="cpu")["model"]

    is_swin = "swin" in config.MODEL.TYPE

    # ConvLoRA wraps the conv as .conv, so rename the conv keys of a plain
    # backbone checkpoint
    if lora_enabled and config.MODEL.RESUME_BACKBONE:
        lora_keys = [
            k.split("backbone.")[-1]
            for k in model.state_dict().keys()
            if "lora" in k
        ]

        def map_conv_state_dict_keys(model, state_dict, all_lora_keys):
            mapping = {}

            lora_prefixes = set(
                [k.split(".lora")[0] for k in all_lora_keys]
            )

            conv_keys = set()

            for k in model.backbone.state_dict().keys():
                for p in lora_prefixes:
                    if k.startswith(p) and "conv" in k:
                        conv_keys.add(k.split("conv")[0])

            for k in state_dict.keys():
                for p in conv_keys:
                    if k.startswith(p):
                        typ = (
                            "weight"
                            if k.endswith("weight")
                            else ("bias" if k.endswith("bias") else None)
                        )

                        if typ is not None:
                            new_key = ".".join(
                                k.split(typ)[:-1]
                            ) + f"conv.{typ}"

                            mapping[k] = new_key

            return {
                (k if k not in mapping else mapping[k]): v
                for k, v in state_dict.items()
            }

        model_state = map_conv_state_dict_keys(
            model,
            model_state,
            lora_keys,
        )

    if is_swin:
        for k in [k for k in model_state.keys() if "attn_mask" in k]:
            del model_state[k]

        model_state.pop("head.weight", None)
        model_state.pop("head.bias", None)

        if lora_enabled:
            # shared LoRA uses the plain Swin patch_embed.proj, so rename
            # patch_embed.proj.conv.* keys saved by the CRLoMTL model
            if config.METHOD == "shared_lora":
                if "backbone.patch_embed.proj.conv.weight" in model_state:
                    model_state["backbone.patch_embed.proj.weight"] = model_state.pop(
                        "backbone.patch_embed.proj.conv.weight"
                    )

                if "backbone.patch_embed.proj.conv.bias" in model_state:
                    model_state["backbone.patch_embed.proj.bias"] = model_state.pop(
                        "backbone.patch_embed.proj.conv.bias"
                    )

                if "patch_embed.proj.conv.weight" in model_state:
                    model_state["patch_embed.proj.weight"] = model_state.pop(
                        "patch_embed.proj.conv.weight"
                    )

                if "patch_embed.proj.conv.bias" in model_state:
                    model_state["patch_embed.proj.bias"] = model_state.pop(
                        "patch_embed.proj.conv.bias"
                    )

            else:
                swin_conv_keys = [
                    "backbone.patch_embed.proj.weight",
                    "backbone.patch_embed.proj.bias",
                ]

                for k in swin_conv_keys:
                    if k in model_state:
                        model_state[k.replace("proj", "proj.conv")] = (
                            model_state.pop(k)
                        )

        if config.MODEL.MTI and not config.MODEL.RESUME_BACKBONE:
            for k in [k for k in model_state.keys() if "backbone" not in k]:
                del model_state[k]

    if config.MODEL.RESUME_BACKBONE:
        # If loading only backbone, remove "backbone." prefix if checkpoint has it.
        if any(k.startswith("backbone.") for k in model_state.keys()):
            model_state = {
                k.replace("backbone.", "", 1): v
                for k, v in model_state.items()
                if k.startswith("backbone.")
            }

        missing, unexpected = model.backbone.load_state_dict(
            model_state,
            strict=False,
        )

    else:

        missing, unexpected = model.load_state_dict(
            model_state,
            strict=False,
        )

    logger.info("==Missing Keys==")
    for m in missing:
        logger.info(m)
    logger.info("==End Missing Keys==")

    logger.info("==Unexpected Keys==")
    for u in unexpected:
        logger.info(u)
    logger.info("==End Unexpected Keys==")

    # report LoRA keys that were not loaded
    if lora_enabled:
        missing_lora = [
            k for k in missing
            if "lora_A" in k or "lora_B" in k
        ]

        unexpected_lora = [
            k for k in unexpected
            if "lora_A" in k or "lora_B" in k
        ]

        logger.info("==LoRA Loading Debug==")
        logger.info(f"Number of missing LoRA keys: {len(missing_lora)}")
        logger.info(f"Number of unexpected LoRA keys: {len(unexpected_lora)}")
        logger.info(f"First missing LoRA keys: {missing_lora[:30]}")
        logger.info(f"First unexpected LoRA keys: {unexpected_lora[:30]}")
        logger.info("==End LoRA Loading Debug==")

    logger.info(
        f"Resumed {'backbone' if config.MODEL.RESUME_BACKBONE else ''} "
        f"from {config.MODEL.RESUME}"
    )

    if config.MODEL.RESUME_LORA:
        model.load_state_dict(
            torch.load(config.MODEL.RESUME_LORA, map_location="cpu")["model"],
            strict=False,
        )
        logger.info(f"Resumed LoRA from {config.MODEL.RESUME_LORA}")
        
def create_cmd_parser(models, methods, dataset_loaders):
    parser = ArgumentParser("CRLoMTL")
    parser.add_argument("--cfg", type=str, metavar="FILE",
                        help="path to config file")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        nargs="+",
    )
    parser.add_argument(
        "--method", type=str, choices=list(set(methods)), help="Optimization method"
    )

    parser.add_argument(
        "--model", type=str, choices=list(set(models)), help="Model type"
    )

    parser.add_argument(
        "--accumulation_steps", type=int, help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--tasks", type=str, help="Comma-seperated list of tasks to override"
    )

    parser.add_argument("--lora_r", type=int, help="LoRA rank")

    parser.add_argument("--lora_alpha", type=float, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    parser.add_argument(
        "--crlomtl_mode",
        type=str,
        choices=["pretrain", "alternating_with_interm", "frozen"],
        help="CRLoMTL training mode",
    )
    parser.add_argument("--crlomtl_resolver", type=str,
                        help="CRLoMTL resolver")
    parser.add_argument(
        "--crlomtl_full_rank_resolver",
        action="store_true",
        help="CRLoMTL full rank resolver",
    )
    parser.add_argument(
        "--crlomtl_unfreeze_backbone",
        action="store_true",
        help="Unfreeze the backbone during CRLoMTL training",
    )
    parser.add_argument(
        "--crlomtl_cosine_loss",
        action="store_true",
        help="Use cosine loss for CRLoMTL training",
    )
    parser.add_argument(
        "--crlomtl_feature_regularization",
        type=str,
        choices=["l1", "l2", "CE", "MMD", "none"],
        help="Feature regularization for CRLoMTL training",
    )
    parser.add_argument(
        "--crlomtl_feature_regularization_weight",
        type=float,
        help="Feature regularization weight for CRLoMTL training",
    )
    parser.add_argument(
        "--dataset", type=str, choices=list(dataset_loaders.keys()), help="Dataset"
    )

    parser.add_argument("--data_path", type=str, help="Data directory")

    parser.add_argument("--seed", type=int, help="Random seed")

    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--method_lr", type=float, help="Learning rate")

    parser.add_argument("--eval", action="store_true",
                        help="Evaluate and exit")
    parser.add_argument(
        "--unfreeze_all", action="store_true", help="Unfreeze all layers"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone"
    )

    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, help="Eval batch size")

    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--local_rank", type=int, help="Local rank")

    parser.add_argument("--device", type=str, help="Device")

    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--resume", type=str,
                        help="Resume training from checkpoint")
    parser.add_argument("--resume_optimizer", type=str,
                        help="Resume optimizer training from checkpoint")
    parser.add_argument("--resume_scheduler", type=str,
                        help="Resume scheduler training from checkpoint")
    parser.add_argument(
        "--resume_lora", type=str, help="Resume training from LoRA checkpoint"
    )
    parser.add_argument(
        "--resume_backbone",
        action="store_true",
        help="Resume training from backbone checkpoint",
    )

    parser.add_argument(
        "--eval_and_save_freq", type=int, help="Evaluate and save model every N epochs"
    )
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--output", type=str)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb")
    parser.add_argument(
        "--adaptive_lambda", action="store_true",
        help="Enable adaptive lambda for CRLoMTL")
    parser.add_argument(
        "--draw_cosine_matrix", action="store_true",
        help="Enable drawing heatmap for cosine matix.")    
    parser.add_argument(
        "--eval_task_specific",
        action="store_true",
        help="Evaluate task-specific passes",
    )
    parser.add_argument(
        "--disable_progress", action="store_true", help="Disable tqdm progress bar"
    )
    parser.add_argument(
        "--print_layers", action="store_true", help="Print trainable layers"
    )

    # Add wandb model saving options
    parser.add_argument("--wandb-save-model", action="store_true",
                        help="Save model checkpoints to wandb")
    parser.add_argument("--wandb-save-freq", type=int, default=None,
                        help="How often to save model checkpoints to wandb (in epochs)")

    parser.add_argument("--svd_analysis", action="store_true", help="Enable SVD conflict-suppression analysis for CRLoMTL resolver.")

    parser.add_argument( "--svd_analysis_freq", type=int, default=None, help="Frequency in epochs for SVD analysis collection.")                    
    parser.add_argument("--logging_training_time", action="store_true",help="logging time in external file")
    return parser

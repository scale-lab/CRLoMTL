import torch

from lr_scheduler import build_scheduler
from methods.weight_methods import WeightMethods
from mtl_loss_schemes import MultiTaskLoss, get_loss
from optimizer import build_optimizer
from utils import NativeScalerWithGradNormCount


WEIGHT_METHOD_NAMES = ["pcgrad", "cagrad", "nashmtl", "mgda", "imtl", "famo", "ls", "uw", "consmtl", "go4align", "fairgrad"]


def build_optimizer_scheduler_criterion(config, model, train_loader, tasks):
    loss_scaler = NativeScalerWithGradNormCount()

    use_weight_method = config.METHOD in WEIGHT_METHOD_NAMES
    wmethod = None

    if use_weight_method:
        wmethod = WeightMethods(
            method=config.METHOD,
            device=torch.device(config.DEVICE),
            n_tasks=len(tasks),
        )
        print(f"Using weight method: {config.METHOD}")

    if wmethod is not None:
        optim_config = {
            "eps": config.TRAIN.OPTIMIZER.EPS,
            "betas": config.TRAIN.OPTIMIZER.BETAS,
            "weight_decay": config.TRAIN.WEIGHT_DECAY,
        }

        optim_params = [
            dict(
                params=model.parameters(),
                **optim_config,
                lr=config.TRAIN.BASE_LR,
            )
        ]

        optim_params.append(
            dict(
                params=wmethod.parameters(),
                **optim_config,
                lr=config.TRAIN.METHOD_LR,
            )
        )

        optimizer = torch.optim.AdamW(optim_params)
    else:
        optimizer = build_optimizer(config=config, model=model)

    scheduler = build_scheduler(
        config=config,
        optimizer=optimizer,
        n_iter_per_epoch=len(train_loader)
        // (
            config.TRAIN.ACCUMULATION_STEPS
            if config.TRAIN.ACCUMULATION_STEPS > 1
            else 1
        ),
    )

    loss_fns = torch.nn.ModuleDict(
        {
            task: get_loss(config["TASKS_CONFIG"], task, config)
            for task in config.TASKS
        }
    )

    criterion = MultiTaskLoss(loss_fns, config.LOSS_WEIGHTS)

    return optimizer, scheduler, criterion, loss_scaler, wmethod, use_weight_method
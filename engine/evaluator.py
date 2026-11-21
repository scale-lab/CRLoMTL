import logging

import torch
from timm.utils import AverageMeter
from tqdm import tqdm

from analysis_utils import unpack_batch
from evaluation.evaluate_utils import PerformanceMeter, get_output


@torch.no_grad()
def evaluate(
    config,
    model,
    val_loader,
    tasks,
    device,
    criterion,
    single_pass=True,
    eval_task_specific=False,
    disable_progress=False,
):
    model.eval()

    performance_meter = PerformanceMeter(config, config.DATA.DATASET)

    logging.info(
        f"Evaluating {'in a single pass' if single_pass and not eval_task_specific else 'in multiple passes'}..."
    )

    loss_meters = {task: AverageMeter() for task in tasks}

    for idx, batch in enumerate(
        tqdm(val_loader, desc="Batches", leave=False, disable=disable_progress)
    ):
        img, labels = unpack_batch(batch, config)

        img = img.to(device, non_blocking=True)
        labels = {
            k: v.to(device, non_blocking=True)
            for k, v in labels.items()
        }

        if single_pass and not eval_task_specific:
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                predictions = model(img)
                _, loss_dict = criterion(predictions, labels)

                for task in predictions.keys():
                    loss_meters[task].update(loss_dict[task].item())

            performance_meter.update(
                {
                    task: get_output(predictions[task], task, config.DATA.DATASET)
                    for task in config.TASKS
                },
                labels,
            )

        else:
            for task in tasks:
                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    predictions = model(img, task=task)
                    _, task_loss_dict = criterion(predictions, labels)

                    for pred_task in predictions.keys():
                        loss_meters[pred_task].update(
                            task_loss_dict[pred_task].item()
                        )

                performance_meter.update(
                    {
                        task: get_output(predictions[task], task, config.DATA.DATASET)
                    },
                    labels,
                )

    return performance_meter, loss_meters
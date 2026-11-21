#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os

import numpy as np
import torch
import torch.nn.functional as F


def get_output(output, task, database=None):
    if len(output.shape) == 4:
        output = output.permute(0, 2, 3, 1)

    if database == "CelebA":
        # SwinCelebA returns logits. Convert to probabilities.
        return torch.sigmoid(output)

    if task == "normals":
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {"semseg", "human_parts"}:
        _, output = torch.max(output, dim=3)

    elif task in {"edge", "sal"}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

    elif task in {"depth", "task_L", "task_R"}:
        pass

    else:
        raise ValueError(f"Select one of the valid tasks, got {task}")

    return output
class CelebAAttributeMeter(object):
    """
    CelebA binary attribute metric accumulator for one attribute.

    pred: probability tensor [B] or [B, 1]
    gt: binary tensor [B] or [B, 1]
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.correct = 0.0
        self.total = 0.0

    def update(self, pred, gt):
        pred = pred.detach()
        gt = gt.detach()

        if pred.dim() > 1:
            pred = pred.view(-1)

        if gt.dim() > 1:
            gt = gt.view(-1)

        pred_label = pred.gt(0.5).float()
        gt_label = gt.float()

        self.tp += (pred_label * gt_label).sum().item()
        self.fp += (pred_label * (1.0 - gt_label)).sum().item()
        self.fn += ((1.0 - pred_label) * gt_label).sum().item()

        self.correct += (pred_label == gt_label).float().sum().item()
        self.total += gt_label.numel()

    def get_score(self, verbose=True):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        acc = self.correct / (self.total + 1e-8)

        return {
            "f1": f1,
            "acc": acc,
            "precision": precision,
            "recall": recall,
        }    
class PerformanceMeter(object):
    """A general performance meter which shows performance across one or more tasks"""

    def __init__(self, config, db_name="NYUD"):
        self.database = db_name
        self.tasks = config.TASKS
        self.meters = {
            t: get_single_task_meter(config, t, self.database) for t in self.tasks
        }

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in pred.keys():
            self.meters[t].update(pred[t], gt[t])

    def get_task_scores(self, epoch=None, verbose=True):
        eval_results = self.get_score(verbose=verbose)
        scores_logs = {
            "val/epoch": epoch,
        }

        # ---------------------------------------------------------------------
        # CelebA: each task returns a dict with f1/acc/precision/recall.
        # Add per-attribute logs and mean metrics over all 40 attributes.
        # ---------------------------------------------------------------------
        if self.database == "CelebA":
            f1_values = []
            acc_values = []

            for task in self.tasks:
                if task not in eval_results:
                    continue

                result = eval_results[task]

                if isinstance(result, dict):
                    if "f1" in result:
                        scores_logs[f"val/tasks/{task}/f1"] = 100.0 * result["f1"]
                        f1_values.append(result["f1"])

                    if "acc" in result:
                        scores_logs[f"val/tasks/{task}/acc"] = 100.0 * result["acc"]
                        acc_values.append(result["acc"])

                    if "precision" in result:
                        scores_logs[f"val/tasks/{task}/precision"] = (
                            100.0 * result["precision"]
                        )

                    if "recall" in result:
                        scores_logs[f"val/tasks/{task}/recall"] = (
                            100.0 * result["recall"]
                        )

            if len(f1_values) > 0:
                scores_logs["val/tasks/CelebA/mean_f1"] = 100.0 * np.mean(f1_values)

            if len(acc_values) > 0:
                scores_logs["val/tasks/CelebA/mean_acc"] = 100.0 * np.mean(acc_values)

            return scores_logs

        if "semseg" in eval_results:
            scores_logs["val/tasks/semseg/mIoU"] = (
                100.0 * eval_results["semseg"]["mIoU"]
            )
        if "normals" in eval_results:
            scores_logs["val/tasks/normals/mean"] = eval_results["normals"]["mean"]
            scores_logs["val/tasks/normals/rmse"] = eval_results["normals"]["rmse"]
        if "human_parts" in eval_results:
            scores_logs["val/tasks/human_parts/mIoU"] = (
                100.0 * eval_results["human_parts"]["mIoU"]
            )
        if "sal" in eval_results:
            scores_logs["val/tasks/sal/maxF"] = 100.0 * eval_results["sal"]["maxF"]
            scores_logs["val/tasks/sal/mIoU"] = 100.0 * eval_results["sal"]["mIoU"]
        if "edge" in eval_results:
            scores_logs["val/tasks/sal/loss"] = eval_results["edge"]["loss"]
        if "depth" in eval_results:
            scores_logs["val/tasks/depth/rmse"] = eval_results["depth"]["rmse"]
            scores_logs["val/tasks/depth/log_rmse"] = eval_results["depth"]["log_rmse"]
        if "task_L" in eval_results:
            scores_logs["val/tasks/task_L/accuracy"] = eval_results["task_L"]
        if "task_R" in eval_results:
            scores_logs["val/tasks/task_R/accuracy"] = eval_results["task_R"]
        if "Average" in eval_results:
            scores_logs["val/tasks/Average"] = eval_results["Average"]
        return scores_logs

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        # ---------------------------------------------------------------------
        # CelebA: keep every attribute score, then add mean over all attributes.
        # ---------------------------------------------------------------------
        if self.database == "CelebA":
            f1_values = []
            acc_values = []

            for t in self.tasks:
                result = eval_dict[t]

                if isinstance(result, dict):
                    if "f1" in result:
                        f1_values.append(result["f1"])
                    if "acc" in result:
                        acc_values.append(result["acc"])

            if len(f1_values) > 0:
                eval_dict["mean_f1"] = float(np.mean(f1_values))

            if len(acc_values) > 0:
                eval_dict["mean_acc"] = float(np.mean(acc_values))

        elif "task_L" in eval_dict or "task_R" in eval_dict:
            eval_dict["Average"] = np.mean(
                [
                    v
                    for v in eval_dict.values()
                    if isinstance(v, (int, float, np.number))
                ]
            )

        return eval_dict

def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert set(eval_dict.keys()) == set(single_task_dict.keys())
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == "depth":  # rmse lower is better
            mtl_performance -= (mtl["rmse"] - stl["rmse"]) / stl["rmse"]

        elif task in ["semseg", "sal", "human_parts"]:  # mIoU higher is better
            mtl_performance += (mtl["mIoU"] - stl["mIoU"]) / stl["mIoU"]

        elif task == "normals":  # mean error lower is better
            mtl_performance -= (mtl["mean"] - stl["mean"]) / stl["mean"]

        elif task == "edge":  # odsF higher is better
            mtl_performance += (mtl["odsF"] - stl["odsF"]) / stl["odsF"]

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks


def get_single_task_meter(config, task, database="NYUD"):
    """Retrieve a meter to measure the single-task performance"""

    if task == "semseg":
        from evaluation.eval_semseg import SemsegMeter

        return SemsegMeter(database, config)

    elif task == "human_parts":
        from evaluation.eval_human_parts import HumanPartsMeter

        return HumanPartsMeter(database)

    elif task == "normals":
        from evaluation.eval_normals import NormalsMeter

        return NormalsMeter()

    elif task == "sal":
        from evaluation.eval_sal import SaliencyMeter

        return SaliencyMeter()

    elif task == "depth":
        from evaluation.eval_depth import DepthMeter

        return DepthMeter()
    elif task in ["task_L", "task_R"]:
        from evaluation.eval_classification import ClassificationMeter

        return ClassificationMeter(10)
        

    # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
    elif task == "edge":
        from evaluation.eval_edge import EdgeMeter

    elif database == "CelebA":
        return CelebAAttributeMeter()    

        # TODO: fix edge validation
        return EdgeMeter(pos_weight=0.95)
        # return EdgeMeter()

    else:
        raise NotImplementedError

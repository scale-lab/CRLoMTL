import json
from pathlib import Path
import logging
import numpy as np

def get_single_task_performance(single_task_dict_path):
   
    with open(single_task_dict_path, 'r') as f:
        raw_data = json.load(f)

  
    metric_map = {
        "semseg": "mIoU",
        "human_parts": "mIoU",
        "sal": "mIoU",
        "normals": "rmse",
        "depth": "rmse",
    }

    single_task_dict = {}
    for task, value in raw_data.items():
        metric = metric_map.get(task)
        if metric:
            single_task_dict[task] = {metric: value}


    return single_task_dict

def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert(set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == 'depth':  # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse'])/stl['rmse']
           
    
        elif task in ['semseg','human_parts','sal']:  # mIoU/F higher is better    
            mtl_performance += (mtl['mIoU'] - stl['mIoU'])/stl['mIoU']
          
        #elif task =='sal':  # mIoU/F higher is better
        #    mtl_performance += (mtl['maxF'] - stl['maxF'])/stl['maxF']    
    
        elif task == 'normals':  # mean error lower is better
            #mtl_performance -= (mtl['mean'] - stl['mean'])/stl['mean']
            mtl_performance -= (mtl['rmse'] - stl['rmse'])/stl['rmse']
         
        elif task == 'edge':  # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF'])/stl['odsF']
          
        else:
            raise NotImplementedError

    return mtl_performance / num_tasks

def extract_eval_dict_for_mtl_performance(scores, tasks):
    """
    Converts meter.get_task_scores(...) output into the format expected by
    calculate_multi_task_performance().

    Expected output:
        {
            "semseg": {"mIoU": value},
            "human_parts": {"mIoU": value},
            "sal": {"mIoU": value},
            "normals": {"rmse": value},
        }
    """
    metric_map = {
        "semseg": "mIoU",
        "human_parts": "mIoU",
        "sal": "mIoU",
        "normals": "rmse",
        "depth": "rmse",
        "edge": "odsF",
    }

    eval_dict = {}

    for task in tasks:
        metric = metric_map.get(task)
        if metric is None:
            continue

        possible_keys = [
            f"{task}/{metric}",
            f"val/{task}/{metric}",
            f"val/tasks/{task}/{metric}",
            f"{task}_{metric}",
            f"{task}-{metric}",
            f"{task} {metric}",
            metric if len(tasks) == 1 else None,
        ]

        value = None
        for key in possible_keys:
            if key is not None and key in scores:
                value = scores[key]
                break

        # Fallback: search any key containing both task and metric.
        if value is None:
            for key, score_value in scores.items():
                key_lower = str(key).lower()
                if task.lower() in key_lower and metric.lower() in key_lower:
                    value = score_value
                    break

        if value is not None:
            eval_dict[task] = {metric: float(value)}
    return eval_dict


# single-task reference numbers used for delta_m
SINGLE_TASK_FILES = {
    "PASCALContext": "single_task_performance.json",
    "NYUD": "single_task_nyud.json",
}


def load_single_task_performance(dataset="PASCALContext"):
    if dataset not in SINGLE_TASK_FILES:
        return None
    single_task_dict_path = Path(__file__).resolve().parent / SINGLE_TASK_FILES[dataset]
    single_task_dict = get_single_task_performance(single_task_dict_path)
    logging.info(f"Loaded single-task performance from: {single_task_dict_path}")
    logging.info(f"single_task_dict: {single_task_dict}")
    return single_task_dict


def load_stl_f1(stl_json_path):

    with open(stl_json_path, "r") as f:

        data = json.load(f)

    stl_f1 = {}

    for task, result in data["tasks"].items():

        if "f1" in result and "error" not in result:

            stl_f1[task] = float(result["f1"])

    return stl_f1

def compute_celeba_delta_m(stl_f1_by_task, method_f1_by_task):

    common_tasks = sorted(set(stl_f1_by_task) & set(method_f1_by_task))

    if len(common_tasks) == 0:

        raise ValueError("No common tasks between STL and method results.")

    stl = np.array([stl_f1_by_task[t] for t in common_tasks], dtype=np.float64)

    method = np.array([method_f1_by_task[t] for t in common_tasks], dtype=np.float64)

    delta_i = np.mean((stl - method) / stl) * 100.0
    delta_m = np.mean((method - stl) / (stl )) * 100.0

    return {

        "delta_m": float(delta_m),
        "delta_i": float(delta_i),
        "num_tasks": len(common_tasks),

        "tasks": common_tasks,

        "stl_f1": {t: float(stl_f1_by_task[t]) for t in common_tasks},

        "method_f1": {t: float(method_f1_by_task[t]) for t in common_tasks},

    }
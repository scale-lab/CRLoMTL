import torch


def classification_accuracy(logits, gt):
    return ((logits.argmax(dim=-1) == gt).float()).mean()


class ClassificationMeter(object):
    """Performance meter for single classification task"""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.accuracies = []

    def reset(self):
        self.accuracies = []

    def update(self, pred, gt):
        self.accuracies.append(classification_accuracy(pred, gt).item())

    def get_score(self, verbose=True):
        return 100.0 * sum(self.accuracies) / len(self.accuracies)

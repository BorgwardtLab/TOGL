"""Implementation of metrics."""
import torch
from typing import Any, Callable, Optional
from pytorch_lightning.metrics import Metric, Accuracy
from pytorch_lightning.metrics.functional import confusion_matrix
from pytorch_lightning.metrics.utils import _input_format_classification


class WeightedAccuracy(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        nb_classes = preds.shape[1]
        preds, target = _input_format_classification(
            preds, target, self.threshold)
        assert preds.shape == target.shape
        CM = confusion_matrix(
            preds, target, num_classes=nb_classes, normalize='true')
        n_elements = target.numel()
        self.correct += torch.mean(torch.diag(CM)) * n_elements
        self.total += n_elements

    def compute(self):
        """
        Computes accuracy over state.
        """
        return self.correct / self.total

"""Implementation of metrics."""
import torch
from typing import Any, Callable, Optional
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import confusion_matrix


class WeightedAccuracy(Metric):
    def __init__(
        self,
        n_classes,
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

        self.add_state("correct", default=torch.zeros(n_classes, dtype=int),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(
            n_classes, dtype=int), dist_reduce_fx="sum")

        self.threshold = threshold
        self.n_classes = n_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape[1] == self.n_classes

        preds, target = _input_format_classification(
            preds, target, self.threshold)
        assert preds.shape == target.shape
        CM = confusion_matrix(
            preds, target, num_classes=self.n_classes)
        self.correct += torch.diag(CM).long()
        self.total += torch.bincount(target, minlength=self.n_classes).long()

    def compute(self):
        """
        Computes accuracy over state.
        """
        return (self.correct.float() / self.total.float()).sum() / self.n_classes


def _input_format_classification(
        preds: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Convert preds and target tensors into label tensors
    Args:
        preds: either tensor with labels, tensor with probabilities/logits or
            multilabel tensor
        target: tensor with ground true labels
        threshold: float used for thresholding multilabel input
    Returns:
        preds: tensor with labels
        target: tensor with labels
    """
    if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )

    if len(preds.shape) == len(target.shape) + 1:
        # multi class probabilites
        preds = torch.argmax(preds, dim=1)

    if len(preds.shape) == len(target.shape) and preds.dtype == torch.float:
        # binary or multilabel probablities
        preds = (preds >= threshold).long()
    return preds, target

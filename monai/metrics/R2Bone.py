import warnings
from typing import Union

import torch

from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction

from monai.metrics import CumulativeIterationMetric


class R2BoneMetric(CumulativeIterationMetric):
    """
    Compute R2 in bone volumes between two tensors. A batch-first Tensor (BCHW[D]).

    Args:
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
    """

    def __init__(self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        raise NotImplementedError

        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn("y_pred should be a binarized tensor.")
        if not torch.all(y.byte() == y):
            warnings.warn("y should be a binarized tensor.")
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        bv_pred = y_pred.sum(dim=tuple(range(1, dims))).to(torch.float)
        bv_true = y.sum(dim=tuple(range(1, dims))).to(torch.float)

        result = 1 - ((bv_true - bv_pred)**2).sum() / (((bv_true - bv_true.mean())**2).sum()+1e-7)
        return result

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        """
        Execute reduction logic for the output.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return f


if __name__ == "__main__":
    shape = (100, 1, 96, 96, 96)
    y_true = torch.randint(0, 2, size=shape, dtype=torch.bool)
    y_pred = torch.randint(0, 2, size=shape, dtype=torch.bool)  # => negative
    acc_func = R2BoneMetric()
    print(acc_func(y_true, y_pred))
    y_pred = y_true  # => 1
    print(acc_func(y_true, y_pred))

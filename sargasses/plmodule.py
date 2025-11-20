from typing import Literal

import torch
from mfai.pytorch.lightning_modules.segmentation import (
    SegmentationLightningModule,
)
from mfai.pytorch.models.base import ModelABC
from torch import Tensor
from torchmetrics import AUROC

from sargasses.plots import plot_pred_and_target


class SargassesLightningModule(SegmentationLightningModule):
    """The sargasses project lightning Module.
    Implements mfai's `SegmentationLightningModule`, a lightning module
    specialized in segmentation tasks.
    """

    def __init__(
        self,
        model: ModelABC,
        segmentation_type: Literal["binary", "multiclass", "multilabel", "regression"],
        loss: torch.nn.modules.loss._Loss,  # type: ignore[reportPrivateUsage]
    ) -> None:
        """
        Args:
            model: A model that implement the mfai's interface.
            segmentation_type: Segmentation type, either 'binary',
                'multiclass', 'multilabel' or 'regression.
            loss: The loss to use in training.
        """
        super().__init__(model, segmentation_type, loss)  # type: ignore[reportArgumentType]

        self.metrics["AUROC"] = AUROC(task="binary", thresholds=100)

        self.save_hyperparameters()

    def val_plot_step(self, batch_idx: int, y: Tensor, y_hat: Tensor) -> None:
        """Plots images on first batch of validation and log them in tensorboard."""
        if self.logger is None:
            return

        interesting_batches = [6, 14, 48, 78]
        if batch_idx in interesting_batches:
            fig = plot_pred_and_target(y=y[0], y_hat=y_hat[0])

            tb = self.logger.experiment  # type: ignore[reporteAttributeAccessIssue]
            tb.add_figure(f"val_plots/test_figure_{batch_idx}", fig, self.current_epoch)

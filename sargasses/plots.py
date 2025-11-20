from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from torch import Tensor

from sargasses.sample import Sample


def plot_xy(
    input_image: Tensor,
    target_mask: Tensor,
    crop: tuple[int, int, int, int] | None = None,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Plot with side by side an input image and a target mask.

    Args:
        input_image: The input image, shape (1, height, width).
        target_mask: The target mask, shape (channels, height, width).
        crop: Crop coordinates (top, left, right, bottom). Defaults to None.
        title: The plot's title. Defaults to "".
        save_path: Where to save the plot, or None. If None, shows the plot.
            Defaults to None.
    """

    # Crop
    if crop is not None:
        input_image = input_image[crop[0] : crop[1], crop[2] : crop[3]]
        target_mask = target_mask[crop[0] : crop[1], crop[2] : crop[3]]

    # Plot
    _, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].imshow(torch.moveaxis(input_image, 0, -1).int())
    axs[0].set_title("Input otci image.")
    axs[1].imshow(torch.moveaxis(target_mask, 0, -1).int())
    axs[1].set_title("Target mask.")

    # Title
    title = title if title != "" else "Sargasses input / target plot."
    plt.suptitle(title)

    # Save or show
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_sample(
    sample: Sample,
    crop: tuple[int, int, int, int] | None = None,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Plots side by side a sample's input image and a its target mask.

    Args:
        sample: The sample to plot.
        crop: Crop coordinates (top, left, right, bottom). Defaults to None.
        title: The plot's title. Defaults to "".
        save_path: Where to save the plot, or None. If None, shows the plot.
            Defaults to None.
    """

    plot_xy(
        input_image=torch.from_numpy(sample.otci),
        target_mask=torch.from_numpy(sample.label_mask),
        crop=crop,
        title=title,
        save_path=save_path,
    )


def plot_pred_and_target(y: Tensor, y_hat: Tensor) -> Figure:
    """Returns a plot of a prediction and target.

    Args:
        y: Target.
        y_hat: Model prediction.

    Returns:
        Plot with 2 axes, for the prediction and the target.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    image = axs[0].imshow(torch.squeeze(y_hat).cpu(), vmin=0, vmax=1, cmap="gist_yarg")
    axs[1].imshow(torch.squeeze(y).int().cpu(), vmin=0, vmax=1, cmap="gist_yarg")

    fig.colorbar(
        image,
        ax=axs,
        orientation="horizontal",
        fraction=0.05,
        location="bottom",
        shrink=0.7,
    )

    axs[0].set_title("Prediction")
    axs[1].set_title("Target")

    plt.close()

    return fig


def plot_prediction(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    x_otci: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plots a prediction.

    Args:
        true_positive: True positive mask where 1 if pixel ok else np.nan.
        false_positive: False positive mask where 1 if pixel ok else np.nan.
        false_negative: True negative mask where 1 if pixel ok else np.nan.
        x_otci: Input otci image.
        title: Plot title.
        save_path: Output plot file path.
    """
    # Calculate  False Alarm Rate and Probability Of Detection
    true_negative = true_positive.shape[0] * true_positive.shape[1]
    true_negative -= (
        np.nansum(true_positive) - np.nansum(false_positive) - np.nansum(false_negative)
    )
    false_alarm_rate = np.nansum(false_positive) / (
        np.nansum(false_positive) + np.nansum(true_negative)
    )
    probability_of_detection = np.nansum(true_positive) / (
        np.nansum(true_positive) + np.nansum(false_negative)
    )

    # define labels names and colors for plots
    masks_for_plots = [true_positive, false_positive, false_negative]
    colors_for_plots = ["lime", "magenta", "red"]
    labels_for_plots = ["Détections", "Fausses alarmes", "Non détections"]

    plt.figure(dpi=1000)
    plt.suptitle(f"Détection de sargasses pour image : {title}", fontsize=15)
    plt.title(
        f"Probabilities of - False Alarm: {false_alarm_rate:.2e} "
        f"- Detection: {probability_of_detection:.2f}",
        fontsize=10,
    )

    # plot background map with otci image
    plt.imshow(x_otci[0].astype(int), interpolation="none", alpha=0.4)

    # plot binary masks
    for i, mask in enumerate(masks_for_plots):
        cmp = ListedColormap(["white", colors_for_plots[i]])
        plt.imshow(mask, interpolation="none", cmap=cmp, vmin=0, vmax=1, alpha=1)

    # custom legends
    legend_elements = [
        Patch(facecolor=colors_for_plots[i], edgecolor="w", label=labels_for_plots[i])
        for i in range(3)
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    plt.savefig(save_path)


def compute_masks_tp_pf_fn(
    y: np.ndarray | Tensor, y_hat: np.ndarray | Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute three masks: true positives, false positives and false negative.

    Args:
        y: Target.
        y_hat: Model prediction.

    Returns:
        np.ndarray: True positive.
        np.ndarray: False positive.
        np.ndarray: False negative.
            1 if pixel ok else np.nan.
    """
    # Ensure np.ndarray type
    y = y if isinstance(y, np.ndarray) else y.numpy()
    y_hat = y_hat if isinstance(y_hat, np.ndarray) else y_hat.numpy()

    # Prepare mask
    y_hat = np.where(y_hat < 0.5, np.nan, 1)
    y = np.where(y[0] < 0.5, np.nan, 1)

    # Compute accuracy metrics binary mask
    # 1 if pixel ok else np.nan
    true_positives = np.where(y == y_hat, y, np.nan)
    false_positives = np.where((y_hat == 1) & (np.isnan(y)), y_hat, np.nan)
    false_negatives = np.where((np.isnan(y_hat)) & (y == 1), y, np.nan)

    return true_positives, false_positives, false_negatives

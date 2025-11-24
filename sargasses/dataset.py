from pathlib import Path, PurePosixPath
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from sargasses.sample import Sample
from sargasses.settings import (
    ALGAE_MASK_PATH,
    CROP_SIZE,
    CROPS_FILE,
    OTCI_PATH,
    PLOTS_PATH,
    SHAPE_IMGS,
)


class SargassesDataset(torch.utils.data.Dataset):
    """Dataset used by the Ligthning DataModule to load
    train, validation and test data.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        pct_in_train: float = 0.7,
    ):
        """Args:
        split: Determines if it's a train, val or test dataset.
        pct_in_train: Defines the percentage of the total dataset
            included in it's train subsection.
        """

        self.split = split
        self.pct_in_train = pct_in_train

        # Load input file paths
        self.algae_masks = [
            path
            for path in list(ALGAE_MASK_PATH.glob("*.npz"))
            if (OTCI_PATH / f"{path.name[:8]}S3_OTCI.png").exists()
        ]

        # Read dataframe with precomputed crops
        crops_dataframe = pd.read_csv(CROPS_FILE)

        # Filter dates for the different splits
        filenames = sorted(list(pd.unique(crops_dataframe.algae_mask_path)))
        num_day_month_test_or_train = (
            30 * self.pct_in_train
        )  # 30 is an approx of number of days per month
        dates_train, dates_valid_and_test = [], []
        for filename in filenames:
            date = PurePosixPath(filename).stem.split("_")[0]
            day = int(date[-2:])
            if day < num_day_month_test_or_train:
                dates_train.append(filename)
            else:
                dates_valid_and_test.append(filename)

        if split == "train":
            self.dates = dates_train
        elif split == "val":
            self.dates = dates_valid_and_test[: len(dates_valid_and_test) // 2]
        else:
            self.dates = dates_valid_and_test[len(dates_valid_and_test) // 2 :]

        # Filter and sort dataframe
        crops_dataframe = crops_dataframe[
            crops_dataframe["algae_mask_path"].isin(self.dates)
        ]
        self.crops_dataframe = crops_dataframe.sort_values(  # type: ignore[reportAttributeAccessIssue]
            "algae_mask_path", ascending=True
        )

    def __len__(self) -> int:
        """Required to define a torch.utils.data.Dataset."""
        return len(self.crops_dataframe.index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Required to define a torch.utils.data.Dataset."""

        # Retreive crop information
        df_crop = self.crops_dataframe.iloc[idx]
        algae_mask_path, left, top = df_crop[["algae_mask_path", "left", "top"]].values
        crop: tuple[int, int, int, int] = (top, left, CROP_SIZE[0], CROP_SIZE[1])

        # Retreive data
        sample = Sample(Path(algae_mask_path))
        x: Tensor
        y: Tensor
        x, y = sample.get_xy(crop)

        return Tensor(x), Tensor(y)


if __name__ == "__main__":
    from sargasses.plots import plot_sample, plot_xy

    """Dataset inspection.
    Helps developpers to peer into the inner working of the
    Dataset class and make manual checks.
    """
    path_label = ALGAE_MASK_PATH / "20210319_dm.npz"
    sample = Sample(path_label)

    # Plot whole sample
    plot_sample(sample)

    # Plot a small area
    plot_sample(sample, crop=(0, 512, 1500, 2012))
    plot_sample(sample, crop=(2500, 3012, 2000, 2512))

    # Create dataset and plot some samples
    dataset = SargassesDataset("train")
    for i in tqdm([100, 300, 20000]):
        x, y = dataset[i]
        plot_xy(x, y, title=f"Sample index: {i}")

    # Plots repartition of crops for all samples
    for path in tqdm(sorted(list(ALGAE_MASK_PATH.glob("*.npz")))):
        sample = Sample(path)
        df = dataset.crops_dataframe
        df = df[df["algae_mask_path"] == str(path)]

        res = np.zeros(SHAPE_IMGS)
        for i in df.index.values:
            df_crop = df.loc[i]
            left, top = df_crop[["left", "top"]].values
            res[top : top + 512, left : left + 512] += np.ones((512, 512))

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].imshow(sample.input_image)
        axs[0, 0].set_title("OTCI")
        axs[0, 1].imshow(sample.target_mask)
        axs[0, 1].set_title("Label Mask")
        axs[1, 1].imshow(res, vmin=0, vmax=30)
        axs[1, 1].set_title("Crops")
        fig.delaxes(axs[1, 0])
        plt.tight_layout()
        plt.savefig(PLOTS_PATH / f"{path.stem}.png")
        plt.close()

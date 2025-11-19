from pathlib import Path, PurePosixPath
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mfai.pytorch.transforms import RandomCropWithMinPositivePixels
from torch import Tensor
from tqdm import tqdm

from sargasses.sample import Sample
from sargasses.settings import (
    ALGAE_MASK_PATH,
    CROP_SIZE,
    CROPS_FILE,
    MIN_POSITIVE_PERC,
    PLOTS_PATH,
    SHAPE_IMGS,
    TRIES,
)

cropper = RandomCropWithMinPositivePixels(
    crop_size=CROP_SIZE, min_positive_percentage=MIN_POSITIVE_PERC, tries=TRIES
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

        # read df with precomputed crops
        df = pd.read_csv(CROPS_FILE)

        # filter dates for the different splits
        filenames = sorted(list(pd.unique(df.algae_mask_path)))
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
        df = df[df["algae_mask_path"].isin(self.dates)]
        self.df_crops = df.sort_values("algae_mask_path", ascending=True)  # type: ignore[reportAttributeAccessIssue]

    def __len__(self) -> int:
        """Required to define a torch.utils.data.Dataset."""
        return len(self.df_crops.index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Required to define a torch.utils.data.Dataset."""
        df_crop = self.df_crops.iloc[idx]
        algae_mask_path, left, top = df_crop[["algae_mask_path", "left", "top"]].values
        sample = Sample(Path(algae_mask_path))
        x, y = sample.get_cropped_xy(top, left)
        return Tensor(x), Tensor(y)

    def plot_xy(self, x: Tensor, y: Tensor, title: str = "") -> None:
        _, axs = plt.subplots(1, 2, figsize=(15, 7))
        axs[0].imshow(torch.moveaxis(x, 0, -1).int())
        axs[0].set_title("OTCI")
        axs[1].imshow(torch.moveaxis(y, 0, -1).int())
        axs[1].set_title("Label Mask")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    """Dataset inspection.
    Helps developpers to peer into the inner working of the
    Dataset class and make manual checks.
    """
    path_label = ALGAE_MASK_PATH / "20210319_dm.npz"
    sample = Sample(path_label)

    # Plot whole sample
    sample.plot(save=False)

    # Plot a small area
    sample.plot((0, 512, 1500, 2012), save=False)
    sample.plot((2500, 3012, 2000, 2512), save=False)

    # Create dataset and plot some samples
    dataset = SargassesDataset("train")
    for i in tqdm([100, 300, 20000]):
        x, y = dataset[i]
        dataset.plot_xy(x, y, f"Sample index: {i}")

    # Plots repartition of crops for all samples
    for path in tqdm(sorted(list(ALGAE_MASK_PATH.glob("*.npz")))):
        sample = Sample(path)
        df = dataset.df_crops
        df = df[df["algae_mask_path"] == str(path)]

        res = np.zeros(SHAPE_IMGS)
        for i in df.index.values:
            df_crop = df.loc[i]
            left, top = df_crop[["left", "top"]].values
            res[top : top + 512, left : left + 512] += np.ones((512, 512))

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].imshow(sample.otci)
        axs[0, 0].set_title("OTCI")
        axs[0, 1].imshow(sample.label_mask)
        axs[0, 1].set_title("Label Mask")
        axs[1, 1].imshow(res, vmin=0, vmax=30)
        axs[1, 1].set_title("Crops")
        fig.delaxes(axs[1, 0])
        plt.tight_layout()
        plt.savefig(PLOTS_PATH / f"{path.stem}.png")
        plt.close()

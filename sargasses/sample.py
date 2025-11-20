import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import xarray as xr
from mfai.pytorch.transforms import RandomCropWithMinPositivePixels
from PIL import Image
from torch import Tensor

from sargasses.settings import (
    CLOUD_MASK_PATH,
    CROP_SIZE,
    DATASET_PATH,
    MIN_POSITIVE_PERC,
    NETCDF_PATH,
    OTCI_PATH,
    PLOTS_PATH,
    SHAPE_IMGS,
    TRIES,
)

cropper = RandomCropWithMinPositivePixels(
    crop_size=CROP_SIZE, min_positive_percentage=MIN_POSITIVE_PERC, tries=TRIES
)


class Sample:
    """Sample class for the sargasses dataset.
    Responsibilities:
        - Load and make available an input image.
        - Load and make available an target mask.
    """

    def __init__(self, path_algae_mask: Path) -> None:
        """
        Args:
            path_algae_mask: Path to an algae mask.
        """
        self.path_algae_mask = path_algae_mask
        self.date_str = path_algae_mask.name[:8]
        self.date = datetime.strptime(self.date_str, "%Y%m%d")
        self.annotator = path_algae_mask.name.split("_")[1][:2]
        list_netcdf = list(NETCDF_PATH.glob(f"*__{self.date_str}T*"))
        self.nc_folder = list_netcdf[0] if list_netcdf != [] else None

    @property
    def label_mask(self) -> np.ndarray:
        """The sample's target mask.

        Returns:
            np.ndarray: The target mask, shape (height, width).
        """
        return np.load(self.path_algae_mask)["arr_0"]

    @property
    def otci(self) -> np.ndarray:
        """The sample's input image.

        Returns:
            np.ndarray: The input image, shape (height, width, channels)
        """

        filepath = list(OTCI_PATH.glob(f"{self.date_str}*.png"))[0]
        return self.load_otci_image(filepath)

    @staticmethod
    def load_otci_image(file_path: Path) -> np.ndarray:
        """Loads a png otci image.

        Args:
            file_path: path to a otci image `.png`.

        Returns:
            np.ndarray: The image of shape (height, width, channels).
        """

        with Image.open(file_path) as img:
            arr = np.asarray(img)[:, :, :3]

        # Reshape images that are smaller
        res = np.zeros((SHAPE_IMGS[0], SHAPE_IMGS[1], 3))
        res[: arr.shape[0], : arr.shape[1], :] = arr
        return res

    @property
    def cloud_mask(self) -> np.ndarray:
        filepath = list(CLOUD_MASK_PATH.glob(f"{self.date_str}*.nc"))[0]
        with xr.open_dataset(filepath) as ds:
            return ds.valid_mask.values

    @property
    def land_sea_mask(self) -> np.ndarray | None:
        if self.nc_folder is None:
            warnings.warn(
                "\nsargasses.dataset.Sample.__init__()\n\t"
                "no .nc found for date {self.date}"
            )
            return None

        with xr.open_dataset(self.nc_folder / "geo_coordinates.nc") as ds:
            alti = ds.altitude.values
            return np.where(alti > -30, 1, 0)

    def plot(self, crop: tuple[int, ...] | None = None, save: bool = True) -> None:
        _, axs = plt.subplots(1, 2, figsize=(15, 7))
        if crop is not None:
            otci = self.otci[crop[0] : crop[1], crop[2] : crop[3]]
            label = self.label_mask[crop[0] : crop[1], crop[2] : crop[3]]
        else:
            otci, label = self.otci, self.label_mask
        axs[0].imshow(otci)
        axs[0].set_title("OTCI")
        axs[1].imshow(label)
        axs[1].set_title("Label Mask")
        plt.suptitle(self.date_str)
        plt.tight_layout()
        if save:
            plt.savefig(PLOTS_PATH / f"{self.date_str}_{self.annotator}.png")
        else:
            plt.show()
        plt.close()

    def get_xy(self) -> tuple[Tensor, Tensor]:
        """Returns the sample's input image and target mask.

        Returns:
            Tensor: Input image, shape (channels, height, width).
            Tensor: Target mask, shape (1, height, width).
        """
        # We use "copy" because of warning :
        # The given NumPy array is not writable, and
        # PyTorch does not support non-writable tensors
        x = torch.from_numpy(np.copy(self.otci)).float()
        x = torch.moveaxis(x, 2, 0)
        y = torch.from_numpy(np.copy(self.label_mask)).float()
        y = torch.unsqueeze(y, 0)
        return x, y

    def get_random_cropped_xy(self) -> tuple[Tensor, Tensor, int, int]:
        x, y = self.get_xy()
        cropped_x, cropped_y, left, top = cropper((x, y))
        return cropped_x, cropped_y, left, top

    def get_cropped_xy(
        self, top: int, left: int, load: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Returns a crop of the sample's input image and target mask.

        Args:
            top: Pixel cordinate of the top of the crop.
            left: Pixel cordinate of the left of the crop.
            load_from_cache: Wether to load from a cached file or not.
                Defaults to True.

        Returns:
            Tensor: Crop of the input image, shape (channels, height, width).
            Tensor: Crop of the target mask, shape (1, height, width).
        """
        cropped_file = DATASET_PATH / f"{self.path_algae_mask.stem}_{top}_{left}.npy"
        if cropped_file.exists() and load:
            arr = np.load(cropped_file)
            cropped_x, cropped_y = arr[:3], arr[-1:]
        else:
            x, y = self.get_xy()
            cropped_x = TF.crop(x, top, left, CROP_SIZE[0], CROP_SIZE[1]).float()
            cropped_y = TF.crop(y, top, left, CROP_SIZE[0], CROP_SIZE[1]).float()
        return cropped_x, cropped_y

    def save_crop(self, top: int, left: int) -> None:
        cropped_x, cropped_y = self.get_cropped_xy(top, left, load=False)
        x, y = cropped_x.numpy(), cropped_y.numpy()
        arr = np.concatenate([x, y])
        filename = f"{self.path_algae_mask.stem}_{top}_{left}.npy"
        np.save(DATASET_PATH / filename, arr)

    def get_x_prediction(self) -> Tensor:
        """Increases size of full x input to fit in neural network for prediction."""
        x, _ = self.get_xy()
        # get original shapes for full size images
        _, dim_x, dim_y = x.shape
        # compute shapes mutiple of 64
        new_dim_x = (int(dim_x / 64) + 1) * 64
        new_dim_y = (int(dim_y / 64) + 1) * 64
        # add batch dim to original input
        x = torch.unsqueeze(x, dim=0)
        # created input multiple of 64 containing x
        x_reshaped = torch.zeros(1, 3, new_dim_x, new_dim_y)
        x_reshaped[:, :, :dim_x, :dim_y] += x
        return x_reshaped

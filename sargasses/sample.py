from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from mfai.pytorch.transforms import RandomCropWithMinPositivePixels
from PIL import Image
from torch import Tensor

from sargasses.settings import (
    CROP_SIZE,
    DATASET_PATH,
    MIN_POSITIVE_PERC,
    NETCDF_PATH,
    OTCI_PATH,
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

    def get_cropped_xy(
        self, top: int, left: int, load_from_cache: bool = True
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
        if cropped_file.exists() and load_from_cache:
            arr = np.load(cropped_file)
            cropped_x, cropped_y = arr[:3], arr[-1:]
        else:
            x, y = self.get_xy()
            cropped_x = TF.crop(x, top, left, CROP_SIZE[0], CROP_SIZE[1]).float()
            cropped_y = TF.crop(y, top, left, CROP_SIZE[0], CROP_SIZE[1]).float()
        return cropped_x, cropped_y

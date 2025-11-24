from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as torchvision_functional
from PIL import Image
from torch import Tensor

from sargasses.settings import (
    OTCI_PATH,
    SHAPE_IMGS,
)


class Sample:
    """Sample class for the sargasses dataset.
    Responsibilities:
        - Load and make available an input image.
        - Load and make available an target mask.
    """

    def __init__(
        self, algae_mask_path: Path, otci_folder_path: Path = OTCI_PATH
    ) -> None:
        """
        Args:
            algae_mask_path: Path to an algae mask.
            otci_folder_path: Path to the otci images folder.
                Defaults to the one defined in sargasses.settings.
        """

        # Extract date and anotator from algae mask file name
        self.date_str = algae_mask_path.name[:8]
        self.date = datetime.strptime(self.date_str, "%Y%m%d")
        self.annotator = algae_mask_path.name.split("_")[1][:2]

        # File paths
        self._target_mask_path: Path = algae_mask_path
        self._input_image_path: Path = list(
            otci_folder_path.glob(f"{self.date_str}*.png")
        )[0]

    @property
    def target_mask(self) -> np.ndarray:
        """The sample's target mask.

        Returns:
            np.ndarray: The target mask, shape (height, width).
        """
        return np.load(self._target_mask_path)["arr_0"]

    @property
    def input_image(self) -> np.ndarray:
        """The sample's input image.

        Returns:
            np.ndarray: The input image, shape (height, width, channels)
        """

        return self.load_otci_image(self._input_image_path)

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

    def get_xy(
        self, crop: tuple[int, int, int, int] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Returns a crop of the sample's input image and target mask.

        Args:
            crop: Crop coordinates (top, left, width, height).

        Returns:
            Tensor: Crop of the input image, shape (channels, height, width).
            Tensor: Crop of the target mask, shape (1, height, width).
        """

        input_image = torch.from_numpy(np.copy(self.input_image)).float()
        input_image = torch.moveaxis(input_image, 2, 0)
        target_mask = torch.from_numpy(np.copy(self.target_mask)).float()
        target_mask = torch.unsqueeze(target_mask, 0)

        # Crop
        if crop:
            input_image = torchvision_functional.crop(
                input_image, crop[0], crop[1], crop[2], crop[3]
            )
            target_mask = torchvision_functional.crop(
                target_mask, crop[0], crop[1], crop[2], crop[3]
            )

        return input_image, target_mask

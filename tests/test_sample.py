from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from sargasses.sample import Sample


def test_sample() -> None:
    """Test the class Sample"""
    # Instantiate
    algae_mask_path = Path("tests/data/20220401_w1.npz")
    otci_image_path = Path("tests/data/20220401S3_OTCI.png")
    sample = Sample(algae_mask_path, otci_folder_path=otci_image_path.parent)

    # Test file paths
    assert sample._input_image_path == otci_image_path  # type: ignore[reportPrivateUsage]
    assert sample._target_mask_path == algae_mask_path  # type: ignore[reportPrivateUsage]

    # Test get_xy, input_image and output_image
    input_image: Tensor
    target_mask: Tensor
    input_image, target_mask = sample.get_xy()
    assert isinstance(input_image, Tensor)
    assert isinstance(target_mask, Tensor)
    assert isinstance(sample.input_image, np.ndarray)
    assert isinstance(sample.target_mask, np.ndarray)
    transformed_input_image = torch.from_numpy(sample.input_image).float()
    transformed_input_image = torch.moveaxis(transformed_input_image, 2, 0)
    transformed_target_mask = torch.from_numpy(sample.target_mask).float()
    transformed_target_mask = torch.unsqueeze(transformed_target_mask, 0)
    assert torch.equal(input_image, transformed_input_image)
    assert torch.equal(target_mask, transformed_target_mask)

    # Test static function
    otci_image = Sample.load_otci_image(otci_image_path)
    assert isinstance(otci_image, np.ndarray)
    assert np.array_equal(otci_image, sample.input_image)

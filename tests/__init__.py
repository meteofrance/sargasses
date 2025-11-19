"""Sargasses test module.
Contains unit test utilities such as a datamodule test.
"""

import torch
from lightning.pytorch.core import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def get_test_data(
    num_input_features: int,
    num_output_features: int,
    image_size: tuple[int, int],
    num_samples: int = 10,
):
    """Generate random samples for the training."""
    img_height, img_width = image_size
    x = torch.rand(
        (num_samples, num_input_features, img_height, img_width), dtype=torch.float
    )
    y = torch.randint(2, (num_samples, num_output_features, img_height, img_width))
    return x, y


class SargassesDataModuleTest(LightningDataModule):
    """A Lightning Test DataModule wrapping our sargasses dataset.
    It defines the train/valid/test/predict datasets and their dataloaders.
    """

    def __init__(self, batch_size: int):
        """
        Args:
            batch_size: Number of datapoint processed each step of training.
        """
        super().__init__()
        self.batch_size = batch_size

        x_train, y_train = get_test_data(
            num_input_features=3,
            num_output_features=1,
            image_size=(64, 64),
            num_samples=5,
        )
        self.sargasses_train = TensorDataset(x_train, y_train)

        x_val, y_val = get_test_data(
            num_input_features=3,
            num_output_features=1,
            image_size=(64, 64),
            num_samples=2,
        )
        self.sargasses_val = TensorDataset(x_val, y_val)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        Returns:
            The train dataloader.
        """
        return DataLoader(self.sargasses_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        Returns:
            The validation dataloader.
        """
        return DataLoader(self.sargasses_val, batch_size=self.batch_size)

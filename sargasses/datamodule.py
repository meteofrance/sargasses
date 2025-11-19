from typing import Literal

from lightning.pytorch.core import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from sargasses.dataset import SargassesDataset


class SargassesDataModule(LightningDataModule):
    """LightningDataModule for the Sargasses dataset.
    Instantiates fit/val/test datasets and packages them
    in a pytorch Dataloader class.
    """

    def __init__(
        self,
        batch_size: int,  # > 0
        num_workers: int,  # > 0
        pct_in_train: float,  # ∈ [0, 1]
    ) -> None:
        """Args:
        batch_size: Number of datapoint processed each step of training.
        num_workers: Number of processes used to reading data from disk.
        pct_in_train: Percentage of the data included in the train dataset,
            test and valid dataset share the remaining data.
        """
        if pct_in_train < 0 or pct_in_train > 1:
            raise ValueError(
                "model.pct_in_train should be contained between 0 and 1 included."
            )

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pct_in_train = pct_in_train

    def setup(  # type:ignore[override]
        self,
        stage: Literal["fit", "validate", "test", "predict"],
    ) -> None:
        """Called by lightning at the start of a stage.
        Instantiates the stage's dataset.

        Args:
            stage: Either 'fit', 'validate', 'test' or 'predict'.
                Dictates which dataset will be instantiated.

        Raises:
            NotImplementedError: 'predict' dataset is not implemented.
                Prediction can be made by calling `bin/evaluate/1_predict.py`.
        """
        if stage == "fit":
            self.train_dataset = SargassesDataset(
                split="train", pct_in_train=self.pct_in_train
            )

        if stage in ("fit", "validate"):
            self.val_dataset = SargassesDataset(
                split="val", pct_in_train=self.pct_in_train
            )

        if stage == "test":
            self.test_dataset = SargassesDataset(
                split="test", pct_in_train=self.pct_in_train
            )

        if stage == "predict":
            raise NotImplementedError("Use script `bin/predict.py` for predictions")

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        # for test, batch_size = 1 to log loss and metrics for each sample
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )


if __name__ == "__main__":
    """Datamodule inspection.
    Helps developpers to peer into the inner working of the
    DataModule class and make manual checks.
    """
    splits = ["train", "test", "val"]
    DataModule = SargassesDataModule(
        batch_size=1,
        num_workers=1,
        pct_in_train=0.75,
    )
    loaders = [
        DataModule.train_dataloader(),
        DataModule.test_dataloader(),
        DataModule.val_dataloader(),
    ]
    for i in range(len(splits)):
        print("------")
        print(splits[i])
        print("------")
        print(loaders[i].dataset.df_crops)  # type: ignore[reportAttributeAccessIssue]

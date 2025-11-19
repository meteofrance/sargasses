"""Script used to interact directly with the lightning cli.
It is recommended to use the specialized scripts to interact with your models,
for better checkpoint management.
- Use `bin/fit_and_val.py` to fit and validate from scratch or a checkpoint.
- Use `bin/predict.py` to run inference from a checkpoint.
"""

from lightning.pytorch.core import LightningDataModule

from sargasses.cli import SargassesLightningCLI
from sargasses.datamodule import SargassesDataModule
from sargasses.plmodule import SargassesLightningModule


def cli_main(
    datamodule: type[LightningDataModule] = SargassesDataModule,
    args: list[str] = None,
) -> None:
    SargassesLightningCLI(
        model_class=SargassesLightningModule,
        datamodule_class=datamodule,
        args=args,
    )


if __name__ == "__main__":
    cli_main()

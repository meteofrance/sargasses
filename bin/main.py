"""Script used to interact directly with the lightning cli.
It is recommended to use the specialized scripts to interact with your models,
for better checkpoint management.
- Use `bin/fit_and_val.py` to fit and validate from scratch or a checkpoint.
- Use `bin/predict.py` to run inference from a checkpoint.
"""

from sargasses.cli import SargassesLightningCLI
from sargasses.datamodule import SargassesDataModule
from sargasses.plmodule import SargassesLightningModule

if __name__ == "__main__":
    SargassesLightningCLI(
        model_class=SargassesLightningModule,
        datamodule_class=SargassesDataModule,
    )

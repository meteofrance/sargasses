from typing import Any

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.core import LightningDataModule, LightningModule


class SargassesLightningCLI(LightningCLI):
    """Implement LightningCLI class to customize the cli user interface.
    Responsabilities:
        - Instantiate the lightning module.
        - Instantiate the datamodule.
        - Instantiate the trainer.
        - Launch one of the commands 'fit', 'val', 'test'.
    """

    def __init__(
        self,
        model_class: type[LightningModule],
        datamodule_class: type[LightningDataModule],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_class: The lightning module class to instantiate.
            datamodule_class: The data module class to instantiate.
            *args: Arguments passed to the lightning cli.
            **kwargs: Arguments passed to the lightning cli.
        """
        super().__init__(
            model_class,
            datamodule_class,
            *args,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Allows to add, link and set default arguments
        https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html.
        """
        pass

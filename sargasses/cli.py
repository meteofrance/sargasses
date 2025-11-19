from typing import Any

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.core import LightningDataModule, LightningModule


class SargassesLightningCLI(LightningCLI):
    """Implement LightningCLI class to customize the cli user interface."""

    def __init__(
        self,
        model_class: type[LightningModule],
        datamodule_class: type[LightningDataModule],
        *args: Any,
        **kwargs: Any,
    ) -> None:
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

# Pytorch lightning

pytorch lightning is a framework that aims at organizing your pytorch code.
Lightning requires us to implement 2 classes:


### DataModule [doc](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule) [ref](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule) [source](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/datamodule.html#LightningDataModule)

The DataModule is responsible for the instantiation of your dataset, and for packaging them in a [`pytorch.Dataloader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
The Dataloader is the optimized iterable that will give batches to the training loop.

For this project,the DataModule is implemented here: [SargassesDataModule](../sargasses/datamodule.py)


### LightningModule [doc](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightningmodule) [ref](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule) [source](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule)

A LightningModule organizes your PyTorch code into 6 sections:
- Initialization
- Train Loop
- Validation Loop
- Test Loop
- Prediction Loop
- Optimizers and LR Schedulers

In this project, we use the `SegmentationLightningModule` [doc](https://github.com/meteofrance/mfai/blob/main/README.md#segmentation), [source](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/lightning_modules/segmentation.py). It is already written for segmentation tasks such as sargasses detection. That is why we only needed to implement a plot function called by lightning during the validation loop.

You can read it here: [SargassesLightningModule](../sargasses/plmodule.py)

# Lightning CLI
lightning_cli is a command line interface for your train / test / val and predict needs. LightningCli ensures that your configurations are separate from your source code and that your experiments are reproducible.

You can read about this projects [config files](./config_files.md).

You can customize it by implementing the LightningCli class. See the [doc](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), [ref](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html#lightning.pytorch.cli.LightningCLI), [source](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/cli.html#LightningCLI).

In this project, it was not necessary, you can still read our implementation (that does nothing)[SargassesLightningCli](../sargasses/cli.py)
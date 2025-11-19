# Config files
Making and using multiple config files is an efficient way of tracking different experiments you want to run with your models. Not only do they track model, dataset and trainer hyperparameters, but also technical aspects of your dataset or model implementation (ex: the number of cpu threads used to load your dataset).

Config files are passed to your lightning scripts with the `--config <path/to/config/file.yaml>` option (or `-c`).

Config files are used by `LightningCLI` to instantiate the `LightningModule`, `Datamodule` and `Trainer` classes. This means that the config files contain values for the `__init__()` parameters of those three classes.


## Datamodule
This is made obvious when comparing the `SargassesDataModule.__init__()` parameters and the corresponding attributes in the config.
```py
# sargasses/datamodule.py

class SargassesDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,  # > 0
        num_workers: int,  # > 0
        pct_in_train: float,  # ∈ [0, 1]
    ) -> None:
```
```yaml
# config.yaml

data:
  batch_size: 4
  num_workers: 10
  pct_in_train: 0.75
```

## Model
Here, the arguments for the `LightningModule` are a bit more complicated, let's detail them:
```py
# sargasses/lightning_module.py

class SargassesLightningModule(SegmentationLightningModule):
    def __init__(
        self,
        model: mfai.pytorch.models.base.ModelABC,
        type_segmentation: Literal["binary", "multiclass", "multilabel", "regression"],
        loss: torch.nn.Module,
    ) -> None:
```
The `SargassesLightningModule` requires init parameters for `type_segmentation`, `model` and `loss. The first one is a string attribute. Easy.
```yaml
model:
  type_segmentation: "binary"
```

On the other hand, the `model` parameter is an instance of a class that implement `ModelABC`. This means that we need to give `LightningCLI` the class path and the necessary parameters to instantiate this specific class. Let's choose `mfai.pytorch.models.swineunetr.SwinUNETR`:
```py
class SwinUNETR(ModelABC, MonaiSwinUNETR):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: tuple[int, ...] = (1,),
        settings: SwinUNETRSettings = SwinUNETRSettings(),
        *args: Any,
        **kwargs: Any,
    ) -> None:
```
```yaml
model:
  type_segmentation: "binary"
  model: 
    class_path: mfai.pytorch.models.swinunetr.SwinUNETR
    init_args:
      in_channels: 3
      out_channels: 1
      input_shape:
        - 64
        - 64
```
> ⚠️ the `model` key word is used twice in the config file, the first one is the name the `LightningCLI` gives to the `LightningModule` it will instantiate. The seconde one is the `__init__()` parameter of the `SargassesLightningModule` we defined.

Now, by referencing the mftools [DiceLoss](../../../libs/mftools/torch/losses.py), you should be able to understand the full `LightningModule` config:
```yaml
# config.yaml

model:
  type_segmentation: "binary"
  model: 
    class_path: mfai.pytorch.models.swinunetr.SwinUNETR
    init_args:
      in_channels: 3
      out_channels: 1
      input_shape:
        - 64
        - 64
  loss:
    class_path: mftools.torch.losses.DiceLoss
    init_args:
      mode: "binary"
```


## Trainer
With the previous exemples, you should be able to understand the part of the config dedicated to the trainer. You can reference the related classes here:
- [Trainer](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer).
- [TensorBoardLogger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html#module-lightning.pytorch.loggers.tensorboard).
- [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint).
- [GitLogCallback](../sargasses/git_log.py).
```yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      default_hp_metric: false
      save_dir: /home/labia/dewasmeso/monorepo4ai/projects/sargasses/runs/
      name: test_config
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: "train_loss"
        filename: "ckpt-{epoch:02d}-{val_loss:.2f}"
        save_weights_only: false
        save_top_k: -1
    - class_path: sargasses.git_log.GitLogCallback
  max_epochs: 1
  max_steps: 1
  fast_dev_run: false
```


## All together
A classic config file might look like this:
```yaml
model:
  model: 
    class_path: mfai.pytorch.models.swinunetr.SwinUNETR
    init_args:
      in_channels: 3
      out_channels: 1
      input_shape:
        - 64
        - 64
  type_segmentation: "binary"
  loss:
    class_path: mfai_tools.torch.losses.DiceLoss
    init_args:
      mode: "binary"

data:
  batch_size: 4
  num_workers: 10
  pct_in_train: 0.75

trainer:
  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      default_hp_metric: false
      save_dir: /home/labia/dewasmeso/monorepo4ai/projects/sargasses/runs/
      name: test_config
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: "train_loss"
        filename: "ckpt-{epoch:02d}-{val_loss:.2f}"
        save_weights_only: false
        save_top_k: -1
    - class_path: sargasses.git_log.GitLogCallback
  max_epochs: 1
  max_steps: 1
  fast_dev_run: false
```

> ⚠️ Parameters that are defined in code with a default value do not need to be included in config files. Consequently, this config file is not an exhaustive list of all config parameters possible. (In particular, the `Trainer` class has a lot of parameters to play with).


# Advanced Config File Usage
## Argument linking
Reading the `LightningModule`, `DataModule` and `Trainer` class definitions helps a lot to read / write config files.
If you notice a discrepancy between the class definition and a config file, do not fret ! This could be due to [argument linking](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#argument-linking):
```py
class SargassesLightningCLI(LightningCLI):
    """Implement LightningCLI class to customize the cli user interface."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Allows to add, link and set default arguments
        https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html
        """
        parser.link_arguments("data.batch_size", "model.batch_size")
```

By implementing the `LightningCLI` class, and calling `parser.link_arguments()` in the `add_arguments_to_arser()` function, it is possible to link two arguments. In this exemple, the `batch_size` argument that is required in the `DataModule` and the model, can now be given only to the `DataModule` and it will be automaticly given to the model.

## Config file composition
When calling a `LightningCLI` script, you can compose multiple config files and arguments like so:
```sh
python main.py -c config/file/1.yaml -c config/file/2.yaml --data.pct_in_train 0.5
```

The options given the latest to the command have priority over the others. This means that the parameter `data.pct_in_train` will have the value 0.5 no matter what is written in the 2 config files. In the same fashion, values declared in the config file number two, will take precedence over the values from the config file number one.

Using this methods, you could divide your config files, one for your `Datamodule`, one for the `trainer` and one for the `LightningModule`.

While this soud like a good idea to keep your config file smallers, we discourage it. Explicit config files, that contain all the values used for an experiment helps with reproducibility and tracking.
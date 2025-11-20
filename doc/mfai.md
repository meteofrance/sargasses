# mfai

This project require the lib `mfai`. You can install it with pip or any other 

`mfai` is Meteo-France's Artificial Intelligence Python Package. It is a collection of neural network architectures and training tools developped by Météo France AI-lab.

`mfai`'s models and other tools all conform to the same interface, allowing you to swap one for an other.


## Models
The best models we found for sargasses detection are [UNetRPP](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/models/unetrpp.py) and [HalfUNet](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/models/half_unet.py). Thanks to lightning-cli, we were able to run an experience plan comparing multiple convolution models from `mfai`, by changing a few values in our config files.

See [doc/config_files.md](./config_files.md) to have a full walkthrough of config files with lightning-cli.

| References                                                                                                   |
| ------------------------------------------------------------------------------------------------------------ |
| [config/best_unetrpp.py](../config/best_unetrpp.yaml)                                                        |
| [`mfai` list of models](https://github.com/meteofrance/mfai?tab=readme-ov-file#neural-network-architectures) |
| [lightningcli doc](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)                          |


## Loss

The loss that gave us the best result for sargasses detection is the [DiceLoss]().

| References                                                                                         |
| -------------------------------------------------------------------------------------------------- |
| [`mfai` source](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/losses/toolbelt.py#L14) |

## Segmentation lightning module
The sargasses project uses `mfai`'s implementation of `pytorch-lightning`'s `LightningModule` specialized for segmentation tasks.

| References                                                                                                    |
| ------------------------------------------------------------------------------------------------------------- |
| [sargasses/plmodule.py](../sargasses/plmodule.py)                                                             |
| [`mfai` doc](https://github.com/meteofrance/mfai/blob/main/README.md#segmentation)                            |
| [`mfai` source](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/lightning_modules/segmentation.py) |
| [lightning doc](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)                        |


## Transform
`mfai` offers multiple transforms that perform common data transformations. To adapat our dataset, we used the [RandomCropWithMinPositivePixels](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/transforms.py#L8).

This transform randomly crops an input image to a 512x512 image, with min 15% of positive pixels.


| References                                                                                                   |
| ------------------------------------------------------------------------------------------------------------ |
| [`mfai` source](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/transforms.py#L8)                 |
| [pytorch doc](https://docs.pytorch.org/vision/stable/transforms.html)                                        |
| [pytorch guide](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html) |



## NamedTensors

`pytorch`'s NamedTensors has not been approved yet and is pending validation. In the mean time, we use `mfai`'s own NamedTensors, tensors with named dimensions and feature dimensions.

| References                                                                                 |
| ------------------------------------------------------------------------------------------ |
| [`mfai` doc](https://github.com/meteofrance/mfai/blob/main/doc/namedtensor.md)             |
| [`mfai` source](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/namedtensor.py) |
|                                                                                            |


## Onnx export and inference
`mfai`'s models comme with onnx capabilities. We use them to export and infer our model.

| Reference                                                                               |
| --------------------------------------------------------------------------------------- |
| [`mfai` source](https://github.com/meteofrance/mfai/blob/main/mfai/pytorch/__init__.py) |
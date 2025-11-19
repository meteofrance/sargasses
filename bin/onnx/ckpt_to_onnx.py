"""Exports a bmr model to onnx from a checkpoint.
Usage:

```
python bin/onnx/ckpt_to_onnx.py -i <path to a .ckpt file> -o <output path>
```
"""

from pathlib import Path

import torch
from mfai.pytorch import export_to_onnx

from sargasses.plmodule import SargassesLightningModule
from sargasses.settings import SHAPE_IMGS


def ckpt_to_onnx(ckpt_path: Path | str, output_path: Path | str) -> None:
    """Export the model from a pytorch checkpoint to a onnx model.

    Args:
        ckpt_path: Path to a '.ckpt' checkpoint file.
        output_path: exported onnx file output path.
    """
    model = SargassesLightningModule.load_from_checkpoint(ckpt_path)

    # Fake Tensor (batch, 3, SHAPE_IMGS). 3 channels because inputs are in RGB
    sample = torch.ones((1, 3) + SHAPE_IMGS, dtype=torch.float32)

    export_to_onnx(model, sample, output_path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--ckpt-path", type=Path, required=True, help="Path to a '.ckpt' file."
    )
    parser.add_argument("-o", "--output-path", type=Path, default=Path("model.onnx"))
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    # Validate checkpoint path
    ckpt_path: Path = args.ckpt_path
    if not ckpt_path.exists() or ckpt_path.suffix != ".ckpt":
        raise ValueError("Argument `--ckpt-path` should point to a '.ckpt' file.")

    # Validate output path
    output_path: Path = args.output_path
    if output_path.exists() and not args.override:
        raise FileExistsError(
            "Output file already exist. To override, pass `--override` argument."
        )

    # Export
    ckpt_to_onnx(ckpt_path=ckpt_path, output_path=output_path)
    print(f"onnx model exported to {output_path}")

"""Inference script using a model stored as onnx."""

from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from mfai.pytorch import onnx_load_and_infer

from sargasses.plots import compute_masks_tp_pf_fn, plot_prediction
from sargasses.sample import Sample


def onnx_predict(
    onnx_path: Path,
    image: np.ndarray,
    altitude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Load and run an inference with the ONNX model.

    Args:
        onnx_path: Path to a `.onnx` file.
        image: A rgb otci image, of shape (height, width, channels).
        altitude_mask: The altitude level is used to mask land areas.
            Defaults to None.

    Returns:
        np.ndarray: Prediction, masked if altitude_mask argument is given.
    """

    # Rearange and add batch dimension
    image_input = torch.from_numpy(image).type(torch.float32)
    image_input = rearrange(image_input, "h w c -> 1 c h w")

    # Inference
    y_hat: np.ndarray = onnx_load_and_infer(onnx_path, image_input)[0]

    # Extract
    _, _, dim_x, dim_y = image_input.shape
    y_hat = y_hat[0, 0, :dim_x, :dim_y]

    # Apply altitude mask
    y_hat_masked: np.ndarray = y_hat
    if altitude_mask is not None:
        y_hat_masked = np.where(altitude_mask > 0.0, 0.0, y_hat)

    return y_hat_masked


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--onnx-path", type=Path, help="Path to a '.onnx' file.", required=True
    )
    parser.add_argument(
        "--img-path", type=Path, help="Path to a png octi image.", required=True
    )
    parser.add_argument(
        "-o", "--output_path", type=Path, default=Path("prediction.png")
    )
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    # Validate onnx path
    onnx_path: Path = args.ckpt_path
    if not onnx_path.exists() or onnx_path.suffix != ".ckpt":
        raise FileNotFoundError(
            "Argument `--ckpt-path` should point to a '.ckpt' file."
        )

    # Validate img path
    img_path: Path = args.img_path
    if not img_path.exists():
        raise FileNotFoundError(
            "Argument `--img-path` should point to a png octi image"
        )

    # Validate output path
    output_path: Path = args.output_path
    if output_path.exists() and not args.override:
        raise FileExistsError(
            "Output file already exist. To override, pass `--override` argument."
        )

    # Load image
    image: np.ndarray = Sample.load_otci_image(args.img_path)

    # Prediction
    prediction: np.ndarray = onnx_predict(onnx_path, image)

    # Write image
    true_positive, false_positive, false_negative = compute_masks_tp_pf_fn(
        image, prediction
    )
    plot_prediction(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        x_otci=image,
        title=f"Prediction on image {output_path}",
        save_path=output_path,
    )

from pathlib import Path

import numpy as np

from bin.fit_and_val import fit_and_val
from bin.onnx.ckpt_to_onnx import ckpt_to_onnx
from bin.onnx.predict_onnx import onnx_predict
from bin.predict import ckpt_predict
from sargasses.sample import Sample
from tests import SargassesDataModuleTest


def test_full_pipeline() -> None:
    """Test the full project life cycle.
    Test the cli interface entry points.

    - Training
    - Checkpoint writting
    - Predict from checkpoint
    - Export to onnx
    - Predict from onnx
    """

    # Paths
    log_folder = Path("/tmp/sargasses_unittests")
    ckpt_path = log_folder / "version_0/checkpoints/checkpoint.ckpt"
    image_path = Path("tests/data/20220401S3_OTCI.png")
    onnx_path = log_folder / "model.onnx"

    # Train with a test config.
    # This config writes a checkpoint file at:
    #     unittest_logs/version_0/checkpoints/checkpoint.ckpt
    fit_and_val(
        datamodule_cls=SargassesDataModuleTest,
        args=["--config", "tests/data/test_config.yaml"],
    )
    assert ckpt_path.exists(), "Checkpoint not saved while training."

    # Predict from checkpoint
    image = Sample.load_otci_image(image_path)
    _: np.ndarray = ckpt_predict(
        ckpt_path=ckpt_path,
        image=image,
    )


if __name__ == "__main__":
    test_full_pipeline()

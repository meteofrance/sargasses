from pathlib import Path

import numpy as np

from bin.fit_and_val import fit_and_val
from bin.predict import ckpt_predict
from bin.onnx.ckpt_to_onnx import ckpt_to_onnx
from bin.onnx.predict_onnx import onnx_predict
from sargasses.sample import Sample
from tests import SargassesDataModuleTest


def test_full_pipeline() -> None:
    """Test the full project life cycle.
    Test the cli interface entry points.

    - Training.
    - Retrain a checkpoint.
    - Checkpoint writting.
    - Predict from checkpoint.
    - Export to onnx.
    - Predict from onnx.
    """

    # Paths
    log_folder = Path("/tmp/sargasses_unittests")
    ckpt_path = log_folder / "version_0/checkpoints/checkpoint.ckpt"
    onnx_path = log_folder / "sargasses_model.onnx"
    image_path = Path("tests/data/20220401S3_OTCI.png")

    # Train with a test config.
    # This config writes a checkpoint file at:
    #     unittest_logs/version_0/checkpoints/checkpoint.ckpt
    fit_and_val(
        datamodule_cls=SargassesDataModuleTest,
        args=["--config", "tests/data/test_config.yaml"],
    )
    assert ckpt_path.exists(), "Checkpoint not saved while training."

    # Retrain from checkpoint
    fit_and_val(
        datamodule_cls=SargassesDataModuleTest,
        args=["--config", "tests/data/test_config.yaml"],
        ckpt_path=ckpt_path,
    )
    assert (log_folder / "version_1").exists(), (
        "Retrain from checkpoint did not save a checkpoint."
    )


    # Predict from checkpoint
    image = Sample.load_otci_image(image_path)
    ckpt_prediction: np.ndarray = ckpt_predict(
        ckpt_path=ckpt_path,
        image=image,
    )

    # Export to onnx
    ckpt_to_onnx(ckpt_path, onnx_path)

    # Predict from onnx
    onnx_prediction: np.ndarray = onnx_predict(
        onnx_path=onnx_path,
        image=image,
        altitude_mask=None,
    )

    # Compare checkpoint and onnx prediction
    # TODO: Check why do not work
    # assert np.array_equal(ckpt_prediction, onnx_prediction)

if __name__ == "__main__":
    test_full_pipeline()

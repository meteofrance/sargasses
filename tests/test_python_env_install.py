import subprocess
from pathlib import Path


def test_python_env_install() -> None:
    """Compare the curent python environment's list and versions of packages.
    Uses the command `pip list --format=freeze` to generate the list of
    installed packages along their versions.
    """
    # Get current pip freeze
    pip_freeze: dict[str, str]
    try:
        result = subprocess.run(
            ["pip", "list", "--format=freeze"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        pip_freeze = {
            line.split("==")[0]: line.split("==")[1]
            for line in result.stdout.strip().split("\n")
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running pip command: {e}")
        pip_freeze = {}

    expected_pip_freeze_path = (
        Path(__file__).resolve().parent / "data" / "pip_freeze.txt"
    )

    # Open the expected pip_freeze
    with expected_pip_freeze_path.open("r") as file:
        expected_pip_freeze: dict[str, str] = {
            line.split("==")[0]: line.split("==")[1].rstrip("\n").strip()
            for line in file.readlines()
        }

    # Compare
    if pip_freeze == expected_pip_freeze:
        return

    # Display changed libs
    print(
        "Dependencies that changed:\n\t"
        + "\n\t".join(
            [
                f"{k} {expected_pip_freeze[k]} -> {pip_freeze[k]}"
                for k in pip_freeze.keys()
                if k in expected_pip_freeze
                if expected_pip_freeze[k] != pip_freeze[k]
            ]
        )
    )
    print(
        "Dependencies deleted:\n\t"
        + "\n\t".join(
            [
                f"{k} {expected_pip_freeze[k]}"
                for k in expected_pip_freeze.keys()
                if k not in pip_freeze
            ]
        )
    )
    print(
        "Dependencies added:\n\t"
        + "\n\t".join(
            [
                f"{k} {pip_freeze[k]}"
                for k in pip_freeze.keys()
                if k not in expected_pip_freeze
            ]
        )
    )

    assert pip_freeze == expected_pip_freeze


if __name__ == "__main__":
    test_python_env_install()

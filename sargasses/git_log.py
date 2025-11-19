import re
import subprocess
import warnings
from pathlib import Path

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_only


def ckpt_git_tag_is_compatible(ckpt_path: Path | str) -> bool:
    # Check if checkpoint contains a git log
    ckpt_git_log_path = Path(ckpt_path).parent.parent / "git_log.txt"
    if not ckpt_git_log_path.exists():
        warnings.warn(
            "\n\033[31;1;4mGitLogCallback:\033[0m"
            "No git log found for this checkpoint.\n"
        )
        return True

    # Read the checkpoint's latest tag
    with open(ckpt_git_log_path, "r", encoding="utf-8") as file:
        txt = file.read()
        re_match = re.search(
            pattern=r"^git describe --tags\n(.*)$", string=txt, flags=re.MULTILINE
        )

    if not re_match:
        warnings.warn(
            "\n\033[31;1;4mGitLogCallback:\033[0m Tag not found in checkpoint git log\n"
        )
        return True
    ckpt_latest_tag = re_match.group(1)

    # Read current git tag
    current_git_tag = (
        subprocess.check_output(["git", "describe", "--tags"]).strip().decode("utf-8")
    )

    # Compare git tags
    if ckpt_latest_tag != current_git_tag:
        warnings.warn(
            "\n\033[31;1;4mGitLogCallback:\033[0m Using a checkpoint created "
            f"on git tag{ckpt_latest_tag} while curently on tag "
            f"{current_git_tag}\nYou can checkout the git tag used to train "
            f"this checkpoint with `git checkout tags/{ckpt_latest_tag}`"
        )
        return False

    return True


class GitLogCallback(Callback):
    """Callback that saves the `git log` and `git status` informations when
    starting training.
    """

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        self.git_log_path: Path | None = None
        if trainer.logger is None or trainer.logger.log_dir is None:
            return

        git_log_path = Path(trainer.logger.log_dir) / "git_log.txt"

        # Run git log commands
        self.git_log: dict[str, str] = self._generate_git_log()

        # If able, write a git log
        if not trainer.fast_dev_run and trainer.logger:  # type: ignore[reportAttributeAccessIssue]
            self._write_git_log(git_log_path)

        # If a checkpoint path is given, compare it's git log to the one just written
        if trainer.ckpt_path:
            ckpt_git_tag_is_compatible(trainer.ckpt_path)

    def _write_git_log(self, save_path: Path) -> None:
        # Write command outputs
        with open(save_path, "w") as f:
            for key, value in self.git_log.items():
                f.write(f"\n\n{key}\n")
                f.write(value)

    def _generate_git_log(self) -> dict[str, str]:
        git_commands = [  # commands whose output we want to log
            ["git", "describe", "--tags"],
            ["git", "log", "-n", "1"],
            ["git", "status"],
            ["git", "diff"],
        ]

        # Run commands and store their outputs
        git_log = {}
        for command in git_commands:
            git_log[" ".join(command)] = (
                subprocess.check_output(command).strip().decode("utf-8")
            )
        return git_log

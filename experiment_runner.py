"""Shared command construction and execution helpers for training runs."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


TRAINING_SCRIPT_NAME = "view_classification_with_noise.py"


@dataclass(frozen=True)
class RunConfig:
    """CLI argument bundle for a single training or evaluation run."""

    noise_percentage: int
    epoch_weights: Optional[int] = None
    temperature: float = 0
    drop_k: int = 0
    backbone: int = 0
    loss: int = 7

    def to_cli_args(self) -> List[str]:
        args = [
            "--noise_percentage",
            str(self.noise_percentage),
            "--temperature",
            str(self.temperature),
            "--drop_k",
            str(self.drop_k),
            "--backbone",
            str(self.backbone),
            "--loss",
            str(self.loss),
        ]
        if self.epoch_weights is not None:
            args.extend(["--use_epoch_weights", str(self.epoch_weights)])
        return args


def mode_flags(use_corrections: bool, remove_as_noisy_labels: bool) -> List[str]:
    """Return mutually exclusive mode flags for correction/removal settings."""

    if use_corrections and remove_as_noisy_labels:
        raise ValueError("use_corrections and remove_as_noisy_labels cannot both be True.")
    if use_corrections:
        return ["--use_corrections"]
    if remove_as_noisy_labels:
        return ["--remove_as_noisy_labels"]
    return []


def resolve_training_script(script_name: str = TRAINING_SCRIPT_NAME) -> Path:
    """Resolve the training script path from the current working directory."""

    return Path.cwd() / script_name


def build_command(flags: Sequence[str], config: RunConfig, script_name: str = TRAINING_SCRIPT_NAME) -> List[str]:
    """Build the Python command used to execute one run."""

    script_path = resolve_training_script(script_name)
    return [sys.executable, str(script_path), *flags, *config.to_cli_args()]


def run_training(flags: Sequence[str], config: RunConfig, script_name: str = TRAINING_SCRIPT_NAME) -> int:
    """Execute one run command and return its exit code."""

    command = build_command(flags=flags, config=config, script_name=script_name)
    print("Running:", subprocess.list2cmdline(command))
    return subprocess.run(command, check=False).returncode

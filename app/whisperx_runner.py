"""Execution helpers for running WhisperX via subprocess."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional

from .config import WhisperXConfig

LOGGER = logging.getLogger(__name__)


class WhisperXRuntimeError(RuntimeError):
    """Raised when WhisperX exits with a non-zero status."""


def _detect_device_and_precision() -> Dict[str, str]:
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover - optional
        return {"device": "cpu", "compute_type": "int8"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return {"device": device, "compute_type": compute_type}


def build_command(
    config: WhisperXConfig,
    audio_path: Path,
    output_dir: Path,
    *,
    additional_args: Optional[Iterable[str]] = None,
) -> str:
    """Compose the command that launches WhisperX."""

    detected = _detect_device_and_precision()
    device = detected["device"]
    compute_type = config.compute_type or detected["compute_type"]

    base_args = [
        "whisperx",
        str(audio_path),
        "--device",
        device,
        "--model",
        config.model,
        "--language",
        config.language,
        "--batch_size",
        str(config.batch_size),
        "--compute_type",
        compute_type,
        "--output_dir",
        str(output_dir),
        "--output_format",
        "all",
    ]

    if config.hf_token or os.environ.get("HF_TOKEN"):
        base_args.extend(["--hf_token", config.hf_token or os.environ.get("HF_TOKEN", "")])

    if config.diarize:
        base_args.append("--diarize")

    if config.enable_alignment:
        base_args.extend(["--align_model", config.align_model])

    if additional_args:
        base_args.extend(additional_args)

    return " ".join(shlex.quote(arg) for arg in base_args)


def run_whisperx(
    config: WhisperXConfig,
    audio_path: Path,
    output_dir: Path,
    *,
    env: Optional[Dict[str, str]] = None,
    additional_args: Optional[Iterable[str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Execute WhisperX and return the completed process instance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(config, audio_path, output_dir, additional_args=additional_args)
    LOGGER.info("Running WhisperX: %s", command)

    process_env = dict(os.environ)
    if config.hf_token:
        process_env["HF_TOKEN"] = config.hf_token
    if env:
        process_env.update(env)

    completed = subprocess.run(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        LOGGER.error("WhisperX failed with code %s", completed.returncode)
        LOGGER.debug("STDOUT:\n%s", completed.stdout)
        LOGGER.debug("STDERR:\n%s", completed.stderr)
        raise WhisperXRuntimeError(
            f"WhisperX exited with code {completed.returncode}. "
            "Inspect stdout/stderr for more details."
        )

    LOGGER.debug("WhisperX stdout:\n%s", completed.stdout)
    LOGGER.debug("WhisperX stderr:\n%s", completed.stderr)

    return completed

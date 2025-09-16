"""Utilities for downloading audio files from various sources."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when audio download fails."""


def _download_with_requests(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code != requests.codes.ok:
        raise DownloadError(f"Failed to download {url!r}: HTTP {response.status_code}")

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)


def _download_google_drive(url: str, destination: Path) -> None:
    try:
        import gdown  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DownloadError("gdown is required to download from Google Drive") from exc

    # gdown handles both share links and direct download identifiers.
    output = gdown.download(url, str(destination), quiet=False, fuzzy=True)
    if not output:
        raise DownloadError(f"gdown was unable to download {url!r}")


def download_audio(url: str, destination: Path, *, overwrite: bool = True) -> Path:
    """Download an audio file from ``url`` into ``destination``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and overwrite:
        destination.unlink()

    if "drive.google.com" in url or "docs.google.com" in url:
        LOGGER.info("Downloading %s via gdown", url)
        _download_google_drive(url, destination)
    else:
        LOGGER.info("Downloading %s via HTTP", url)
        _download_with_requests(url, destination)

    if not destination.exists():
        raise DownloadError(f"Download succeeded but {destination} is missing")

    return destination


def save_telegram_file(file_url: str, destination: Path, *, session: Optional[requests.Session] = None) -> Path:
    """Download a Telegram file using a direct ``file_url``."""

    session = session or requests.Session()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(file_url, stream=True, timeout=60) as response:
        if response.status_code != requests.codes.ok:
            raise DownloadError(
                f"Unable to fetch Telegram file {file_url!r}: HTTP {response.status_code}"
            )
        with destination.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    return destination

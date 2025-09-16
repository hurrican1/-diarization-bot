"""Utility for storing reference embeddings for known speakers."""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class EnrollmentError(RuntimeError):
    """Raised when enrollment of speaker profiles fails."""


def _load_embedding_model():
    try:
        import torch
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    except Exception as exc:  # pragma: no cover - optional heavy dependency
        raise EnrollmentError("pyannote.audio and its dependencies must be installed") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)


def _cut_wav(audio: Path, start: float, end: float, destination: Path) -> None:
    duration = max(0.0, end - start)
    if duration <= 0:
        raise EnrollmentError("Segment duration must be positive")

    command = (
        f"ffmpeg -y -nostdin -loglevel error -ss {start:.3f} -t {duration:.3f} "
        f"-i {shlex.quote(str(audio))} -ar 16000 -ac 1 -f wav {shlex.quote(str(destination))}"
    )
    result = subprocess.run(shlex.split(command), check=False)
    if result.returncode != 0:
        raise EnrollmentError(f"ffmpeg failed with code {result.returncode}")


def enroll_speakers(
    audio_path: Path,
    segments: Iterable[Dict[str, float]],
    speaker_map: Dict[str, str],
    output_dir: Path,
    *,
    max_segments_per_speaker: int = 5,
) -> List[Path]:
    """Create averaged embeddings per speaker and save them as ``.npy`` files."""

    embedder = _load_embedding_model()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if end - start < 2.0:
            continue
        grouped.setdefault(speaker, []).append((start, end))

    created: List[Path] = []
    with tempfile.TemporaryDirectory(prefix="enroll_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for speaker_code, windows in grouped.items():
            windows.sort(key=lambda item: item[1] - item[0], reverse=True)
            speaker_name = speaker_map.get(speaker_code, speaker_code)
            embeddings: List[np.ndarray] = []
            for idx, (start, end) in enumerate(windows[:max_segments_per_speaker]):
                wav_path = temp_dir_path / f"{speaker_code}_{idx}.wav"
                _cut_wav(audio_path, start, end, wav_path)
                vector = embedder(str(wav_path)).squeeze(0).cpu().numpy()
                embeddings.append(vector)
            if embeddings:
                mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0).astype("float32")
                output_path = output_dir / f"{speaker_name}.npy"
                np.save(output_path, mean_embedding)
                created.append(output_path)

    return created

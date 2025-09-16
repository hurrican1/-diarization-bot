"""High level orchestration of the diarization workflow."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .downloader import DownloadError, download_audio
from .postprocess import apply_speaker_map, load_segments, write_segments
from .speaker_enrollment import EnrollmentError, enroll_speakers
from .whisperx_runner import WhisperXRuntimeError, run_whisperx

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    audio_path: Path
    output_dir: Path
    diarized_text: Path
    raw_segments_path: Path


class AudioDiarizationPipeline:
    """Encapsulates the steps required to diarize an audio recording."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def process(self, source: str, *, run_timestamp: Optional[str] = None) -> PipelineResult:
        """Download an audio file (or copy a local file) and run the pipeline."""

        run_timestamp = run_timestamp or time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_root / run_timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(source)
        audio_path = run_dir / (source_path.name if source_path.exists() else "input_audio.mp3")

        if source_path.exists():
            LOGGER.info("Copying local audio %s", source_path)
            shutil.copy2(source_path, audio_path)
        else:
            LOGGER.info("Downloading audio from %s", source)
            try:
                download_audio(source, audio_path)
            except DownloadError as exc:
                raise RuntimeError(f"Failed to download {source!r}") from exc

        whisperx_output = run_dir / "whisperx"
        try:
            run_whisperx(self.config.whisperx, audio_path, whisperx_output)
        except WhisperXRuntimeError as exc:
            raise RuntimeError("WhisperX execution failed") from exc

        json_path = whisperx_output / f"{audio_path.stem}.json"
        if not json_path.exists():
            raise RuntimeError(f"Expected WhisperX JSON at {json_path} is missing")

        segments = load_segments(json_path)
        mapped_segments = apply_speaker_map(segments, self.config.speaker_map)
        diarized_text_path = run_dir / "meeting_diarized.txt"
        write_segments(mapped_segments, diarized_text_path)

        if self.config.speaker_map:
            try:
                enroll_speakers(
                    audio_path,
                    [
                        {
                            "speaker": segment.speaker,
                            "start": segment.start,
                            "end": segment.end,
                        }
                        for segment in segments
                    ],
                    self.config.speaker_map,
                    self.config.speaker_db,
                )
            except EnrollmentError as exc:
                LOGGER.warning("Speaker enrollment failed: %s", exc)

        return PipelineResult(
            audio_path=audio_path,
            output_dir=run_dir,
            diarized_text=diarized_text_path,
            raw_segments_path=json_path,
        )

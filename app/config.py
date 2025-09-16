"""Configuration models used across the diarization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(slots=True)
class WhisperXConfig:
    """Options that control how WhisperX is executed."""

    model: str = "medium"
    language: str = "ru"
    batch_size: int = 8
    compute_type: Optional[str] = None
    diarize: bool = True
    align_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    enable_alignment: bool = True
    hf_token: Optional[str] = None


@dataclass(slots=True)
class PipelineConfig:
    """High-level configuration for the diarization pipeline."""

    whisperx: WhisperXConfig = field(default_factory=WhisperXConfig)
    output_root: Path = Path("outputs")
    speaker_map: Dict[str, str] = field(default_factory=dict)
    speaker_db: Path = Path("speakers_db")

    def with_overrides(
        self,
        *,
        output_root: Optional[Path] = None,
        speaker_map: Optional[Dict[str, str]] = None,
        speaker_db: Optional[Path] = None,
    ) -> "PipelineConfig":
        """Return a new instance with select fields overridden."""

        return PipelineConfig(
            whisperx=self.whisperx,
            output_root=output_root or self.output_root,
            speaker_map=speaker_map or dict(self.speaker_map),
            speaker_db=speaker_db or self.speaker_db,
        )


def load_speaker_map(path: Path) -> Dict[str, str]:
    """Load a speaker mapping JSON file if it exists."""

    import json

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        data: Dict[str, str] = json.load(fh)
    return data

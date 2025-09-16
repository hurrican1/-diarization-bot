"""Utilities for transforming WhisperX output into human-readable formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(slots=True)
class Segment:
    """A diarized speech segment."""

    speaker: str
    text: str
    start: float
    end: float

    @property
    def pretty_time(self) -> str:
        return f"[{_format_time(self.start)}–{_format_time(self.end)}]"


def _format_time(timestamp: float) -> str:
    seconds = int(timestamp)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_segments(json_path: Path) -> List[Segment]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = []
    for raw in data.get("segments", []):
        speaker = raw.get("speaker", "SPEAKER")
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            Segment(
                speaker=speaker,
                text=text,
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
            )
        )
    return segments


def apply_speaker_map(segments: Iterable[Segment], mapping: Dict[str, str]) -> List[Segment]:
    mapped = []
    for segment in segments:
        speaker = mapping.get(segment.speaker, segment.speaker)
        mapped.append(
            Segment(
                speaker=speaker,
                text=segment.text,
                start=segment.start,
                end=segment.end,
            )
        )
    return mapped


def segments_to_text(segments: Sequence[Segment]) -> str:
    lines = [f"{segment.pretty_time} {segment.speaker}: {segment.text}" for segment in segments]
    return "\n".join(lines)


def write_segments(segments: Sequence[Segment], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(segments_to_text(segments), encoding="utf-8")
    return destination

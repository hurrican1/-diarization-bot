"""High-level package for audio diarization pipeline with Telegram bot support."""

from .config import PipelineConfig, WhisperXConfig
from .pipeline import AudioDiarizationPipeline, PipelineResult

__all__ = [
    "PipelineConfig",
    "WhisperXConfig",
    "AudioDiarizationPipeline",
    "PipelineResult",
]

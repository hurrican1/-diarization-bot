"""Command line wrapper for the diarization pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import PipelineConfig, WhisperXConfig, load_speaker_map
from app.pipeline import AudioDiarizationPipeline

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WhisperX diarization pipeline")
    parser.add_argument("source", help="URL или локальный путь до аудио")
    parser.add_argument(
        "--model",
        default="medium",
        help="WhisperX модель (например, medium, large-v2, large-v3)",
    )
    parser.add_argument("--language", default="ru", help="Язык речи")
    parser.add_argument("--no-diarize", action="store_true", help="Отключить диаризацию")
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Отключить выравнивание через wav2vec2",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        default=None,
        help="Hugging Face токен (можно также задать через переменную окружения HF_TOKEN)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Базовая папка для результатов",
    )
    parser.add_argument(
        "--speaker-map",
        default="config/speaker_map.json",
        help="JSON с отображением SPEAKER_XX -> Имя",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speaker_map = load_speaker_map(Path(args.speaker_map))

    pipeline = AudioDiarizationPipeline(
        PipelineConfig(
            whisperx=WhisperXConfig(
                model=args.model,
                language=args.language,
                diarize=not args.no_diarize,
                enable_alignment=not args.no_align,
                hf_token=args.hf_token,
            ),
            output_root=Path(args.output_dir),
            speaker_map=speaker_map,
        )
    )

    result = pipeline.process(args.source)
    logging.info("Диаризация завершена. Итоговый файл: %s", result.diarized_text)


if __name__ == "__main__":
    main()

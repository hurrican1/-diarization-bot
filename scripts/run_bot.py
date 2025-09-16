"""Entry point to run the Telegram diarization bot."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import PipelineConfig, WhisperXConfig, load_speaker_map
from app.telegram_bot import build_bot_from_env

logging.basicConfig(level=logging.INFO)


def main() -> None:
    speaker_map_path = Path("config/speaker_map.json")
    speaker_map = load_speaker_map(speaker_map_path)

    config = PipelineConfig(
        whisperx=WhisperXConfig(
            hf_token=None,
        ),
        speaker_map=speaker_map,
    )

    bot = build_bot_from_env(config)
    bot.run_polling()


if __name__ == "__main__":
    main()

"""Telegram bot integration for the diarization pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import PipelineConfig
from .downloader import save_telegram_file
from .pipeline import AudioDiarizationPipeline

LOGGER = logging.getLogger(__name__)


class DiarizationBot:
    """Wraps python-telegram-bot application that invokes the pipeline."""

    def __init__(self, telegram_token: str, pipeline: AudioDiarizationPipeline):
        self.pipeline = pipeline
        self.application: Application = (
            ApplicationBuilder()
            .token(telegram_token)
            .rate_limiter(AIORateLimiter())
            .post_init(self._post_init)
            .build()
        )
        self._register_handlers()

    async def _post_init(self, application: Application) -> None:  # pragma: no cover - callback signature
        LOGGER.info("Bot initialized as @%s", application.bot.username)

    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler("start", self._start))
        self.application.add_handler(CommandHandler("help", self._help))
        self.application.add_handler(CommandHandler("process", self._process_url))
        self.application.add_handler(
            MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, self._process_audio)
        )

    def run_polling(self) -> None:
        LOGGER.info("Starting bot polling loop")
        self.application.run_polling(stop_signals=None)

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Привет! Отправь /process <ссылка> или аудиофайл, чтобы запустить диаризацию."
        )

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Доступные команды:\n"
            "  /process <url> — скачать аудио по ссылке и запустить распознавание.\n"
            "  Отправь аудио/voice-файл напрямую — бот обработает его автоматически."
        )

    async def _process_url(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.args:
            await update.message.reply_text("Укажи ссылку: /process https://...")
            return
        url = context.args[0]
        await self._run_pipeline(update, url=url)

    async def _process_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.message is not None  # telegram guarantees presence for this handler
        file = None
        if update.message.audio:
            file = await update.message.audio.get_file()
        elif update.message.voice:
            file = await update.message.voice.get_file()
        elif update.message.document:
            file = await update.message.document.get_file()
        if not file:
            await update.message.reply_text("Не удалось получить файл")
            return

        with tempfile.TemporaryDirectory(prefix="tg_audio_") as tmp:
            suffix = Path(file.file_path).suffix or ".ogg"
            local_path = Path(tmp) / f"input{suffix}"
            save_telegram_file(file.file_path, local_path)
            await self._run_pipeline(update, local_file=local_path)

    async def _run_pipeline(
        self,
        update: Update,
        *,
        url: Optional[str] = None,
        local_file: Optional[Path] = None,
    ) -> None:
        assert update.message is not None
        await update.message.chat.send_action(ChatAction.TYPING)

        def job() -> Path:
            if local_file is not None:
                # Pretend there is a URL by copying the file into the pipeline directory.
                output_root = self.pipeline.config.output_root
                output_root.mkdir(parents=True, exist_ok=True)
                target = output_root / "telegram_upload.ogg"
                target.write_bytes(local_file.read_bytes())
                result = self.pipeline.process(str(target))
                try:
                    target.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
            elif url is not None:
                result = self.pipeline.process(url)
            else:  # pragma: no cover - guard
                raise ValueError("Either url or local_file must be provided")
            return result.diarized_text

        loop = asyncio.get_running_loop()
        try:
            diarized_path = await loop.run_in_executor(None, job)
        except Exception as exc:
            LOGGER.exception("Pipeline failed: %s", exc)
            await update.message.reply_text(f"Ошибка обработки: {exc}")
            return

        await update.message.reply_document(
            document=diarized_path.read_bytes(),
            filename=diarized_path.name,
            caption="Готово! Диаризация завершена.",
        )


def build_bot_from_env(config: PipelineConfig) -> DiarizationBot:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    pipeline = AudioDiarizationPipeline(config)
    return DiarizationBot(token, pipeline)

# test
# WhisperX diarization toolkit

Этот репозиторий содержит переработанный вариант изначального скрипта для распознавания речи и диаризации. Код разделён на модули, его можно запускать как из командной строки, так и через Telegram‑бота.

## Структура проекта

```
app/
  config.py               – модели конфигурации
  downloader.py           – загрузка аудио (HTTP, Google Drive, Telegram)
  pipeline.py             – высокоуровневый конвейер WhisperX
  postprocess.py          – преобразование JSON в читаемый текст
  speaker_enrollment.py   – генерация эмбеддингов известных спикеров
  telegram_bot.py         – интеграция с Telegram
  whisperx_runner.py      – запуск WhisperX через subprocess
config/
  speaker_map.json        – отображение SPEAKER_XX -> ФИО
scripts/
  run_pipeline.py         – CLI-обёртка поверх конвейера
  run_bot.py              – запуск Telegram-бота
```

## Подготовка окружения

1. В Colab или локально установите зависимости (пример для CUDA 12.1):
   ```bash
   pip uninstall -y tensorflow tensorflow-gpu tensorflow-intel tensorflow-estimator keras keras-preprocessing || true
   pip uninstall -y pyannote.audio pyannote.core torch torchaudio torchvision transformers || true

   pip install --no-cache-dir \
     torch==2.3.1+cu121 torchaudio==2.3.1+cu121 \
     --index-url https://download.pytorch.org/whl/cu121

   pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git
   pip install -U "transformers==4.41.2" "accelerate==0.33.0" sentencepiece "huggingface_hub>=0.23.0"
   pip install "pyannote.audio==2.1.1" "pyannote.core<5"
   pip install gdown requests python-telegram-bot==20.6 speechbrain numpy
   sudo apt-get install -y ffmpeg
   ```

2. В переменные окружения добавьте токены:
   ```bash
   export HF_TOKEN="<ваш_huggingface_token>"
   export TELEGRAM_BOT_TOKEN="<токен_бота>"
   ```

3. При необходимости скорректируйте файл `config/speaker_map.json` под свои метки.

## Запуск из командной строки

```bash
python scripts/run_pipeline.py "https://drive.google.com/uc?export=download&id=<ID>"
```

Основные параметры:

* `--model` – размер модели WhisperX (`medium`, `large-v2`, `large-v3` и т.д.).
* `--language` – язык речи.
* `--no-diarize` – отключить диаризацию.
* `--no-align` – отключить выравнивание wav2vec2.
* `--hf-token` – токен Hugging Face (если не задан через переменную окружения).
* `--output-dir` – корневая папка для результатов.
* `--speaker-map` – путь к JSON с соответствием SPEAKER_XX → имя.

После завершения работы итоговый файл `meeting_diarized.txt` будет лежать в папке `outputs/<дата_и_время>/`.

## Telegram-бот

1. Убедитесь, что переменная окружения `TELEGRAM_BOT_TOKEN` задана.
2. Запустите:
   ```bash
   python scripts/run_bot.py
   ```
3. В Telegram отправьте боту ссылку (`/process https://...`) или аудиофайл/voice-сообщение. Бот скачает запись, выполнит распознавание и пришлёт готовый файл с результатами.

## Расширение функционала

* Модуль `speaker_enrollment.py` сохраняет усреднённые голосовые эмбеддинги в папку `speakers_db`. Их можно использовать для дальнейшей идентификации.
* Для отладки можно включить подробные логи: `logging.basicConfig(level=logging.DEBUG)`.

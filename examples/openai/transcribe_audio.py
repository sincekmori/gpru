import os
from pathlib import Path

from gpru.openai.api import AudioModel, OpenAiApi, TranscriptionRequest

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key, timeout=60)

req = TranscriptionRequest(file=Path("/path/to/audio.wav"), model=AudioModel.WHISPER_1)
transcription = api.transcribe_audio(req)
print(transcription.json(indent=2, ensure_ascii=False))  # type: ignore[union-attr]
# Example output:
# {
#   "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."
# }

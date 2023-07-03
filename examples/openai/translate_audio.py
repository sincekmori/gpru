import os
from pathlib import Path

from gpru.openai.api import AudioModel, OpenAiApi, TranslationRequest

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key, timeout=60)

req = TranslationRequest(file=Path("/path/to/audio.wav"), model=AudioModel.WHISPER_1)
translation = api.translate_audio(req)
print(translation.json(indent=2, ensure_ascii=False))  # type: ignore[union-attr]
# Example output:
# {
#   "text": "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?"
# }

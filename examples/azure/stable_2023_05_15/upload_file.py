import os
from pathlib import Path

from gpru.azure.stable_2023_05_15 import AzureOpenAiApi, Purpose

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

file = api.upload_file(file=Path("/path/to/puppy.jsonl"), purpose=Purpose.FINE_TUNE)
print(file.json(indent=2))
# Example output:
# {
#   "object": "file",
#   "status": "notRunning",
#   "created_at": 1646126127,
#   "updated_at": 1646127311,
#   "id": "file-181a1cbdcdcf4677ada87f63a0928099",
#   "bytes": null,
#   "purpose": "fine-tune",
#   "filename": "puppy.jsonl",
#   "statistics": null,
#   "error": null
# }

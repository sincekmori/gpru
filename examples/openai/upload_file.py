import os
from pathlib import Path

from gpru.openai.api import OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key, timeout=60)

file = api.upload_file(file=Path("/path/to/mydata.jsonl"), purpose="fine-tune")
print(file.json(indent=2))
# Example output:
# {
#   "id": "file-XjGxS3KTG0uNmNOK362iJua3",
#   "object": "file",
#   "bytes": 140,
#   "created_at": 1613779121,
#   "filename": "mydata.jsonl",
#   "purpose": "fine-tune",
#   "status": null,
#   "status_details": null
# }

import os
from pathlib import Path

from gpru.openai.api import ImageEditing, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key, timeout=60)

image_editing = ImageEditing(
    image=Path("/path/to/sunlit_lounge.png"),
    mask=Path("/path/to/mask.png"),
    prompt="A sunlit indoor lounge area with a pool containing a flamingo",
    n=2,
)
image_list = api.edit_image(image_editing)
print(image_list.json(indent=2))
# Example output:
# {
#   "created": 1683647288,
#   "data": [
#     {
#       "url": "https://oaidalleapiprodscus.blob.core.windows.net/***",
#       "b64_json": null
#     },
#     {
#       "url": "https://oaidalleapiprodscus.blob.core.windows.net/***",
#       "b64_json": null
#     }
#   ]
# }

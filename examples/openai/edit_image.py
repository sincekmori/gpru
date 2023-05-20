import os
from pathlib import Path

from gpru.openai.api import ImageEditing, OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key, timeout=60)

image_editing = ImageEditing(
    image=Path("/path/to/sunlit_lounge.png"),
    mask=Path("/path/to/mask.png"),
    prompt="A sunlit indoor lounge area with a pool containing a flamingo",
    n=2,
)
images = api.edit_image(image_editing)
print(images.json(indent=2))
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

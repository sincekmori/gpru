import os
from pathlib import Path

from gpru.openai.api import ImageVariation, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key, timeout=60)

image_variation = ImageVariation(image=Path("/path/to/corgi_and_cat_paw.png"), n=1)
image_list = api.create_image_variation(image_variation)
print(image_list.json(indent=2))
# Example output:
# {
#   "created": 1683647288,
#   "data": [
#     {
#       "url": "https://oaidalleapiprodscus.blob.core.windows.net/***",
#       "b64_json": null
#     }
#   ]
# }

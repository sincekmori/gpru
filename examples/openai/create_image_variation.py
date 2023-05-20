import os
from pathlib import Path

from gpru.openai.api import ImageVariation, OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key, timeout=60)

image_variation = ImageVariation(image=Path("/path/to/corgi_and_cat_paw.png"), n=1)
images = api.create_image_variation(image_variation)
print(images.json(indent=2))
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

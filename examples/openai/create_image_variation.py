import os
from pathlib import Path

from gpru.openai.api import ImageVariationRequest, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key, timeout=60)

req = ImageVariationRequest(image=Path("/path/to/corgi_and_cat_paw.png"), n=1)
image_list = api.create_image_variation(req)
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

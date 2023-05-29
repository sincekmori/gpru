import os

from gpru.openai.api import ImageRequest, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key, timeout=60)

req = ImageRequest(prompt="a white siamese cat", n=2)
image_list = api.create_images(req)
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

import os

from gpru.azure.preview_2023_06_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

operation = api.get_image_operation("f508bcf2-e651-4b4b-85a7-58ad77981ffa")
print(operation.json(indent=2))
# Example output:
# {
#   "id": "41dc2981-bf72-492a-b4fe-7eed680a1681",
#   "created": 1679320850,
#   "expires": 1679407255,
#   "result": {
#     "created": 1679320850,
#     "data": [
#       {
#         "url": "https://aoairesource.blob.core.windows.net/private/images?SAS-token",
#         "error": null
#       }
#     ]
#   },
#   "status": "succeeded",
#   "error": null
# }

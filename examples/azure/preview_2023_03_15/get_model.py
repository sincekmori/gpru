import os

from gpru.azure.preview_2023_03_15 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

model = api.get_model("curie")
print(model.json(indent=2))
# Example output:
# {
#   "object": "model",
#   "status": "succeeded",
#   "created_at": 1646126127,
#   "updated_at": 1646127311,
#   "id": "curie",
#   "model": null,
#   "fine_tune": null,
#   "capabilities": {
#     "fine_tune": true,
#     "inference": true,
#     "completion": true,
#     "chat_completion": false,
#     "embeddings": false,
#     "scale_types": [
#       "standard"
#     ]
#   },
#   "lifecycle_status": "generally-available",
#   "deprecation": {
#     "fine_tune": 1677662127,
#     "inference": 1709284527
#   }
# }

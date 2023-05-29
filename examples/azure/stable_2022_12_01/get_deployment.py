import os

from gpru.azure.stable_2022_12_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

deployment = api.get_deployment("deployment-afa0669ca01e4693ae3a93baf40f26d6")
print(deployment.json(indent=2))
# Example output:
# {
#   "object": "deployment",
#   "status": "succeeded",
#   "created_at": 1646126127,
#   "updated_at": 1646127311,
#   "id": "deployment-afa0669ca01e4693ae3a93baf40f26d6",
#   "model": "curie",
#   "owner": "organization-owner",
#   "scale_settings": {
#     "capacity": 2,
#     "scale_type": "manual"
#   },
#   "error": null
# }

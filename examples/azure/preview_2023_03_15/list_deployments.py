import os

from gpru.azure.preview_2023_03_15 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

deployments = api.list_deployments()
print(deployments.json(indent=2))
# Example output:
# {
#   "object": "list",
#   "data": [
#     {
#       "object": "deployment",
#       "status": "succeeded",
#       "created_at": 1646126127,
#       "updated_at": 1646127311,
#       "id": "deployment-afa0669ca01e4693ae3a93baf40f26d6",
#       "model": "curie",
#       "owner": "organization-owner",
#       "scale_settings": {
#         "capacity": 2,
#         "scale_type": "manual"
#       },
#       "error": null
#     }
#   ]
# }

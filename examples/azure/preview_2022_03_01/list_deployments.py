import os

from gpru.azure.preview_2022_03_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

deployment_list = api.list_deployments()
print(deployment_list.json(indent=2))
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
#       }
#     }
#   ]
# }

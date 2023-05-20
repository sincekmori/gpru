import os

from gpru.azure.stable_2022_12_01 import (
    AzureOpenAiApi,
    Deployment,
    ScaleSettings,
    ScaleType,
)

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

deployment = Deployment(
    model="curie",
    scale_settings=ScaleSettings(capacity=2, scale_type=ScaleType.MANUAL),
)
created_deployment = api.create_deployment(deployment)
print(created_deployment.json(indent=2))
# Example output:
# {
#   "object": "deployment",
#   "status": "notRunning",
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

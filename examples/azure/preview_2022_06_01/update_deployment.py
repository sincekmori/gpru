import os

from gpru.azure.preview_2022_06_01 import AzureOpenAiApi, ScaleSettings, ScaleType

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

deployment_id = "deployment-afa0669ca01e4693ae3a93baf40f26d6"
scale_settings = ScaleSettings(capacity=1, scale_type=ScaleType.MANUAL)
deployment = api.update_deployment(
    deployment_id=deployment_id, scale_settings=scale_settings
)
print(deployment.json(indent=2))
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
#     "capacity": 1,
#     "scale_type": "manual"
#   }
# }

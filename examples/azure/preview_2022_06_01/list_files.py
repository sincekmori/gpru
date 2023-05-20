import os

from gpru.azure.preview_2022_06_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

file_list = api.list_files()
print(file_list.json(indent=2))
# Example output:
# {
#   "object": "list",
#   "data": [
#     {
#       "object": "file",
#       "status": "succeeded",
#       "created_at": 1646126127,
#       "updated_at": 1646127311,
#       "id": "file-181a1cbdcdcf4677ada87f63a0928099",
#       "bytes": 140,
#       "purpose": "fine-tune",
#       "filename": "puppy.jsonl"
#     },
#     {
#       "object": "file",
#       "status": "succeeded",
#       "created_at": 1646126127,
#       "updated_at": 1646127311,
#       "id": "file-181a1cbdcdcf4677ada87f63a0928099",
#       "bytes": 32423,
#       "purpose": "fine-tune-results",
#       "filename": "results.csv"
#     }
#   ]
# }

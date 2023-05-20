import os

from gpru.azure.preview_2023_03_15 import AzureOpenAiApi, EmbeddingRequest

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, api_key)

req = EmbeddingRequest(input="This is a test.")
embedding = api.create_embedding(deployment_id, req)
print(embedding.json(indent=2))
# Example output:
# {
#   "object": "list",
#   "model": "ada",
#   "data": [
#     {
#       "index": 0,
#       "object": "embedding",
#       "embedding": [
#         -0.0035420605,
#         -0.004260177,
#         0.001081218,
#         .
#         .
#         .
#         -0.020702453,
#         0.008449189,
#         -0.00050300494
#       ]
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 5,
#     "total_tokens": 5
#   }
# }

import os

from gpru.openai.api import EmbeddingRequest, OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key)

req = EmbeddingRequest(
    model="text-embedding-ada-002", input="The food was delicious and the waiter..."
)
embedding = api.create_embedding(req)
print(embedding.json(indent=2))
# Example output:
# {
#   "object": "list",
#   "model": "text-embedding-ada-002-v2",
#   "data": [
#     {
#       "index": 0,
#       "object": "embedding",
#       "embedding": [
#         0.0022356957,
#         -0.009273057,
#         0.015815007,
#         .
#         .
#         .
#         -0.015357706,
#         -0.019397201,
#         -0.0028613096
#       ]
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 8,
#     "total_tokens": 8
#   }
# }

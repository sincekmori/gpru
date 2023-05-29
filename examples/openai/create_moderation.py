import os

from gpru.openai.api import ModerationRequest, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

req = ModerationRequest(input="I want to kill them.")
moderation = api.create_moderation(req)
print(moderation.json(indent=2))
# Example output:
# {
#   "id": "modr-5MWoLO",
#   "model": "text-moderation-001",
#   "results": [
#     {
#       "flagged": true,
#       "categories": {
#         "hate": false,
#         "hate_threatening": true,
#         "self_harm": false,
#         "sexual": false,
#         "sexual_minors": false,
#         "violence": true,
#         "violence_graphic": false
#       },
#       "category_scores": {
#         "hate": 0.22714105248451233,
#         "hate_threatening": 0.4132447838783264,
#         "self_harm": 0.005232391878962517,
#         "sexual": 0.01407341007143259,
#         "sexual_minors": 0.0038522258400917053,
#         "violence": 0.9223177433013916,
#         "violence_graphic": 0.036865197122097015
#       }
#     }
#   ]
# }

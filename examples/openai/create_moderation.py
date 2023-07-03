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
#   "model": "text-moderation-005",
#   "results": [
#     {
#       "flagged": true,
#       "categories": {
#         "hate": false,
#         "hate_threatening": false,
#         "self_harm": false,
#         "self_harm_instructions": false,
#         "self_harm_intent": false,
#         "sexual": false,
#         "sexual_minors": false,
#         "violence": true,
#         "violence_graphic": false,
#         "harassment": false,
#         "harassment_threatening": true
#       },
#       "category_scores": {
#         "hate": 0.010125076,
#         "hate_threatening": 0.005791877,
#         "self_harm": 1.4948828e-08,
#         "self_harm_instructions": 2.6517654e-11,
#         "self_harm_intent": 7.9006474e-10,
#         "sexual": 1.1691392e-06,
#         "sexual_minors": 5.2431027e-08,
#         "violence": 0.990058,
#         "violence_graphic": 4.3258656e-06,
#         "harassment": 0.289753,
#         "harassment_threatening": 0.621052
#       }
#     }
#   ]
# }

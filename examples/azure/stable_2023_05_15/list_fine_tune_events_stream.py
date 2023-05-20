import os

from gpru.azure.stable_2023_05_15 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

for fine_tune_event in api.list_fine_tune_events(
    fine_tune_id="ft-72a2792ef7d24ba7b82c7fe4a37e379f", stream=True
):
    print(fine_tune_event.json(indent=2))  # type: ignore[union-attr]
# Example output:
# {
#   "object": "fine-tune-event",
#   "created_at": 1646126127,
#   "level": "info",
#   "message": "Job enqueued. Waiting for jobs ahead to complete."
# }
# {
#   "object": "fine-tune-event",
#   "created_at": 1646126169,
#   "level": "info",
#   "message": "Job started."
# }
# {
#   "object": "fine-tune-event",
#   "created_at": 1646126192,
#   "level": "info",
#   "message": "Job succeeded."
# }

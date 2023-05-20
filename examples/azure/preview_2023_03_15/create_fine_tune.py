import os

from gpru.azure.preview_2023_03_15 import AzureOpenAiApi, FineTuneRequest

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

req = FineTuneRequest(
    model="curie", training_file="file-181a1cbdcdcf4677ada87f63a0928099"
)
fine_tune = api.create_fine_tune(req)
print(fine_tune.json(indent=2))
# Example output:
# {
#   "object": "fine-tune",
#   "status": "notRunning",
#   "created_at": 1646126127,
#   "updated_at": 1646127311,
#   "id": "ft-72a2792ef7d24ba7b82c7fe4a37e379f",
#   "model": "curie",
#   "fine_tuned_model": null,
#   "training_files": [
#     {
#       "object": "file",
#       "status": "succeeded",
#       "created_at": 1646126127,
#       "updated_at": 1646127311,
#       "id": "file-181a1cbdcdcf4677ada87f63a0928099",
#       "bytes": 140,
#       "purpose": "fine-tune",
#       "filename": "puppy.jsonl",
#       "statistics": {
#         "tokens": 42,
#         "examples": 23
#       },
#       "error": null
#     }
#   ],
#   "validation_files": null,
#   "result_files": null,
#   "events": null,
#   "organisation_id": null,
#   "user_id": null,
#   "hyperparams": {
#     "batch_size": 32,
#     "learning_rate_multiplier": 1.0,
#     "n_epochs": 2,
#     "prompt_loss_weight": 0.1,
#     "compute_classification_metrics": null,
#     "classification_n_classes": null,
#     "classification_positive_class": null,
#     "classification_betas": null
#   },
#   "suffix": null,
#   "error": null
# }

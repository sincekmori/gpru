import os

from gpru.azure.preview_2023_06_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

fine_tune = api.get_fine_tune("ft-72a2792ef7d24ba7b82c7fe4a37e379f")
print(fine_tune.json(indent=2))
# Example output:
# {
#   "object": "fine-tune",
#   "status": "succeeded",
#   "created_at": 1646126127,
#   "updated_at": 1646127311,
#   "id": "ft-72a2792ef7d24ba7b82c7fe4a37e379f",
#   "model": "curie",
#   "fine_tuned_model": "curie.ft-72a2792ef7d24ba7b82c7fe4a37e379f",
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
#   "result_files": [
#     {
#       "object": "file",
#       "status": "succeeded",
#       "created_at": 1646126127,
#       "updated_at": 1646127311,
#       "id": "file-181a1cbdcdcf4677ada87f63a0928099",
#       "bytes": 32423,
#       "purpose": "fine-tune-results",
#       "filename": "results.csv",
#       "statistics": null,
#       "error": null
#     }
#   ],
#   "events": [
#     {
#       "object": "fine-tune-event",
#       "created_at": 1646126127,
#       "level": "info",
#       "message": "Job enqueued. Waiting for jobs ahead to complete."
#     },
#     {
#       "object": "fine-tune-event",
#       "created_at": 1646126169,
#       "level": "info",
#       "message": "Job started."
#     },
#     {
#       "object": "fine-tune-event",
#       "created_at": 1646126192,
#       "level": "info",
#       "message": "Job succeeded."
#     }
#   ],
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

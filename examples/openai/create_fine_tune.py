import os

from gpru.openai.api import FineTuneRequest, OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key)

req = FineTuneRequest(training_file="file-XGinujblHPwGLSztz8cPS8XY")
fine_tune = api.create_fine_tune(req)
print(fine_tune.json(indent=2))
# Example output:
# {
#   "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
#   "object": "fine-tune",
#   "created_at": 1614807352,
#   "updated_at": 1614807352,
#   "model": "curie",
#   "fine_tuned_model": null,
#   "organization_id": "org-...",
#   "status": "pending",
#   "hyperparams": {
#     "batch_size": 4,
#     "learning_rate_multiplier": 0.1,
#     "n_epochs": 4,
#     "prompt_loss_weight": 0.1
#   },
#   "training_files": [
#     {
#       "id": "file-XGinujblHPwGLSztz8cPS8XY",
#       "object": "file",
#       "bytes": 1547276,
#       "created_at": 1610062281,
#       "filename": "my-data-train.jsonl",
#       "purpose": "fine-tune-train",
#       "status": null,
#       "status_details": null
#     }
#   ],
#   "validation_files": [],
#   "result_files": [],
#   "events": [
#     {
#       "object": "fine-tune-event",
#       "created_at": 1614807352,
#       "level": "info",
#       "message": "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
#     }
#   ]
# }

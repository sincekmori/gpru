import os

from gpru.azure.preview_2023_06_01 import AzureOpenAiApi, CompletionRequest

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, key)

req = CompletionRequest(prompt="Say this is a test.")
completion = api.create_completion(deployment_id, req)
print(completion.text)  # type: ignore[union-attr]
# Example output:
# This is indeed a test!

import os

from gpru.azure.stable_2022_12_01 import AzureOpenAiApi, CompletionRequest

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, api_key)

req = CompletionRequest(prompt="Say this is a test.")
completion = api.create_completion(deployment_id, req)
print(completion.text)  # type: ignore[union-attr]
# Example output:
# This is indeed a test!

import os

from gpru.azure.preview_2022_06_01 import AzureOpenAiApi, CompletionRequest

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, api_key)

req = CompletionRequest(prompt="Say this is a test.", stream=True)
for completion in api.create_completion(deployment_id, req):
    print(completion.text, end="")  # type: ignore[union-attr]
# Example output:
# This is indeed a test!

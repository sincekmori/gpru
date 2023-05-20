import os

from gpru.azure.preview_2023_03_15 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

model_list = api.list_models()
print(model_list.ids)
# Example output:
# ["curie", "curie.ft-72a2792ef7d24ba7b82c7fe4a37e379f"]

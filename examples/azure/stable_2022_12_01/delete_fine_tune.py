import os

from gpru.azure.stable_2022_12_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

api.delete_fine_tune("ft-72a2792ef7d24ba7b82c7fe4a37e379f")

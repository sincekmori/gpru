import os

from gpru.azure.stable_2023_05_15 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

api.delete_file("file-181a1cbdcdcf4677ada87f63a0928099")

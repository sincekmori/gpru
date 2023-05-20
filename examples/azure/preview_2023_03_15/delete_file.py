import os

from gpru.azure.preview_2023_03_15 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

api.delete_file("file-181a1cbdcdcf4677ada87f63a0928099")

import os

from gpru.azure.stable_2022_12_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

api.delete_file("file-181a1cbdcdcf4677ada87f63a0928099")

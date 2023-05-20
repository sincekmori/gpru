import os

from gpru.azure.preview_2022_03_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

api.delete_deployment("deployment-afa0669ca01e4693ae3a93baf40f26d6")

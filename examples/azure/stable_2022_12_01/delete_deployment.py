import os

from gpru.azure.stable_2022_12_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

api.delete_deployment("deployment-afa0669ca01e4693ae3a93baf40f26d6")

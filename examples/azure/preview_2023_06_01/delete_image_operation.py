import os

from gpru.azure.preview_2023_06_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, key)

api.delete_image_operation("f508bcf2-e651-4b4b-85a7-58ad77981ffa")

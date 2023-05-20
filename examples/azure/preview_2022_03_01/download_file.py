import os

from gpru.azure.preview_2022_03_01 import AzureOpenAiApi

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api = AzureOpenAiApi(endpoint, api_key)

file_content = api.download_file("file-181a1cbdcdcf4677ada87f63a0928099")
print(file_content)
# Example output:
# {"prompt": "Lorem ipsum", "completion": "dolor sit amet"}
# {"prompt": "consectetur adipiscing elit", "completion": "sed do eiusmod"}
# {"prompt": "tempor incididunt ut", "completion": "labore et dolore"}

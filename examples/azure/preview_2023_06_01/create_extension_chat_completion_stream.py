import os

from gpru.azure.preview_2023_06_01 import (
    AzureOpenAiApi,
    DataSource,
    ExtensionChatCompletionRequest,
    ExtensionMessage,
    Role,
)

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, key)

req = ExtensionChatCompletionRequest(
    data_sources=[
        DataSource(
            type="AzureCognitiveSearch",
            parameters={
                "endpoint": "https://mysearchexample.search.windows.net",
                "key": "***(admin key)",
                "indexName": "my-chunk-index",
                "fieldsMapping": {
                    "titleField": "productName",
                    "urlField": "productUrl",
                    "filepathField": "productFilePath",
                    "contentFields": ["productDescription"],
                    "contentFieldsSeparator": "\n",
                },
                "topNDocuments": 5,
                "queryType": "semantic",
                "semanticConfiguration": "defaultConfiguration",
                "inScope": True,
                "roleInformation": "roleInformation",
            },
        )
    ],
    messages=[
        ExtensionMessage(
            role=Role.USER, content="Where can I find a hiking place in Seattle?"
        )
    ],
    temperature=0.9,
)
for _ in api.create_extension_chat_completion(deployment_id, req):
    raise NotImplementedError

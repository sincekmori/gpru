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
    dataSources=[
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
extension_chat_completion = api.create_extension_chat_completion(deployment_id, req)
print(extension_chat_completion.json(indent=2))  # type: ignore[union-attr]
# Example output:
# {
#   "id": "1",
#   "object": "extensions.chat.completion",
#   "created": 1679201802,
#   "model": "gpt-3.5-turbo-0301",
#   "choices": [
#     {
#       "index": 0,
#       "messages": [
#         {
#           "index": null,
#           "role": "tool",
#           "recipient": null,
#           "content": "{\"citations\":[{\"filepath\":\"ContosoTraveler.pdf\",\"content\":\"This is the content of the citation 1\"},{\"filepath\":\"WestCoastTraveler.html\",\"content\":\"This is the content of the citation 2\"},{\"content\":\"This is the content of the citation 3 without filepath\"}],\"intent\":\"hiking place in seattle\"}",
#           "end_turn": false
#         },
#         {
#           "index": null,
#           "role": "assistant",
#           "recipient": null,
#           "content": "Seattle is a great place for hiking! Here are some of the best hiking places in Seattle according to Contoso Traveler [doc1] and West Coast Traveler, Snow Lake, Mount Si, and Mount Tenerife [doc2]. I hope this helps! Let me know if you need more information.",
#           "end_turn": true
#         }
#       ],
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": null
# }

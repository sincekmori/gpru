import os

from gpru.azure.preview_2023_06_01 import (
    AzureOpenAiApi,
    ChatCompletionRequest,
    UserMessage,
)

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, key)

req = ChatCompletionRequest(messages=[UserMessage("Hello!")], stream=True)
for chat_completion in api.create_chat_completion(deployment_id, req):
    print(chat_completion.delta_content, end="")  # type: ignore[union-attr]
# Example output:
# Hi there! How can I assist you today?

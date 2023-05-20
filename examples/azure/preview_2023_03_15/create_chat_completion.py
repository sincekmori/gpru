import os

from gpru.azure.preview_2023_03_15 import (
    AzureOpenAiApi,
    ChatCompletionRequest,
    UserMessage,
)

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, api_key)

req = ChatCompletionRequest(messages=[UserMessage("Hello!")])
chat_completion = api.create_chat_completion(deployment_id, req)
print(chat_completion.content)  # type: ignore[union-attr]
# Example output:
# Greetings! How can I assist you today?

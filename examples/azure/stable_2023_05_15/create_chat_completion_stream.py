import os

from gpru.azure.stable_2023_05_15 import (
    AzureOpenAiApi,
    ChatCompletionRequest,
    UserMessage,
)

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, api_key)

req = ChatCompletionRequest(messages=[UserMessage("Hello!")], stream=True)
for chat_completion in api.create_chat_completion(deployment_id, req):
    print(chat_completion.delta_content, end="")  # type: ignore[union-attr]
# Example output:
# Hi there! How can I assist you today?

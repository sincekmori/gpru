import os

from gpru.openai.api import (
    ChatCompletionModel,
    ChatCompletionRequest,
    OpenAiApi,
    UserMessage,
)

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

req = ChatCompletionRequest(
    model=ChatCompletionModel.GPT_35_TURBO, messages=[UserMessage("Hello!")]
)
chat_completion = api.create_chat_completion(req)
print(chat_completion.content)  # type: ignore[union-attr]
# Example output:
# Greetings! How can I assist you today?

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
    model=ChatCompletionModel.GPT_35_TURBO,
    messages=[UserMessage("Hello!")],
    stream=True,
)
for chat_completion in api.create_chat_completion(req):
    print(chat_completion.delta_content, end="")  # type: ignore[union-attr]
# Example output:
# Hi there! How can I assist you today?

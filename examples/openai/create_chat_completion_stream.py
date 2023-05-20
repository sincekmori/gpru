import os

from gpru.openai.api import ChatCompletionRequest, OpenAiApi, UserMessage

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key)

req = ChatCompletionRequest(
    model="gpt-3.5-turbo",
    messages=[UserMessage("Hello!")],
    stream=True,
)
for chat_completion in api.create_chat_completion(req):
    print(chat_completion.delta_content, end="")  # type: ignore[union-attr]
# Example output:
# Hi there! How can I assist you today?

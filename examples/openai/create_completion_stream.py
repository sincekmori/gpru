import os

from gpru.openai.api import CompletionModel, CompletionRequest, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

req = CompletionRequest(
    model=CompletionModel.TEXT_DAVINCI_003, prompt="Say this is a test.", stream=True
)
for completion in api.create_completion(req):
    print(completion.text, end="")  # type: ignore[union-attr]
# Example output:
# This is indeed a test!

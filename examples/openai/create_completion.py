import os

from gpru.openai.api import CompletionModel, CompletionRequest, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

req = CompletionRequest(
    model=CompletionModel.TEXT_DAVINCI_003, prompt="Say this is a test."
)
completion = api.create_completion(req)
print(completion.text)  # type: ignore[union-attr]
# Example output:
# This is indeed a test!

import os

from gpru.openai.api import CompletionRequest, OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key)

req = CompletionRequest(model="text-davinci-003", prompt="Say this is a test.")
completion = api.create_completion(req)
print(completion.text)  # type: ignore[union-attr]
# Example output:
# This is indeed a test!

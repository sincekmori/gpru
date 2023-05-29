import os

from gpru.openai.api import EditRequest, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

req = EditRequest(
    model="text-davinci-edit-001",
    input="What day of the wek is it?",
    instruction="Fix the spelling mistakes",
)
edit = api.create_edit(req)
print(edit.text)
# Example output:
# What day of the week is it?

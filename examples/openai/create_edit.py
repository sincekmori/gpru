import os

from gpru.openai.api import EditModel, EditRequest, OpenAiApi

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

req = EditRequest(
    model=EditModel.TEXT_DAVINCI_EDIT_001,
    input="What day of the wek is it?",
    instruction="Fix the spelling mistakes",
)
edit = api.create_edit(req)
print(edit.text)
# Example output:
# What day of the week is it?

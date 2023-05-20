import os

from gpru.openai.api import OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key)

content = api.download_file(file_id="file-XjGxS3KTG0uNmNOK362iJua3")
print(content)
# Example output:
# {"prompt": "Lorem ipsum", "completion": "dolor sit amet"}
# {"prompt": "consectetur adipiscing elit", "completion": "sed do eiusmod"}
# {"prompt": "tempor incididunt ut", "completion": "labore et dolore"}

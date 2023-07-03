import os
from typing import Any, Dict

from gpru.openai.api import (
    ChatCompletionModel,
    ChatCompletionRequest,
    Function,
    Message,
    OpenAiApi,
    Role,
)

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)


def get_current_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
    """Get the current weather in a given location."""
    return {
        "location": location,
        "temperature": 72,
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


req = ChatCompletionRequest(
    model=ChatCompletionModel.GPT_35_TURBO,
    messages=[Message(role=Role.USER, content="What's the weather like in Boston?")],
    functions=[
        Function(
            name="get_current_weather",
            description="Get the current weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        )
    ],
    function_call="auto",
)
chat_completion = api.create_chat_completion(req)
message = chat_completion.messages[0]  # type: ignore[union-attr]
print(message.json(indent=2))
# Example output:
# {
#   "role": "assistant",
#   "content": null,
#   "function_call": {
#     "name": "get_current_weather",
#     "arguments": "{\n  \"location\": \"Boston, MA\"\n}"
#   }
# }

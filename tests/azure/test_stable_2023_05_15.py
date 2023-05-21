import os

import pytest

from gpru.azure.stable_2023_05_15 import (
    AzureOpenAiApi,
    ChatCompletionRequest,
    CompletionRequest,
    UserMessage,
)
from gpru.exceptions import ApiError

PREFIX = "AZURE_OPENAI_API"
endpoint = os.environ[f"{PREFIX}_ENDPOINT"]
api_key = os.environ[f"{PREFIX}_KEY"]
api = AzureOpenAiApi(endpoint, api_key, timeout=30)

completion_deployment_id = os.environ[f"{PREFIX}_COMPLETION_DEPLOYMENT_ID"]
chat_completion_deployment_id = os.environ[f"{PREFIX}_CHAT_COMPLETION_DEPLOYMENT_ID"]


def test_completion() -> None:
    req = CompletionRequest(prompt="Say this is a test.")
    completion = api.create_completion(completion_deployment_id, req)
    assert len(completion.text) > 0  # type: ignore[union-attr]


def test_completion_stream() -> None:
    req = CompletionRequest(prompt="Say this is a test.", stream=True)
    for completion in api.create_completion(completion_deployment_id, req):
        assert type(completion.text) == str  # type: ignore[union-attr]


def test_chat_completion() -> None:
    req = ChatCompletionRequest(messages=[UserMessage("Hello!")])
    chat_completion = api.create_chat_completion(chat_completion_deployment_id, req)
    assert len(chat_completion.content) > 0  # type: ignore[union-attr]


def test_chat_completion_stream() -> None:
    req = ChatCompletionRequest(messages=[UserMessage("Hello!")], stream=True)
    for chat_completion in api.create_chat_completion(
        chat_completion_deployment_id, req
    ):
        assert type(chat_completion.content) == str  # type: ignore[union-attr]


def test_models() -> None:
    model_list = api.list_models()
    model_id = model_list.ids[0]
    model = api.get_model(model_id)
    assert model.id == model_id


def test_error_response() -> None:
    with pytest.raises(ApiError):
        # There is no model with ID "foo".
        api.get_model("foo")

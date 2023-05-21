import os
from pathlib import Path
from time import sleep

import pytest

from gpru.exceptions import ApiError
from gpru.openai.api import (
    ChatCompletionRequest,
    CompletionRequest,
    EditRequest,
    EmbeddingRequest,
    OpenAiApi,
    UserMessage,
)

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key, timeout=30)


def test_completion() -> None:
    req = CompletionRequest(model="text-davinci-003", prompt="Say this is a test.")
    completion = api.create_completion(req)
    assert len(completion.text) > 0  # type: ignore[union-attr]


def test_completion_stream() -> None:
    req = CompletionRequest(
        model="text-davinci-003", prompt="Say this is a test.", stream=True
    )
    for completion in api.create_completion(req):
        assert type(completion.text) == str  # type: ignore[union-attr]


def test_chat_completion() -> None:
    req = ChatCompletionRequest(model="gpt-3.5-turbo", messages=[UserMessage("Hello!")])
    chat_completion = api.create_chat_completion(req)
    assert len(chat_completion.content) > 0  # type: ignore[union-attr]


def test_chat_completion_stream() -> None:
    req = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[UserMessage("Hello!")],
        stream=True,
    )
    for chat_completion in api.create_chat_completion(req):
        assert type(chat_completion.content) == str  # type: ignore[union-attr]


def test_edit() -> None:
    req = EditRequest(
        model="text-davinci-edit-001",
        input="What day of the wek is it?",
        instruction="Fix the spelling mistakes",
    )
    edit = api.create_edit(req)
    assert len(edit.text) > 0


def test_embedding() -> None:
    req = EmbeddingRequest(
        model="text-embedding-ada-002", input="The food was delicious and the waiter..."
    )
    api.create_embedding(req)


def test_files(data_dir: Path) -> None:
    file_path = data_dir / "fine-tune.jsonl"
    assert file_path.exists()

    file = api.upload_file(file_path, "fine-tune")

    file_list = api.list_files()
    assert file.id in file_list.ids

    specified_file = api.get_file(file.id)
    assert specified_file.id == file.id

    sleep(4)  # files cannot be deleted immediately after upload
    api.delete_file(file.id)
    with pytest.raises(ApiError):
        api.get_file(file.id)


def test_models() -> None:
    model_list = api.list_models()
    assert len(model_list.ids) > 0
    model_id = model_list.ids[0]
    model = api.get_model(model_id)
    assert model.id == model_id


def test_error_response() -> None:
    with pytest.raises(ApiError):
        # There is no model with ID "foo".
        api.get_model("foo")

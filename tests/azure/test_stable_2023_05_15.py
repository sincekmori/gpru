import os

import pytest

from gpru.azure.stable_2023_05_15 import AzureOpenAiApi
from gpru.exceptions import ApiError

PREFIX = "AZURE_OPENAI_API"
endpoint = os.environ[f"{PREFIX}_ENDPOINT"]
key = os.environ[f"{PREFIX}_KEY"]
api = AzureOpenAiApi(endpoint, key, timeout=30)


def test_models() -> None:
    model_list = api.list_models()
    model_id = model_list.ids[0]
    model = api.get_model(model_id)
    assert model.id == model_id


def test_error_response() -> None:
    with pytest.raises(ApiError):
        # There is no model with ID "foo".
        api.get_model("foo")

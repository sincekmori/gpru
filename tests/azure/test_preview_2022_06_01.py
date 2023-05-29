import os

import pytest

from gpru.azure.preview_2022_06_01 import AzureOpenAiApi
from gpru.exceptions import ApiError

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
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

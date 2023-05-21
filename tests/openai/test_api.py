import os

import pytest

from gpru.exceptions import ApiError
from gpru.openai.api import OpenAiApi

api_key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(api_key, timeout=30)


def test_models() -> None:
    model_list = api.list_models()
    model_id = model_list.ids[0]
    model = api.get_model(model_id)
    assert model.id == model_id


def test_error_response() -> None:
    with pytest.raises(ApiError):
        # There is no model with ID "foo".
        api.get_model("foo")

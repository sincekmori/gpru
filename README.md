# GPRU: Unofficial Python Client Library for the OpenAI and Azure OpenAI APIs

**PyPI**
[![PyPI - Version](https://img.shields.io/pypi/v/gpru.svg)](https://pypi.org/project/gpru)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gpru.svg)](https://pypi.org/project/gpru)
[![License](https://img.shields.io/pypi/l/gpru.svg)](https://github.com/sincekmori/gpru/blob/main/LICENSE)

**CI/CD**
[![test](https://github.com/sincekmori/gpru/actions/workflows/test.yml/badge.svg)](https://github.com/sincekmori/gpru/actions/workflows/test.yml)
[![lint](https://github.com/sincekmori/gpru/actions/workflows/lint.yml/badge.svg)](https://github.com/sincekmori/gpru/actions/workflows/lint.yml)

**Build System**
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

**Code**
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

**Docstrings**
[![docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![numpy](https://img.shields.io/badge/%20style-numpy-459db9.svg)](https://numpydoc.readthedocs.io/en/latest/format.html)

---

GPRU is an unofficial Python client library for the OpenAI and Azure OpenAI APIs with improved usability through comprehensive annotations.

**WARNING**: GPRU is currently under development and any destructive changes may be made until version `1.0.0`.

## Installation

```console
pip install gpru
```

## Examples

### OpenAI API

**Notes** Before anything else, you must ensure that the [API key](https://platform.openai.com/account/api-keys) is set as an environment variable named `OPENAI_API_KEY`.

Here is an example of [chat completion](https://platform.openai.com/docs/api-reference/chat/create), also known as [ChatGPT](https://chat.openai.com/).

```python
import os

from gpru.openai.api import (
    ChatCompletionModel,
    ChatCompletionRequest,
    OpenAiApi,
    UserMessage,
)

key = os.environ["OPENAI_API_KEY"]
api = OpenAiApi(key)

req = ChatCompletionRequest(
    model=ChatCompletionModel.GPT_35_TURBO, messages=[UserMessage("Hello!")]
)
chat_completion = api.create_chat_completion(req)
print(chat_completion.content)
# Greetings! How can I assist you today?
```

### Azure OpenAI API

The following code replaces the same example with the [Azure OpenAI API](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/).

**Notes** Set the following environment variables before executing the program.

| Variable name                    | Value                                                                |
| -------------------------------- | -------------------------------------------------------------------- |
| `AZURE_OPENAI_API_ENDPOINT`      | URL in the form of `https://{your-resource-name}.openai.azure.com/`. |
| `AZURE_OPENAI_API_KEY`           | Secret key.                                                          |
| `AZURE_OPENAI_API_DEPLOYMENT_ID` | Custom name you chose for your deployment when you deployed a model. |

```python
import os

from gpru.azure.preview_2023_06_01 import (
    AzureOpenAiApi,
    ChatCompletionRequest,
    UserMessage,
)

endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_id = os.environ["AZURE_OPENAI_API_DEPLOYMENT_ID"]
api = AzureOpenAiApi(endpoint, key)

req = ChatCompletionRequest(messages=[UserMessage("Hello!")])
chat_completion = api.create_chat_completion(deployment_id, req)
print(chat_completion.content)
# Greetings! How can I assist you today?
```

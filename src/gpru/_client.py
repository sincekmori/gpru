from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Type, TypeVar

import httpx
from httpx._exceptions import HTTPError, StreamError
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError

from gpru.exceptions import ApiError, IncorrectImplementationError

T = TypeVar("T", bound=BaseModel)


def _raise_api_error(error_response_model: Type[T], response: httpx.Response) -> None:
    error_response = error_response_model.parse_obj(response.json())
    error = error_response.error if hasattr(error_response, "error") else error_response
    raise ApiError(response.status_code, error)


def request_factory(
    error_response_model: Type[T], **client_kwargs: Any
) -> Callable[..., httpx.Response]:
    def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.request(method, path, **kwargs)
                if not response.is_success:
                    _raise_api_error(error_response_model, response)
                return response
        except HTTPError as e:
            raise ApiError(None, e) from e
        except ValidationError as e:
            raise IncorrectImplementationError from e

    return request


def _generate_chunk_models(
    response: httpx.Response, response_model: Type[T]
) -> Generator[T, None, None]:
    for chunk in response.iter_raw():
        for line in chunk.splitlines():
            c = line.strip()

            if c.startswith(b"data: [DONE]"):
                continue

            if not c.startswith(b"data: "):
                continue

            c = c[len(b"data: ") :]
            yield response_model.parse_raw(c)


def stream_factory(
    error_response_model: Type[T],
    **client_kwargs: Any,
) -> Callable[..., Generator[T, None, None]]:
    def stream(
        response_model: Type[T], method: str, path: str, **kwargs: Any
    ) -> Generator[T, None, None]:
        try:
            with httpx.Client(**client_kwargs) as client:  # noqa: SIM117
                with client.stream(method, path, **kwargs) as response:
                    if not response.is_success:
                        _raise_api_error(error_response_model, response)
                    yield from _generate_chunk_models(response, response_model)
        except (HTTPError, StreamError) as e:
            raise ApiError(None, e) from e
        except ValidationError as e:
            raise IncorrectImplementationError from e

    return stream


def kwargs_for_uploading(request_model: T) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    files: Dict[str, Any] = {}
    for k, v in request_model.dict(exclude_none=True).items():
        if isinstance(v, Path):
            files[k] = v.open("rb")
        else:
            value = v.value if isinstance(v, Enum) else v
            data[k] = value
    return {"data": data or None, "files": files or None}

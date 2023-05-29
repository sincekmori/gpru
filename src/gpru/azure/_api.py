from typing import Dict, Optional, Type

from httpx._config import DEFAULT_TIMEOUT_CONFIG
from httpx._types import TimeoutTypes

from gpru._client import T, request_factory, stream_factory
from gpru.exceptions import InvalidConfigError


class Api:
    def __init__(  # noqa: PLR0913
        self,
        error_response_model: Type[T],
        endpoint: str,
        api_version: str,
        key: Optional[str] = None,
        ad_token: Optional[str] = None,
        timeout: Optional[TimeoutTypes] = DEFAULT_TIMEOUT_CONFIG,
    ) -> None:
        client_kwargs = {
            "params": {"api-version": api_version},
            "headers": self._build_base_headers(key, ad_token),
            "http2": True,
            "base_url": f"{endpoint}openai"
            if endpoint.endswith("/")
            else f"{endpoint}/openai",
            "timeout": timeout,
        }
        self._request = request_factory(error_response_model, **client_kwargs)
        self._stream = stream_factory(error_response_model, **client_kwargs)

    def _build_base_headers(
        self, key: Optional[str], ad_token: Optional[str]
    ) -> Dict[str, str]:
        both_none: bool = key is None and ad_token is None
        both_not_none: bool = key is not None and ad_token is not None
        if both_none or both_not_none:
            msg = "Either `key` or `ad_token` is required."
            raise InvalidConfigError(msg)

        return (
            {"api-key": key}
            if key is not None
            else {"Authorization": f"Bearer {ad_token}"}
        )

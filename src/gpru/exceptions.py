from typing import Any, Optional


class GpruError(Exception):
    """A generic, GPRU-specific error."""


class InvalidConfigError(GpruError, ValueError):
    pass


class IncorrectImplementationError(GpruError, RuntimeError):
    def __init__(self) -> None:
        message = "If you encounter this error, please submit an issue at https://github.com/sincekmori/gpru/issues/new with a traceback."  # noqa: E501
        super().__init__(message)


class ApiError(GpruError):
    def __init__(
        self, status_code: Optional[int] = None, error: Optional[Any] = None
    ) -> None:
        self.status_code = status_code
        self.error = error

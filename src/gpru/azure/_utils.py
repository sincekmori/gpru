from warnings import warn

from gpru._logger import logger


def deprecation_warning(module: str) -> None:
    latest_module = "gpru.azure.preview_2023_06_01"
    message = f"`{module}` is outdated. Consider using `{latest_module}` instead."
    warn(message, stacklevel=1)
    logger.warning(message)

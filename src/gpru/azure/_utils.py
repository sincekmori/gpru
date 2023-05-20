from warnings import warn


def deprecation_warning(module: str) -> None:
    latest_module = "gpru.azure.stable_2023_05_15"
    message = f"{module} is outdated. Consider using {latest_module} instead."
    warn(message, stacklevel=1)

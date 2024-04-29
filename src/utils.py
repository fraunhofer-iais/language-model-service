import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_app_config_name_based_on_environment():
    environment = os.getenv("ENVIRONMENT")
    if environment is None:
        raise ValueError("Environment variable 'ENVIRONMENT' not set but needs to be")
    if environment == "local":
        return "local_app_config"
    if environment == "prod":
        return "app_config"
    else:
        raise ValueError(f"Environment variable 'ENVIRONMENT' is set to '{environment}' but only 'prod' and 'local' are valid")


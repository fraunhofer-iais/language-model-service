import sys

import hydra
import uvicorn
from omegaconf import OmegaConf, DictConfig

from src.app import setup_logging, load_app
from src.config.app_config import AppConfig, Platforms
from src.utils import get_app_config_name_based_on_environment

config_dict = None


@hydra.main(version_base=None, config_path="../config_files", config_name="app_config")
def hydra_load_app_config_dict(cfg: DictConfig):
    global config_dict
    config_dict = OmegaConf.to_container(cfg)


def parse_app_config():
    return AppConfig.model_validate(config_dict)


if __name__ == "__main__":
    setup_logging()
    hydra_load_app_config_dict()
    app_config: AppConfig = parse_app_config()
    platform = sys.platform
    print("Platform: ", platform)
    print("Config:\n", app_config)
    app = load_app(app_config)
    host = "127.0.0.1" if "win" in platform else "0.0.0.0"
    uvicorn.run(app, host=host, port=app_config.port, proxy_headers=True)

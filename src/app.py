import logging.config
import os
from pathlib import Path

import yaml
from fastapi import FastAPI, APIRouter

from src.config.app_config import AppConfig
from src.service.generator_service import GeneratorService


def setup_logging():
    log_yml = os.path.join(Path(os.path.dirname(__file__)).parent, "logging.yml")

    with open(log_yml, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    logging.config.dictConfig(config)


def load_generator_service_endpoints(
    router: APIRouter,
    app_config: AppConfig,
) -> APIRouter:
    generator_service = GeneratorService(
        config=app_config.generators,
    )
    router.add_api_route(
        "/generate", endpoint=generator_service.generate, methods=["POST"]
    )
    router.add_api_route(
        "/vectorize", endpoint=generator_service.vectorize, methods=["POST"]
    )
    router.add_api_route(
        "/available_models",
        endpoint=generator_service.available_models,
        methods=["GET"],
    )
    return router


def load_app(app_config: AppConfig) -> FastAPI:
    router = APIRouter()
    router = load_generator_service_endpoints(router=router, app_config=app_config)
    app = FastAPI()
    app.include_router(router)

    return app

from typing import List, Dict, Union

from fastapi import HTTPException

from lms_interface.interface.generator_interface import GeneratorAPI
from lms_interface.dto.generator_dtos import (
    OpenAIGeneratorConfig,
    HuggingFaceGeneratorConfig,
    AlephAlphaGeneratorConfig,
    SentenceTransformersGeneratorConfig,
)
from src.generator.aleph_alpha_generator import AlephAlphaGenerator
from src.generator.generator import Generator
from src.generator.huggingface_generator import HuggingFaceGenerator
from src.generator.openai_generator import OpenAIGenerator
from src.generator.sentence_transformers_generator import SentenceTransformersGenerator


class GeneratorService(GeneratorAPI):
    def __init__(
        self,
        config: Dict[
            str,
            Union[
                HuggingFaceGeneratorConfig,
                OpenAIGeneratorConfig,
                AlephAlphaGeneratorConfig,
                SentenceTransformersGeneratorConfig,
            ],
        ],
    ):
        self.__generators_config = config
        self.__generators = self.__load_generator(cfg=config)

    def generate(
        self,
        model_name: str,
        texts: List[str],
        max_new_tokens: int = 10,
        split_lines: bool = True,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0,
    ) -> List[str]:
        try:
            generator: Generator = self.__generators[model_name]
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"The model name {model_name} is not valid! Available "
                f"generator models: {list(self.__generators.keys())}",
            )
        return generator.generate(
            texts=texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            split_lines=split_lines,
        )

    def vectorize(
        self,
        model_name: str,
        texts: List[str],
    ) -> List[List[float]]:
        try:
            generator: Generator = self.__generators[model_name]
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"The model name {model_name} is not valid! Available "
                f"generator models: {list(self.__generators.keys())}",
            )
        try:
            vectorized = generator.vectorize(texts=texts)
            return vectorized
        except NotImplementedError:
            raise HTTPException(
                status_code=404,
                detail=f"The model name {model_name} is not valid for vectorization",
            )

    def available_models(
        self,
    ) -> Dict[str, Dict]:
        return {
            key: value.model_dump() for key, value in self.__generators_config.items()
        }

    @staticmethod
    def __load_generator(
        cfg: Dict[
            str,
            Union[
                HuggingFaceGeneratorConfig,
                OpenAIGeneratorConfig,
                AlephAlphaGeneratorConfig,
                SentenceTransformersGeneratorConfig,
            ],
        ]
    ) -> Dict[str, Union[HuggingFaceGenerator, OpenAIGenerator]]:
        generators_config = {}
        for config_name, config in cfg.items():
            if isinstance(config, HuggingFaceGeneratorConfig):
                generator = HuggingFaceGenerator(
                    **config.model_dump(exclude=["model_provider"])  # noqa
                )
                generators_config[config_name] = generator
            elif isinstance(config, OpenAIGeneratorConfig):
                generator = OpenAIGenerator(
                    **config.model_dump(exclude=["model_provider"])  # noqa
                )
                generators_config[config_name] = generator
            elif isinstance(config, AlephAlphaGeneratorConfig):
                generator = AlephAlphaGenerator(
                    **config.model_dump(exclude=["model_provider"])  # noqa
                )
                generators_config[config_name] = generator
            elif isinstance(config, SentenceTransformersGeneratorConfig):
                generator = SentenceTransformersGenerator(
                    **config.model_dump(exclude=["model_provider"])  # noqa
                )
                generators_config[config_name] = generator
        return generators_config

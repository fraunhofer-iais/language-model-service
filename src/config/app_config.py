from enum import Enum
from typing import Union, Dict

from pydantic import BaseModel, PositiveInt, Field
from typing_extensions import Annotated

from lms_interface.dto.generator_dtos import (
    OpenAIGeneratorConfig,
    HuggingFaceGeneratorConfig,
    AlephAlphaGeneratorConfig,
    SentenceTransformersGeneratorConfig,
)


class Platforms(str, Enum):
    LINUX = "linux"
    WINDOWS = "windows"
    MAC = "mac"


GeneratorConfig = Annotated[
    Union[
        OpenAIGeneratorConfig,
        HuggingFaceGeneratorConfig,
        AlephAlphaGeneratorConfig,
        SentenceTransformersGeneratorConfig,
    ],
    Field(discriminator="model_provider"),
]


class AppConfig(BaseModel):
    port: PositiveInt
    generators: Dict[
        str,
        GeneratorConfig,
    ]

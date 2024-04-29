from typing import Optional, Literal

from pydantic import BaseModel


class OpenAIGeneratorConfig(BaseModel):
    model_provider: Literal["OpenAI"]
    model: str


class AlephAlphaGeneratorConfig(BaseModel):
    model_provider: Literal["AlephAlpha"]
    model: str


class HuggingFaceGeneratorConfig(BaseModel):
    model_provider: Literal["HuggingFace"]
    model: str
    model_type: str
    tokenizer: str
    use_fast: Optional[bool] = None
    change_pad_token: Optional[bool] = None
    adapter: Optional[str] = None
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    use_accelerate: Optional[bool] = None


class SentenceTransformersGeneratorConfig(BaseModel):
    model_provider: Literal["SentenceTransformers"]
    model: str
    device: str
    cache_dir: str = None

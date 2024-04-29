from typing import Literal, Optional

from pydantic import BaseModel


class HuggingFaceVectorizerConfig(BaseModel):
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

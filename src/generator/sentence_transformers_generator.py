from typing import List

from sentence_transformers import SentenceTransformer

from src.generator.generator import Generator


class SentenceTransformersGenerator(Generator):
    def __init__(self, model: str, device: str = None, cache_dir: str = None):
        self.__model_name = model
        self.__device = device
        self.__cache_dir = cache_dir
        self.__model = self.__load_model()

    def vectorize(self, texts: List[str]) -> List[List[float]]:
        return self.__model.encode(
            sentences=texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def __load_model(self) -> SentenceTransformer:
        return SentenceTransformer(
            model_name_or_path=self.__model_name,
            device=self.__device,
            cache_folder=self.__cache_dir,
        )

    def generate(
        self,
        texts: List[str],
        max_new_tokens: int = 10,
        split_lines: bool = True,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0,
    ) -> List[str]:
        raise NotImplementedError

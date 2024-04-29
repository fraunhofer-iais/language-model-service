from abc import ABC, abstractmethod
from typing import List


class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        texts: List[str],
        max_new_tokens: int = 10,
        split_lines: bool = True,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0,
    ) -> List[str]:
        ...

    @abstractmethod
    def vectorize(self, texts: List[str]) -> List[List[float]]:
        ...

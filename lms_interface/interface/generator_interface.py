from abc import ABC, abstractmethod
from typing import List, Dict

from requests import Session


class GeneratorAPI(ABC):
    @abstractmethod
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
        ...

    @abstractmethod
    def available_models(
        self,
    ) -> Dict[str, Dict]:
        ...

    @abstractmethod
    def vectorize(
        self,
        model_name: str,
        texts: List[str],
    ) -> List[List[float]]:
        ...


class GeneratorRestDelegate(GeneratorAPI):
    def __init__(self, session: Session, endpoint: str):
        self.__session = session
        self.__endpoint = endpoint

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
        response = self.__session.post(
            url=f"{self.__endpoint}/generate",
            params={
                "model_name": model_name,
                "max_new_tokens": max_new_tokens,
                "split_lines": split_lines,
                "temperature": temperature,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            },
            json=texts,
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise ValueError(f"Generator did not work!")

    def vectorize(
        self,
        model_name: str,
        texts: List[str],
    ) -> List[List[float]]:
        response = self.__session.post(
            url=f"{self.__endpoint}/vectorize",
            params={
                "model_name": model_name,
            },
            json=texts,
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise ValueError(f"Vectorizer did not work!")

    def available_models(
        self,
    ) -> Dict[str, Dict]:
        response = self.__session.get(
            url=f"{self.__endpoint}/available_generator_models"
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise ValueError(f"Available generator model endpoint did not work!")

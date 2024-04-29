from abc import ABC, abstractmethod
from typing import List

from requests import Session


class VectorizerAPI(ABC):
    @abstractmethod
    def vectorize(self, model_name: str, texts: List[str]) -> List[List[float]]:
        ...


class VectorizerRestDelegate(VectorizerAPI):
    def __init__(self, session: Session, endpoint: str):
        self.__session = session
        self.__endpoint = endpoint

    def vectorize(self, model_name: str, texts: List[str]) -> List[List[float]]:
        response = self.__session.post(
            url=f"{self.__endpoint}/vectorize",
            params={"model_name": model_name},
            json=texts,
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise ValueError(f"Vectorize did not work!")

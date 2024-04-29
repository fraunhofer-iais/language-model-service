import os
from typing import List, Dict, Any, Optional
from aleph_alpha_client import Client, Prompt, CompletionRequest

from src.generator.generator import Generator


class AlephAlphaGenerator(Generator):
    def __init__(self, model: str = "luminous-extended"):
        self.__model_name = model
        self.__client = self.__get_client()

    def generate(
        self,
        texts: List[str],
        max_new_tokens: int = 10,
        split_lines: bool = True,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0,
    ) -> List[str]:
        responses = []
        for text in texts:
            params = self.__create_prompt(
                text=text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            request = CompletionRequest(**params)
            response = self.__client.complete(request=request, model=self.__model_name)
            completion = response.completions[0].completion
            responses.append(completion)
        return responses

    def __get_client(self) -> Optional[Client]:
        try:
            return Client(token=os.getenv("ALEPH_ALPHA_TOKEN"))
        except TypeError as e:
            raise Exception(
                "Please set your Aleph Alpha token. Use the variable "
                "ALEPH_ALPHA_TOKEN."
            )

    def __create_prompt(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> Dict[str, Any]:
        params = {
            "prompt": Prompt.from_text(text),
            "maximum_tokens": max_new_tokens,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
            # "stop_sequences": [".", ",", "?", "!"],
        }
        return params

    def vectorize(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

from typing import List

from openai import OpenAI

from src.generator.generator import Generator


class OpenAIGenerator(Generator):
    def __init__(self, model: str = "gpt-3.5-turbo"):
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
            response = self.__get_response(
                text=text,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            responses.append(response)

        return responses

    def __get_client(self):
        return OpenAI()

    def __get_response(
        self,
        text: str,
        temperature: float,
        max_new_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        response = self.__client.chat.completions.create(
            model=self.__model_name,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        response = response.choices[0].message.content
        return response

    def vectorize(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

from typing import List

import pytest

from src.generator.aleph_alpha_generator import AlephAlphaGenerator
from src.generator.openai_generator import OpenAIGenerator


@pytest.fixture
def texts() -> List[str]:
    return ["Hi", "How are you?"]


def test_openai_generator(texts):
    generator = OpenAIGenerator(model="gpt-3.5-turbo")
    responses = generator.generate(texts=texts, max_new_tokens=10)
    assert len(responses) == len(texts)

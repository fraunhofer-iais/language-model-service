from typing import List

import pytest

from src.generator.aleph_alpha_generator import AlephAlphaGenerator


@pytest.fixture
def texts() -> List[str]:
    return ["Hi", "How are you?"]


def test_aleph_alpha_generator(texts):
    generator = AlephAlphaGenerator(model="luminous-extended")
    responses = generator.generate(texts=texts, max_new_tokens=10)
    assert len(responses) == len(texts)

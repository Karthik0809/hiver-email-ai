import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.evaluate import extract_json, reference_similarity


def test_extract_json_plain():
    text = '{"relevance": {"score": 4, "why": "ok"}}'
    result = extract_json(text)
    assert result["relevance"]["score"] == 4


def test_extract_json_with_surrounding_prose():
    text = 'Sure, here is my evaluation:\n{"relevance": {"score": 3, "why": "fine"}}\nHope that helps!'
    result = extract_json(text)
    assert result["relevance"]["score"] == 3


def test_extract_json_with_code_fence():
    text = '```json\n{"relevance": {"score": 5, "why": "great"}}\n```'
    result = extract_json(text)
    assert result["relevance"]["score"] == 5


def test_extract_json_raises_on_no_json():
    import pytest
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_reference_similarity_identical_text_is_high():
    sim = reference_similarity("thank you for your order", ["thank you for your order"])
    assert sim > 0.9


def test_reference_similarity_returns_none_for_no_references():
    assert reference_similarity("some reply text", []) is None

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.rubric import DIMENSIONS, WEIGHTS, composite_score


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_weights_cover_all_dimensions():
    assert set(WEIGHTS.keys()) == set(DIMENSIONS)


def test_composite_score_all_max():
    dims = {d: {"score": 5, "why": ""} for d in DIMENSIONS}
    assert composite_score(dims) == 5.0


def test_composite_score_all_min():
    dims = {d: {"score": 0, "why": ""} for d in DIMENSIONS}
    assert composite_score(dims) == 0.0


def test_composite_score_weights_relevance_more_than_conciseness():
    # Scoring max (5) on relevance alone (weight 0.25) with everything else
    # at 0 should outscore scoring max on conciseness alone (weight 0.10)
    # with everything else at 0 - directly demonstrates relevance is
    # weighted more heavily than a lighter dimension, as documented in
    # rubric.py.
    high_relevance_only = {d: {"score": 0, "why": ""} for d in DIMENSIONS}
    high_relevance_only["relevance"]["score"] = 5

    high_conciseness_only = {d: {"score": 0, "why": ""} for d in DIMENSIONS}
    high_conciseness_only["conciseness"]["score"] = 5

    assert composite_score(high_relevance_only) > composite_score(high_conciseness_only)


def test_composite_score_heavy_dims_outweigh_all_light_dims_combined():
    # Being perfect on JUST relevance+correctness (weight 0.5 combined)
    # should score at least as high as being perfect on every lighter
    # dimension combined (also weight 0.5 combined) - they're designed to
    # be equal in aggregate, which this test locks in as a specification.
    high_on_heavy = {d: {"score": 0, "why": ""} for d in DIMENSIONS}
    high_on_heavy["relevance"]["score"] = 5
    high_on_heavy["correctness"]["score"] = 5

    high_on_light = {d: {"score": 5, "why": ""} for d in DIMENSIONS}
    high_on_light["relevance"]["score"] = 0
    high_on_light["correctness"]["score"] = 0

    assert composite_score(high_on_heavy) == composite_score(high_on_light)

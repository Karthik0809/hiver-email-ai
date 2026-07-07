import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.retrieval import load_dataset, EmailRetriever


def test_retriever_returns_k_results():
    dataset = load_dataset()
    retriever = EmailRetriever(dataset)
    results = retriever.top_k("Refund request", "I want my money back for a damaged item", k=3)
    assert len(results) == 3
    for row, score in results:
        assert "id" in row
        assert 0.0 <= score <= 1.0 + 1e-9


def test_retriever_ranks_relevant_examples_higher():
    dataset = load_dataset()
    retriever = EmailRetriever(dataset)
    # A query about a duplicate/double charge should surface billing-related
    # rows above completely unrelated categories at the top of the ranking.
    results = retriever.top_k("Charged twice", "I was billed twice for the same order, please refund the extra charge", k=3)
    top_categories = [row["category"] for row, _ in results]
    assert "billing_dispute" in top_categories or "refund_request" in top_categories


def test_retriever_scores_are_descending():
    dataset = load_dataset()
    retriever = EmailRetriever(dataset)
    results = retriever.top_k("Shipping delay", "My package tracking hasn't updated in a week", k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)

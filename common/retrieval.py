"""
Lightweight retrieval over the past-email dataset, used to ground the
generator in real examples (retrieval-augmented few-shot prompting).

Uses TF-IDF + cosine similarity rather than a neural embedding model on
purpose: it's fast, has zero external dependencies beyond scikit-learn,
needs no API key/download, and is easy to reason about — a sensible
trade-off for a small (~40-row) dataset. For a production system with a
much larger email history, swapping in a real embedding model (e.g. an
Ollama embedding model or OpenAI embeddings) would be the natural upgrade;
the interface below is written so that swap only touches this file.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "dataset", "emails.json")
    with open(path) as f:
        return json.load(f)


class EmailRetriever:
    def __init__(self, dataset):
        self.dataset = dataset
        corpus = [row["subject"] + "\n" + row["incoming_email"] for row in dataset]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def top_k(self, query_subject, query_body, k=3):
        query_vec = self.vectorizer.transform([query_subject + "\n" + query_body])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return [(self.dataset[i], float(sims[i])) for i in ranked[:k]]

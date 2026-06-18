"""
embeddings.py
----------------
A lightweight, fully-offline "embedding" layer using TF-IDF + character n-grams.

Why TF-IDF instead of a neural embedding model?
- The task explicitly allows "embeddings or lightweight models".
- This sandbox/deployment target has no access to model-weight hosts
  (e.g. huggingface.co) at runtime, so a downloaded sentence-transformer
  is not a safe dependency for a "works out of the box" submission.
- TF-IDF + cosine similarity is a well-understood, fast, dependency-light
  way to get semantic-ish similarity for topic segmentation & retrieval,
  and it's exactly how production RAG systems bootstrap before swapping
  in a hosted embedding model.

If you DO have an embedding API (OpenAI, Cohere, Grok, etc.) available,
swap `TfidfEmbedder` for your own class that implements the same
`fit(texts)` / `transform(texts) -> np.ndarray` interface and nothing
else in this codebase needs to change.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfEmbedder:
    def __init__(self, max_features=20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts):
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before transform()")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        mat = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return mat


def cosine_sim_matrix(a, b=None):
    """Cosine similarity between rows of sparse/dense matrix a (and b, or a vs itself)."""
    if b is None:
        b = a
    return cosine_similarity(a, b)

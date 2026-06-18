"""
retriever.py
----------------
RAG retrieval layer. Maintains TWO indices, as required by the task:

  1. TOPIC index   - one TF-IDF vector per topic-checkpoint summary
  2. CHUNK index    - one TF-IDF vector per small sliding-window chunk of
                       raw messages (so we can point back to *specific*
                       lines, not just a paraphrase)

Both indices share a single TfidfVectorizer (fit on the union of all topic
summaries + chunk texts) so that cosine similarity scores from the two
indices are on a comparable scale and a query only needs to be vectorized
once.

retrieve(query, top_k_topics, top_k_chunks) returns the most relevant
items from BOTH indices -- the caller (chatbot.py) then combines them
into the final answer/context.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .summarizer import QA_STOP_WORDS

CHUNK_SIZE = 6
CHUNK_STRIDE = 3


def build_chunks(messages, chunk_size=CHUNK_SIZE, stride=CHUNK_STRIDE):
    chunks = []
    for i in range(0, len(messages), stride):
        block = messages[i:i + chunk_size]
        if not block:
            continue
        chunks.append({
            "start": block[0]["global_id"],
            "end": block[-1]["global_id"] + 1,
            "text": " ".join(m["text"] for m in block),
            "day_range": (block[0]["day"], block[-1]["day"]),
        })
        if i + chunk_size >= len(messages):
            break
    return chunks


class RagRetriever:
    def __init__(self, topic_checkpoints, message_checkpoints, chunks):
        self.topic_checkpoints = topic_checkpoints
        self.message_checkpoints = message_checkpoints
        self.chunks = chunks

        topic_texts = [tc["summary"] for tc in topic_checkpoints]
        chunk_texts = [c["text"] for c in chunks]
        ckpt_texts = [mc["summary"] for mc in message_checkpoints]

        corpus = topic_texts + chunk_texts + ckpt_texts
        self.vectorizer = TfidfVectorizer(stop_words=QA_STOP_WORDS, ngram_range=(1, 2))
        self.vectorizer.fit(corpus if corpus else ["empty"])

        self.topic_matrix = self.vectorizer.transform(topic_texts) if topic_texts else None
        self.chunk_matrix = self.vectorizer.transform(chunk_texts) if chunk_texts else None
        self.ckpt_matrix = self.vectorizer.transform(ckpt_texts) if ckpt_texts else None

    def _top_k(self, query_vec, matrix, items, k):
        if matrix is None or matrix.shape[0] == 0:
            return []
        sims = cosine_similarity(query_vec, matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            {**items[i], "score": float(sims[i])}
            for i in top_idx if sims[i] > 0
        ]

    def retrieve(self, query: str, top_k_topics=2, top_k_chunks=4, top_k_checkpoints=1):
        qvec = self.vectorizer.transform([query])
        topics = self._top_k(qvec, self.topic_matrix, self.topic_checkpoints, top_k_topics)
        chunks = self._top_k(qvec, self.chunk_matrix, self.chunks, top_k_chunks)
        ckpts = self._top_k(qvec, self.ckpt_matrix, self.message_checkpoints, top_k_checkpoints)
        return {"topics": topics, "chunks": chunks, "checkpoints": ckpts}

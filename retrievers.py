"""
Retrievers under test. All share one interface:

    search(query, top_k) -> List[Hit]   (sorted best-first)

Each Hit carries two numbers:
  score       - what the retriever ranks by
  confidence  - what the scope/ambiguity gates threshold on

For most retrievers they're the same. They differ for hybrid (RRF) retrieval,
where the fused score is rank-based and says nothing about absolute match
quality, so confidence is borrowed from the dense retriever's cosine score.
"""
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    confidence: float


class Retriever:
    name = "base"
    llm_calls = 0

    def search(self, query: str, top_k: int = 5) -> List[Hit]:
        raise NotImplementedError


# ---------------------------------------------------------------- lexical ---

class TfidfRetriever(Retriever):
    """Control condition: identical settings to the original harness."""
    name = "tfidf"

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.index_text for c in chunks])

    def search(self, query, top_k=5):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i]), float(scores[i])) for i in order]


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25Retriever(Retriever):
    """BM25 (Okapi): the standard lexical baseline in production search."""
    name = "bm25"

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.index_text) for c in chunks], k1=k1, b=b)

    def search(self, query, top_k=5):
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i]), float(scores[i])) for i in order]


# ------------------------------------------------------------------ dense ---

class DenseRetriever(Retriever):
    """Embedding retriever. Defaults to FastEmbed + BAAI/bge-small-en-v1.5,
    the same embedding stack as the production HR-Policy-QA-Bot.
    `embedder` can be injected (used by the offline smoke test)."""
    name = "dense"

    def __init__(self, chunks: List[Chunk], model: str = "BAAI/bge-small-en-v1.5",
                 embedder=None):
        self.chunks = chunks
        self.model_name = model
        if embedder is None:
            from fastembed import TextEmbedding
            embedder = TextEmbedding(model)
        self.embedder = embedder
        self.matrix = self._norm(np.array(list(self.embedder.passage_embed(
            [c.index_text for c in chunks]))))

    @staticmethod
    def _norm(m):
        return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)

    def embed_query(self, text: str):
        return self._norm(np.array(list(self.embedder.query_embed([text])))[0])

    def embed_passage(self, text: str):
        return self._norm(np.array(list(self.embedder.passage_embed([text])))[0])

    def search_vector(self, vec, top_k=5):
        scores = self.matrix @ vec
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i]), float(scores[i])) for i in order]

    def search(self, query, top_k=5):
        return self.search_vector(self.embed_query(query), top_k)


# ----------------------------------------------------------------- hybrid ---

class HybridRRFRetriever(Retriever):
    """Reciprocal Rank Fusion of several retrievers (k=60, the standard
    constant). Confidence comes from `confidence_from` (default: dense)."""
    name = "hybrid_rrf"

    def __init__(self, retrievers: Dict[str, Retriever], k: int = 60,
                 confidence_from: str = "dense", candidates: int = 20):
        self.retrievers = retrievers
        self.k = k
        self.confidence_from = confidence_from
        self.candidates = candidates

    def search(self, query, top_k=5):
        fused: Dict[str, float] = {}
        by_id: Dict[str, Chunk] = {}
        conf: Dict[str, float] = {}
        for name, r in self.retrievers.items():
            hits = r.search(query, top_k=self.candidates)
            for rank, h in enumerate(hits, start=1):
                cid = h.chunk.chunk_id
                by_id[cid] = h.chunk
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.k + rank)
                if name == self.confidence_from:
                    conf[cid] = h.confidence
        floor = min(conf.values()) if conf else 0.0
        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [Hit(by_id[c], fused[c], conf.get(c, floor)) for c in ranked]


# --------------------------------------------------------------- reranker ---

class RerankRetriever(Retriever):
    """Retrieve `candidates` from a first-stage retriever, re-score each
    (query, chunk) pair with a cross-encoder. Confidence = cross-encoder score."""
    name = "rerank"

    def __init__(self, base: Retriever, model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
                 candidates: int = 20, encoder=None):
        self.base = base
        self.candidates = candidates
        if encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            encoder = TextCrossEncoder(model)
        self.encoder = encoder

    def search(self, query, top_k=5):
        first = self.base.search(query, top_k=self.candidates)
        scores = list(self.encoder.rerank(query, [h.chunk.index_text for h in first]))
        rescored = sorted(zip(first, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [Hit(h.chunk, float(s), float(s)) for h, s in rescored]


# ------------------------------------------------------------------- HyDE ---

HYDE_PROMPT = (
    "Write a short passage (3-5 sentences) from a company HR policy document "
    "that would answer the employee question below. Write it as policy text, "
    "not as a reply to the employee.\n\nQuestion: {q}"
)


class HyDERetriever(Retriever):
    """Hypothetical Document Embeddings (Guidebook p.131). The LLM writes a
    fake policy passage; we embed THAT and search passage-to-passage."""
    name = "hyde"

    def __init__(self, dense: DenseRetriever, llm, blend_query: bool = True):
        self.dense = dense
        self.llm = llm
        self.blend_query = blend_query
        self.hypotheticals: Dict[str, str] = {}

    def search(self, query, top_k=5):
        hypo = self.llm.complete(HYDE_PROMPT.format(q=query))
        self.hypotheticals[query] = hypo
        vec = self.dense.embed_passage(hypo)
        if self.blend_query:  # common variant: average with the raw query vector
            vec = DenseRetriever._norm(vec + self.dense.embed_query(query))
        return self.dense.search_vector(vec, top_k)


# ---------------------------------------------------------------- factory ---

def build_retriever(spec: dict, chunks: List[Chunk], llm=None,
                    embedder=None, encoder=None) -> Retriever:
    kind = spec["type"]
    if kind == "tfidf":
        return TfidfRetriever(chunks)
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind == "dense":
        return DenseRetriever(chunks, model=spec.get("model", "BAAI/bge-small-en-v1.5"),
                              embedder=embedder)
    if kind == "hybrid_rrf":
        parts = {p["type"]: build_retriever(p, chunks, llm, embedder, encoder)
                 for p in spec["components"]}
        return HybridRRFRetriever(parts, confidence_from=spec.get("confidence_from", "dense"))
    if kind == "rerank":
        base = build_retriever(spec["base"], chunks, llm, embedder, encoder)
        return RerankRetriever(base, model=spec.get("model", "Xenova/ms-marco-MiniLM-L-6-v2"),
                               encoder=encoder)
    if kind == "hyde":
        dense = build_retriever({"type": "dense", **spec.get("dense", {})}, chunks,
                                llm, embedder, encoder)
        if llm is None:
            raise RuntimeError("HyDE needs an LLM client")
        return HyDERetriever(dense, llm, blend_query=spec.get("blend_query", True))
    raise ValueError(f"Unknown retriever type: {kind}")


def timed_search(retriever: Retriever, query: str, top_k: int = 5):
    t0 = time.perf_counter()
    hits = retriever.search(query, top_k)
    return hits, (time.perf_counter() - t0) * 1000.0

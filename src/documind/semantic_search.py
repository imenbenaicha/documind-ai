
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text):
    """Convert text into lowercase tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_index(chunks):
    """Build semantic and BM25 indexes for all document chunks."""

    if not chunks:
        raise ValueError("Cannot index an empty document collection.")

    texts = [chunk["text"] for chunk in chunks]

    if not any(text.strip() for text in texts):
        raise ValueError("No usable text was found in the documents.")

    model = SentenceTransformer(MODEL_NAME)

    # Semantic index
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # BM25 keyword index
    tokenized_chunks = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_chunks)

    return model, embeddings, bm25


def search(question, chunks, model, embeddings, top_k=5):
    """Retrieve chunks using semantic similarity."""

    if not question.strip():
        return []

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    ).flatten()

    indices = scores.argsort()[::-1][:top_k]

    return [
        {**chunks[i], "score": float(scores[i])}
        for i in indices
    ]


def hybrid_search(
    question,
    chunks,
    model,
    embeddings,
    bm25,
    top_k=5,
    semantic_weight=0.5,
    bm25_weight=0.5,
    rrf_k=60,
):
    """
    Combine semantic and keyword rankings using
    Reciprocal Rank Fusion (RRF).
    """

    if not question.strip():
        return []

    if not chunks:
        return []

    # 1. Semantic ranking
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        embeddings,
    ).flatten()

    semantic_indices = semantic_scores.argsort()[::-1]

    # 2. BM25 keyword ranking
    tokenized_question = tokenize(question)
    bm25_scores = bm25.get_scores(tokenized_question)

    bm25_indices = bm25_scores.argsort()[::-1]

    # 3. Reciprocal Rank Fusion
    fused_scores = {}

    for rank, index in enumerate(semantic_indices, start=1):
        fused_scores[index] = (
            fused_scores.get(index, 0)
            + semantic_weight / (rrf_k + rank)
        )

    for rank, index in enumerate(bm25_indices, start=1):
        fused_scores[index] = (
            fused_scores.get(index, 0)
            + bm25_weight / (rrf_k + rank)
        )

    # 4. Return the highest-ranked chunks
    ranked_indices = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )[:top_k]

    return [
        {
            **chunks[index],
            "score": float(fused_scores[index]),
        }
        for index in ranked_indices
    ]


def adaptive_search(
    question,
    chunks,
    model,
    embeddings,
    top_k=5,
    relevance_threshold=0.20,
):
    """Retrieve semantically relevant chunks above a threshold."""

    candidates = search(
        question,
        chunks,
        model,
        embeddings,
        top_k=min(top_k, len(chunks)),
    )

    return [
        chunk
        for chunk in candidates
        if chunk["score"] >= relevance_threshold
    ]

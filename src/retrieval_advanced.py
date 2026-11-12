"""Hybrid RAG (BM25 + dense via RRF) and optional cross-encoder reranking.

Run from the project root, after the FATF corpus and Chroma index are built:
    python -m src.retrieval_advanced search "What are TBML payment red flags?" --method hybrid
    python -m src.retrieval_advanced search "What are TBML payment red flags?" --method rerank

The tools retrieve *general AML guidance*, not proof of wrongdoing by an account.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from src.rag import bm25_search, dense_search, load_corpus

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def hybrid_search(
    question: str,
    corpus: list[dict[str, Any]] | None = None,
    *,
    k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Return RRF-fused BM25 + dense hits, ranked highest first.

    Each retriever contributes 1/(rrf_k + 1-based rank), irrespective of
    its raw score scale. Chunk IDs are deduplicated across the two lists.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if k < 1 or candidate_k < k or rrf_k < 1:
        raise ValueError("Require k >= 1, candidate_k >= k, and rrf_k >= 1")

    if corpus is None:
        corpus = load_corpus()
    if not corpus:
        return []

    limit = min(candidate_k, len(corpus))
    lexical = bm25_search(question, corpus, k=limit)
    semantic = dense_search(question, k=limit)

    source_by_id = {c["chunk_id"]: c for c in corpus}
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for method, results in (("bm25", lexical), ("dense", semantic)):
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id not in source_by_id:
                raise ValueError(f"Retriever returned unknown chunk ID: {chunk_id}")
            if method in ranks[chunk_id]:
                continue
            ranks[chunk_id][method] = rank
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [
        {
            **source_by_id[cid],
            "method": "hybrid",
            "rrf_score": scores[cid],
            "bm25_rank": ranks[cid].get("bm25"),
            "dense_rank": ranks[cid].get("dense"),
        }
        for cid in ranked[:k]
    ]


def reranked_search(
    question: str,
    corpus: list[dict[str, Any]] | None = None,
    *,
    k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
    model_name: str = DEFAULT_RERANKER,
) -> list[dict[str, Any]]:
    """Rerank RRF candidates using a question-passage cross-encoder.

    Cross-encoder scores are ranking scores, not calibrated probabilities.
    """
    if k < 1 or candidate_k < k:
        raise ValueError("Require k >= 1 and candidate_k >= k")
    candidates = hybrid_search(
        question, corpus, k=candidate_k, candidate_k=candidate_k, rrf_k=rrf_k
    )
    if not candidates:
        return []

    from sentence_transformers import CrossEncoder  # lazy import

    model = CrossEncoder(model_name)
    pairs = [(question, candidate["text"]) for candidate in candidates]
    predicted = model.predict(pairs, batch_size=8, show_progress_bar=False)
    scores = [float(value) for value in predicted]
    if len(scores) != len(candidates):
        raise RuntimeError("Cross-encoder returned an unexpected number of scores")

    rescored = [
        {**candidate, "method": "hybrid_rerank", "rerank_score": score,
         "reranker_model": model_name}
        for candidate, score in zip(candidates, scores)
    ]
    return sorted(
        rescored, key=lambda item: (-item["rerank_score"], item["chunk_id"])
    )[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced FATF retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Search with RRF or RRF + reranking")
    search.add_argument("question")
    search.add_argument("--method", choices=["hybrid", "rerank"], default="hybrid")
    search.add_argument("-k", type=int, default=5, help="Final top-k passages")
    search.add_argument("--candidate-k", type=int, default=20)
    search.add_argument("--rrf-k", type=int, default=60)
    search.add_argument("--reranker", default=DEFAULT_RERANKER)
    args = parser.parse_args()

    if args.method == "hybrid":
        hits = hybrid_search(
            args.question, k=args.k, candidate_k=args.candidate_k, rrf_k=args.rrf_k
        )
    else:
        hits = reranked_search(
            args.question, k=args.k, candidate_k=args.candidate_k,
            rrf_k=args.rrf_k, model_name=args.reranker,
        )

    for rank, hit in enumerate(hits, start=1):
        score_label = (
            f"rerank_score={hit['rerank_score']:.4f}; RRF={hit['rrf_score']:.4f}"
            if args.method == "rerank" else f"RRF={hit['rrf_score']:.4f}"
        )
        print(
            f"{rank}. {hit['chunk_id']} | {hit['title']} | "
            f"PDF page {hit['page_number']} | {score_label}"
        )
        print(f"   {hit['source_url']}")
        print("   " + hit["text"][:280].replace("\n", " ") + "\n")


if __name__ == "__main__":
    main()

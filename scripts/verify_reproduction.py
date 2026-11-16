"""Validate the generated fresh-clone demo, SQL analysis, alerts and optional FATF RAG.

Usage:
    python -m scripts.verify_reproduction
    python -m scripts.verify_reproduction --strict-rag
No network request or OpenAI call is made by this verifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOCAL = "D00::CENTER"


def verify(strict_rag: bool = False) -> bool:
    from app.data_config import is_generated_demo, resolve_data_paths
    from src.baseline import investigate
    from src.functions import FinancialTools
    from app.intelligence_data import account_bounds, self_transfer_summary
    from app.intelligence_alerts import repeated_payment_alerts, concentration_alerts

    subset, manifest = resolve_data_paths(ROOT, {"FINCRIME_DATASET": "demo"})
    if not (subset.is_file() and manifest.is_file()):
        raise FileNotFoundError("No demo data. Run python -m scripts.bootstrap_demo")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if "NOT IBM" not in data.get("source", ""):
        raise AssertionError("Demo provenance marker missing")
    if not is_generated_demo(subset):
        raise AssertionError("Wrong dataset selected")

    with FinancialTools(subset) as fin:
        if not fin.account_exists(FOCAL):
            raise AssertionError("Missing demonstration focal account")
        tx = fin.get_account_transactions(FOCAL, limit=500)
        if not tx["complete"]:
            raise AssertionError("Truncated focal account data")
        h1 = fin.explore_transaction_network(FOCAL, hops=1, max_edges=3000)
        h2 = fin.explore_transaction_network(FOCAL, hops=2, max_edges=3000)
        if not (h1.get("complete") and h2.get("complete") and
                h2["node_count"] > h1["node_count"]):
            raise AssertionError(f"Two-hop demo not distinct: h1={h1} h2={h2}")
        report = investigate(FOCAL, fin)
        if report.get("status") != "completed":
            raise AssertionError("Deterministic investigation failed")
        bounds = account_bounds(fin, FOCAL)
        if bounds is None:
            raise AssertionError("Account Intelligence date bounds missing")
        start, end = bounds
        if not self_transfer_summary(fin, FOCAL, start, end, None):
            raise AssertionError("Expected self-transfer fixture absent")
        repeats = repeated_payment_alerts(fin, FOCAL, start, end, None)
        if not any(r["direction"] == "Outgoing" and r["transactions"] >= 10
                   and round(float(r["unit_amount"]), 2) == 1250.0 for r in repeats):
            raise AssertionError("Expected repeated-payment alert absent")
        concentrations = concentration_alerts(fin, FOCAL, start, end, None)
        if not concentrations:
            raise AssertionError("Expected concentration alert absent")
    print(f"PASS: {data['n_transactions']} fictional transactions; complete SQL report, "
          f"intelligence views and screening alerts; depth 1 = {h1['node_count']}, "
          f"depth 2 = {h2['node_count']} nodes")

    corpus_path = ROOT / "data/processed/fatf_chunks.jsonl"
    index_path = ROOT / "data/processed/fatf_chroma/model_name.txt"
    have_corpus = corpus_path.is_file()
    have_index = index_path.is_file()
    if strict_rag:
        if not (have_corpus and have_index):
            raise FileNotFoundError(
                "FATF corpus or dense index is missing. Run python -m scripts.bootstrap_demo"
            )
        from src.rag import bm25_search, dense_search, load_corpus
        corpus = load_corpus(corpus_path)
        if not corpus:
            raise AssertionError("FATF corpus is empty")
        question = "What are indicators of trade-based money laundering?"
        if not bm25_search(question, corpus, k=3):
            raise AssertionError("BM25 returned no FATF guidance")
        if not dense_search(question, k=3, index_dir=index_path.parent):
            raise AssertionError("Dense RAG returned no FATF guidance")
        print(f"PASS: {len(corpus)} indexed FATF chunks; BM25 and dense queries returned hits")
    else:
        print("FATF corpus:", "ready" if have_corpus else "missing (normal bootstrap builds it)")
        print("FATF dense index:", "ready" if have_index else "missing (normal bootstrap builds it)")
    print("An agentic LLM run additionally requires a personal OPENAI_API_KEY and enabled model.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-rag", action="store_true")
    args = parser.parse_args()
    verify(args.strict_rag)


if __name__ == "__main__":
    main()

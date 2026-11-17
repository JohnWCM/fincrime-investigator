"""Reproducible, independently generated demo data + official FATF RAG setup.

Run: python -m scripts.bootstrap_demo [--skip-fatf] [--skip-dense] [--force]
Creates NO original IBM observations or benchmark labels. Never alters IBM data.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
FOCAL = "D00::CENTER"
HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]


def make_rows(seed: int = 42) -> list[list[str]]:
    """Deterministic fictional transactions designed to exercise app views."""
    rng = random.Random(seed)
    t0 = datetime(2022, 9, 1, 9, 0)
    rows: list[list[str]] = []

    def tx(src: str, dst: str, amt: float, idx: int, *, currency="EUR",
           receive_currency: str | None = None, fmt="ACH") -> None:
        bank_a, acct_a = src.split("::", 1)
        bank_b, acct_b = dst.split("::", 1)
        rows.append([(t0 + timedelta(minutes=idx * 13)).strftime("%Y/%m/%d %H:%M"),
                     bank_a, acct_a, bank_b, acct_b, f"{amt:.2f}",
                     receive_currency or currency, f"{amt:.2f}", currency, fmt, "0"])

    first_hop = [f"D0{i}::BRANCH{i}" for i in range(1, 5)]
    secondary = [f"D1{i}::LEAF{i}" for i in range(1, 17)]
    for i in range(10):
        tx(FOCAL, first_hop[0], 1250.00, i, fmt="Wire")  # repeated payment signal
    for i in range(10, 22):
        tx(first_hop[1], FOCAL, float(950 + i * 20), i)
    for i in range(22, 32):
        tx(FOCAL, first_hop[2], float(500 + i * 15), i, currency="USD")
    for i in range(32, 40):
        tx(FOCAL, FOCAL, 110 + i, i, currency="USD",
           receive_currency="EUR" if i % 2 else "USD", fmt="Reinvestment")
    for j, first in enumerate(first_hop):
        for k in range(4):
            leaf = secondary[j * 4 + k]
            for r in range(3):
                idx = 40 + (j * 4 + k) * 3 + r
                if r % 2:
                    tx(leaf, first, float(250 + rng.randint(0, 1500)), idx)
                else:
                    tx(first, leaf, float(250 + rng.randint(0, 1500)), idx)
    # Add other connected accounts with predictable but varied traffic.
    for i in range(80):
        a = secondary[i % 16]
        b = secondary[(i + 3 + i // 16) % 16]
        tx(a, b, float(100 + rng.randint(0, 2500)), 88 + i,
           currency="USD" if i % 5 == 0 else "EUR")
    return rows


def write_demo_csv(path: Path, seed: int = 42) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = make_rows(seed)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    return len(rows)


def build_demo(force: bool = False) -> dict:
    """Only touches /data/demo. Labels are kept separate from app datasets."""
    raw = DEMO / "raw" / "HI-Small_Trans.csv"
    processed = DEMO / "processed"
    eval_dir = DEMO / "evaluation"
    subset = processed / "investigation_subset.parquet"
    manifest = processed / "subset_manifest.json"
    if any(p.exists() for p in (subset, manifest, processed / "transactions.parquet")) and not force:
        if subset.is_file() and manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8"))
        raise FileExistsError("Partial demo data exists: review data/demo or rerun with --force")
    if force and DEMO.exists():
        shutil.rmtree(DEMO)  # only generated demonstration directory
    rows = write_demo_csv(raw)
    from src.ingest import ingest
    ingest(raw, processed, eval_dir)
    shutil.copy2(processed / "transactions.parquet", subset)
    cases = [FOCAL] + [f"D0{i}::BRANCH{i}" for i in range(1, 5)] + [
        f"D1{i}::LEAF{i}" for i in range(1, 17)]
    public = {
        "source": "Locally generated fictional demo (NOT IBM AMLworld)",
        "subset_method": "complete generated demonstration graph; no transaction sampling",
        "case_selection_warning": "Artificial fixture; no real AML labels, prevalence or benchmark conclusions",
        "seed": 42, "n_transactions": rows, "n_cases": len(cases),
        "cases": [{"account_id": a} for a in cases],
    }
    manifest.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    return public


def build_fatf(with_dense: bool = True) -> None:
    from src.rag import build_corpus, build_dense, download_pdfs
    raw_dir = ROOT / "data" / "raw" / "fatf"
    processed = ROOT / "data" / "processed"
    download_pdfs(raw_dir)
    corpus = build_corpus(raw_dir, processed / "fatf_chunks.jsonl")
    print(f"FATF corpus: {len(corpus)} chunks")
    if with_dense:
        build_dense(corpus, processed / "fatf_chroma")
        print("Dense Chroma index ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate data/demo only")
    parser.add_argument("--skip-fatf", action="store_true", help="Offline data/graph only; RAG not ready")
    parser.add_argument("--skip-dense", action="store_true", help="BM25 corpus only; hybrid/dense not ready")
    args = parser.parse_args()
    data = build_demo(force=args.force)
    print(f"Generated demo: {data['n_transactions']} fictional transactions, {data['n_cases']} cases")
    if args.skip_fatf:
        print("FATF setup skipped. FATF Explorer/Agentic RAG require the official PDFs and index.")
    else:
        try:
            build_fatf(with_dense=not args.skip_dense)
        except Exception as exc:
            raise SystemExit(
                f"FATF setup could not finish: {type(exc).__name__}: {exc}\n"
                "Follow docs/FRESH_CLONE.md for official PDF downloads, then rerun."
            ) from exc
    print("Next: python -m scripts.verify_reproduction")
    print("Then: python -m streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

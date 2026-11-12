"""Review only the three held-out FATF questions BEFORE running the test benchmark.

From repo root:
  python scripts/annotate_test_retrieval.py          # print evidence to compare with local PDFs
  python scripts/annotate_test_retrieval.py --write  # explicitly approve each evidence set

This is NOT an automatic truth-labeler. You must independently inspect the PDF.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

GOLD = Path("data/evaluation/retrieval_gold.jsonl")
CORPUS = Path("data/processed/fatf_chunks.jsonl")

# Candidate IDs were selected from the previously reviewed candidate passages.
# They are PROPOSALS, not verified labels until manually matched against PDF text.
PROPOSED = {
    "RET-04": {
        "pdf": "data/raw/fatf/banking_rba_2014.pdf",
        "pages": "PDF page 23",
        "ids": ["fatf_banking_rba_2014-p023-c001-58978c51",
                "fatf_banking_rba_2014-p023-c003-58978c51"],
        "answer": ("Ongoing transaction monitoring checks consistency with the bank's knowledge "
                   "of a customer's activity and relationship, detects changes and unusual transactions, "
                   "and adjusts monitoring intensity according to assessed risk."),
    },
    "RET-08": {
        "pdf": "data/raw/fatf/tbml_indicators_2021.pdf",
        "pages": "PDF page 4",
        "ids": ["fatf_tbml_indicators_2021-p004-c001-cf761b4c"],
        "answer": ("One indicator on its own need not establish TBML suspicion or prove an offence; "
                   "it may warrant further monitoring and contextual examination."),
    },
    "RET-12": {
        "pdf": "data/raw/fatf/financial_investigations_2012.pdf",
        "pages": "PDF page 22",
        "ids": ["fatf_financial_investigations_2012-p022-c001-cb1475fd",
                "fatf_financial_investigations_2012-p022-c004-cb1475fd"],
        "answer": ("Financial investigators can cooperate with FIUs and other competent authorities "
                   "by exchanging intelligence, designating points of contact and documenting "
                   "procedures through MoUs or other formal information-sharing arrangements."),
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Prompt for manual PDF confirmation and update gold JSONL")
    args = parser.parse_args()
    if not GOLD.exists() or not CORPUS.exists():
        raise SystemExit("Missing gold/corpus JSONL; run ingestion and benchmark preparation first.")
    records = load_jsonl(GOLD)
    chunks = {r["chunk_id"]: r for r in load_jsonl(CORPUS)}
    by_id = {r["id"]: r for r in records}
    if len(records) != 12 or any(qid not in by_id or by_id[qid].get("split") != "test" for qid in PROPOSED):
        raise SystemExit("Unexpected gold file structure; aborting rather than overwrite.")
    for qid, cfg in PROPOSED.items():
        print("\n" + "="*85)
        print(qid, by_id[qid]["question"])
        print("COMPARE WITH ORIGINAL PDF:", cfg["pdf"], cfg["pages"])
        for cid in cfg["ids"]:
            if cid not in chunks:
                raise SystemExit(f"Missing from actual corpus: {cid}. Do not write annotations.")
            chunk = chunks[cid]
            if chunk["doc_id"] != by_id[qid]["doc_id"]:
                raise SystemExit(f"Unexpected source document for {cid}")
            print(f"\nSOURCE CHUNK: {cid}\n{chunk['text']}\n")
        print("PROPOSED ANSWER:", cfg["answer"])
        if args.write:
            confirmation = input("After checking the actual PDF, enter VERIFIED to approve this answer and chunk set: ")
            if confirmation.strip() != "VERIFIED":
                raise SystemExit("No changes saved. Review the PDF, then rerun if appropriate.")

    if args.write:
        backup = GOLD.with_name("retrieval_gold_before_test_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
        shutil.copy2(GOLD, backup)
        for qid, cfg in PROPOSED.items():
            row = by_id[qid]
            row["gold_chunk_ids"] = cfg["ids"]
            row["gold_answer"] = cfg["answer"]
            row["annotation_status"] = "verified"
            row["annotation_note"] = f"Manually compared source PDF ({cfg['pages']}) with local corpus before held-out scoring."
        GOLD.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        print("\nSaved", GOLD, "\nBackup:", backup)
        print("Do not tune retrieval methods based on the subsequent test metrics.")
    else:
        print("\nNo files modified. Add --write only after independent PDF review.")


if __name__ == "__main__":
    main()

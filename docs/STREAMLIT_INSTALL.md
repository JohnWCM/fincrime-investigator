# FinCrime Investigator — Streamlit upgrade

This is an **additive UI upgrade**, not a replacement for your Day 1–3 pipeline. It replaces only the starter `app/streamlit_app.py` and adds `app/ui_helpers.py`, a pure unit test, and a held-out annotation review script. The existing agent, tools, benchmark and source data are **not modified**.

## Files

- `app/streamlit_app.py` — five-view Streamlit app
- `app/ui_helpers.py` — plotting, scoring and file readers
- `tests/test_streamlit_ui_helpers.py` — regression tests for new pure helpers
- `scripts/annotate_test_retrieval.py` — assisted, manually confirmed test annotations

## Windows PowerShell install

From your project directory:

```powershell
cd "C:\Users\jjwei\final project Ironhack\fincrime-investigator"
Copy-Item "app\streamlit_app.py" "app\streamlit_app_before_upgrade.py"
# Extract the ZIP to the project root, preserving the app/, tests/, scripts/ and docs/ directories.
python -m pip install -r requirements.txt
python -m pip install "langchain-openai>=0.2,<2"  # only if it is not already installed
python -m pytest tests/test_streamlit_ui_helpers.py -q
python -m streamlit run app/streamlit_app.py
```

**Critical:** Do not run `pip install` unnecessarily if your current environment already has the project dependencies. `requirements.txt` is your existing repository file and should be kept as the source of truth; this upgrade does not overwrite it. If an import fails, install that missing package in the same environment used by `python -m streamlit`.

## The five views

1. Investigation: deterministic or agentic reports, verified DuckDB facts, patterns, ledger, references, downloadable JSON.
2. Network: bounded NetworkX ego-network and reciprocal/cycle evidence.
3. FATF Evidence: BM25/Dense/Hybrid/Rerank search and transaction-ID lookup.
4. Dashboard: temporal activity, currency-separated transaction counts, case table.
5. Benchmark Lab: read-only comparison of retrieval metrics, initial QA metrics and the post-fix targeted regression.

The application never loads `data/evaluation/labels.parquet`, `subset_selection_private.json`, or gold answers as an inference input. Agent model calls happen only when explicitly submitted in Investigation.

## Annotating RET-04 / RET-08 / RET-12

The file to edit is **`data/evaluation/retrieval_gold.jsonl`**. The input candidate passages are in `data/evaluation/retrieval_candidates.jsonl`; these must not be used as the gold annotations without checking the actual FATF PDF.

From project root:

```powershell
python scripts/annotate_test_retrieval.py
```

This displays proposed local chunks and the paths to verify against the PDF. The suggestions are:

| Question | Source PDF page | Candidate local chunk IDs |
| --- | --- | --- |
| RET-04 | Banking RBA, PDF 23 | `fatf_banking_rba_2014-p023-c001-58978c51` and `fatf_banking_rba_2014-p023-c003-58978c51` |
| RET-08 | TBML Indicators, PDF 4 | `fatf_tbml_indicators_2021-p004-c001-cf761b4c` |
| RET-12 | Financial Investigations, PDF 22 | `fatf_financial_investigations_2012-p022-c001-cb1475fd` and `fatf_financial_investigations_2012-p022-c004-cb1475fd` |

Once you've confirmed the text **in the PDF** and the local chunk text, run:

```powershell
python scripts/annotate_test_retrieval.py --write
```

Type `VERIFIED` for each of the three source sets **only after independent PDF verification**. The script backs up `retrieval_gold.jsonl`, updates only those three test rows, and keeps all nine development rows unchanged. If one proposed chunk is incorrect, do not approve it; edit the JSONL manually to the right supporting chunk(s) instead.

Then perform a **single** held-out retrieval run:

```powershell
python -m src.benchmark_retrieval run --split test
python -m src.benchmark_retrieval score --split test
```

Do not alter retrieval settings or model selection based on those test results.

## Cautions

- The application uses a synthetic, enriched demo subset; this is not production fraud detection.
- Network visualization may display only the top 70 nodes for readability. Numeric graph summaries come from the bounded full response, not the displayed subset.
- Plot lines do not encode transfer direction. Original transaction rows do.
- Monetary values are displayed separately by currency.
- The app reports saved benchmark results; it does not claim to have re-evaluated the models.
- This upgrade was syntactically and unit-tested in isolation, but it cannot be fully end-to-end tested against the locally held Parquet data or API credentials on your Windows machine.

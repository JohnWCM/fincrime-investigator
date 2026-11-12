# FinCrime Investigator — Expanded-account benchmark

This is an **additional**, account-disjoint evaluation. It does **not** modify your existing Day 3 benchmark, regression scores, Streamlit app, or datasets.

## Installation

Extract the ZIP into the project root, preserving the `scripts/` and `docs/` folders. It adds `scripts/benchmark_expanded.py`.

Run from `C:\Users\jjwei\final project Ironhack\fincrime-investigator` in your existing Python environment.

## Prepare: validates original versus expanded account coverage

```powershell
python scripts/benchmark_expanded.py prepare
```

The script uses `data/processed/transactions.parquet` (unlabelled source), `data/processed/investigation_subset_expanded.parquet`, and `data/processed/subset_manifest_expanded.json`. It checks that each of the eight selected accounts has the same number of directly involving transactions in the source and expanded files. It fails instead of evaluating incomplete cases.

It writes:

- `data/evaluation/benchmark_expanded/questions.jsonl`: 32 fixed questions; 24 dev on six accounts and 8 held-out on two other accounts.
- `data/evaluation/benchmark_expanded/references.jsonl`: independent DuckDB reference answers, **not supplied to the LLM**.
- `data/evaluation/benchmark_expanded/metadata.json`: dataset fingerprint to detect accidental changes.

The four question types per account are: unique transactions, incoming transaction count, outgoing transaction count, and latest timestamp. The agent is forced to call `get_account_statistics` first. The test questions involve entirely new account IDs; do not tune the agent using the held-out predictions.

If you previously ran `prepare`, the script refuses to overwrite the references; preserve the original results. To start over intentionally, back up the entire `data/evaluation/benchmark_expanded` folder.

## Test one small pilot first

```powershell
python scripts/benchmark_expanded.py run --split dev --limit 2 --arms tools_only,agentic_rag --max-calls 4
python scripts/benchmark_expanded.py sheet --split dev
```

Open `data/evaluation/benchmark_expanded/human_scores_dev.csv`. It displays the DuckDB reference alongside the LLM answer and whether the statistics tool ran. Enter `1` or `0` for `factual_correct_0_1`. Enter `evidence_faithful_0_1` only after checking the tool trace. Keep unknown grades blank. These are human judgments, not automatic accuracy labels. `run` may incur model API charges.

```powershell
python scripts/benchmark_expanded.py score --split dev
```

## Expand to all development questions

```powershell
python scripts/benchmark_expanded.py run --split dev --limit 0 --arms tools_only,agentic_rag --max-calls 4
python scripts/benchmark_expanded.py sheet --split dev
python scripts/benchmark_expanded.py score --split dev
```

Expected: 48 prediction files (24 questions x two agents), assuming all complete. Existing predictions are cached, so the pilot is not rerun. You must complete human grading to calculate factual accuracy.

## Only after the design is frozen: held-out test

```powershell
python scripts/benchmark_expanded.py run --split test --limit 0 --arms tools_only,agentic_rag --max-calls 4
python scripts/benchmark_expanded.py sheet --split test
python scripts/benchmark_expanded.py score --split test
```

Test: 8 questions for two never-before-evaluated accounts (16 predictions). Do **not** modify prompts, tools or hyperparameters based on test results.

## Interpretation and limitations

- New data and evaluation outputs are kept separate from the 39-response Day 3 development experiment, six-response regression, and FATF retrieval evaluation.
- All four new question types test **exact SQL statistics**. They do *not* demonstrate improved FATF citation faithfulness, counterparty ranking, graph analysis or financial-crime detection.
- The `agentic_rag` arm has document search available but has no reason to invoke RAG for pure SQL questions. Comparing the arms here tests numerical reliability and orchestration—not the added value of RAG.
- The dataset is selected, synthetic and unlabelled for inference. Do not present this as precision/recall for money-laundering detection or as representative population performance.
- There is no claim of real end-to-end integration testing here; run the scripts in the user's existing project environment.

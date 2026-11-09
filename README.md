# FinCrime Investigator

**Ironhack AI/ML LLM-track final project — Day 1 starter.**

FinCrime Investigator is a read-only AML investigation prototype based on **fully synthetic** IBM AMLworld HI-Small financial transactions. Today the demonstrable component is the deterministic baseline. RAG, an agent, and full benchmark evaluation are planned for Days 2–4; they are **not yet implemented** in this starter.

## Quick start (Windows CMD)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-day1.txt
pip install kaggle
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw --unzip
python -m src.ingest
python -m src.prepare_data
python -m src.evaluation
pytest -q
streamlit run app/streamlit_app.py
```

If Kaggle authentication is required, run `kaggle auth login` as supported by your installed Kaggle CLI, or use an existing authenticated Kaggle setup. Alternatively download the specific CSV through the Kaggle web page and place it at `data/raw/HI-Small_Trans.csv`.

To inspect full-dataset counts and class prevalence:

```bat
python -c "import json; print(json.dumps(json.load(open('data/evaluation/ingest_stats.json')), indent=2))"
```

To run the deterministic baseline using an actual case ID, open `data/processed/subset_manifest.json`, copy a `cases[].account_id`, then run:

```bat
python -m src.baseline --account "BANK::ACCOUNT"
```

`BANK::ACCOUNT` is a placeholder, **not** an actual ID.

## Data lineage and separation

* Source: IBM [AML-Data](https://github.com/IBM/AML-Data) / [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
* IBM identifies its dataset license as **CDLA-Sharing-1.0**; this differs from the repository's Apache-2.0 code license. Download the dataset separately and check the source terms before redistributing derived data.
* `data/raw/HI-Small_Trans.csv`: private, unmodified source.
* `data/processed/transactions.parquet`: all transactions **without** `Is Laundering`.
* `data/evaluation/labels.parquet`: evaluation-only `{tx_id, is_laundering}`. **Not accessible to the inference tools or app.**
* `data/processed/investigation_subset.parquet`: same unlabelled schema, selected by deterministic case-root selection and complete node-induced one-hop ego networks. Edges are never randomly sampled or truncated. If a case violates edge/node budgets, that entire case is discarded.
* `data/processed/subset_manifest.json`: public case IDs and sampling method. No labels or positive/negative case labels.
* `data/evaluation/subset_selection_private.json`: private selection diagnostics, not used by the app.
* `data/evaluation/questions.jsonl`: 24 *predefined question specifications*, with holdout assignment; `gold_answer` is **null** until human verification. This is not yet a completed gold benchmark.
* `data/evaluation/ingest_stats.json`: observed full-dataset distributions and source SHA-256.

**Important:** Case selection deliberately enriches positive-labelled account neighborhoods, so this subset **cannot** estimate population-level precision, recall, or class prevalence. Those metrics should be measured on an appropriate representative held-out population, not the demo subset. Labels must never be supplied to the agent.

## Baseline

`src.functions.FinancialTools` exposes read-only, parameterized financial queries. `src.baseline.investigate` generates deterministic reports with transaction statistics, daily fan-in/out heuristics and candidate rapid-movement patterns. `src.graph` adds NetworkX reciprocal-link and short-cycle analysis on bounded complete networks. Different currencies are never summed into one value. The reported triage priority is a transparent **rule**, not a calibrated likelihood of criminality.

The input CSV does not contain original transaction IDs. `TX-000000000001` and subsequent IDs are generated from 1-based source CSV row ordinals. They are stable for this specific file and its SHA-256 digest; a changed CSV version will change the mapping.

## Evaluation and limitations

The initial 24 question specifications contain 6 guidance, 8 account, 6 network/pattern and 4 unsupported questions; 18 dev and 6 test. Non-knowledge dev and test questions use disjoint account-ID pools selected via deterministic hashing. Fill `gold_answer` and cite verified transaction IDs/doc chunk IDs before reporting scores. Avoid inspecting test answers during prompt development. The unsupported questions require abstention, not accusations.

**Known Day 1 limitations:** heuristics can generate false positives, rapid matching does not establish beneficial ownership or movement of identical funds, central-node neighborhoods may be rejected by the bounded-subgraph constructor, and no FATF RAG or LLM has been implemented yet.

## Next steps

Day 2: FATF PDFs, indexed chunks, BM25/dense/hybrid retrievers and bounded tool-calling agent. Day 3: benchmark all architectures with cost and latency; failure analysis. Day 4: Streamlit polish, tests, slides and a tested 3-minute demo.

## FATF retrieval starter (Day 1, optional if the data pipeline takes longer)

The Day-1 RAG module provides PDF download/extraction, stable page-specific chunk IDs, a BM25 retriever, and a Chroma/SentenceTransformers dense retriever. The FATF source PDFs are downloaded only on your machine. Retrieval evaluation and hybrid/reranking are **not** implemented in this starter.

```bat
pip install -r requirements.txt
python -m src.rag download
python -m src.rag ingest
python -m src.rag search "What is a risk-based approach?" --method bm25
python -m src.rag build-dense
python -m src.rag search "What is a risk-based approach?" --method dense
```

If FATF disallows scripted PDF download, use its original publication pages linked in `src/rag.py` and save each official PDF to the exact filename shown there under `data/raw/fatf/`. The 2014 banking guidance predates changes to Recommendation 1 in 2025; the app must prominently disclose this limitation. PDF-extracted section headings are tentative `section_hint` fields, not guaranteed structural annotations. Review them manually when constructing gold citations.

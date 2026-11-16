# Fresh-clone reproducibility and data provenance

## Goals and boundaries

Two reproducibility paths are supported:

1. **Generated demo:** start the same Streamlit interface from a fresh clone, including SQL, Account Intelligence, Pattern Alerts, 1-/2-hop networks and FATF retrieval (after official PDF downloads and indexing). The tiny fictional dataset is **new and unrelated to the IBM study**. It is not evaluated for fraud/AML accuracy. Agent buttons additionally need your own API key.
2. **Original IBM experiment:** use the identical original IBM HI-Small CSV, ingest/preparation scripts, FATF PDFs and the documented human evaluation to reproduce the research runs. A different source revision, PDF byte sequence, model download or human annotations can alter IDs/scores. The original dataset is not bundled.

Data files, label files and cached models are intentionally ignored by Git. They are locally generated or downloaded. No training labels reach the Streamlit application or the LLM tool boundary.

## Clean installation (Windows PowerShell)

```powershell
# From a fresh terminal; use Python 3.11

git clone https://github.com/JohnWCM/fincrime-investigator.git
cd fincrime-investigator
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-agent.txt
python -m scripts.bootstrap_demo
python -m scripts.verify_reproduction --strict-rag
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

If you already have this checkout, you do **not** need to clone again.

The data generator uses fixed seed 42 and writes only under `data/demo/`. It creates CSV data using the IBM-compatible *format* but not IBM transactions, then invokes `src.ingest.ingest` to produce unlabeled Parquet and an evaluation-only labels file. It copies the full fictional graph to a demo investigation subset. The default account `D00::CENTER` has a real 2-hop expansion and deliberately repeated transfers for testing the UI.

`python -m scripts.bootstrap_demo --force` regenerates **only** `data/demo/`; it does not delete IBM datasets or FATF indices. A normal second invocation reuses the generated demo data and can retry missing FATF downloads or indexing.

### Offline partial smoke test

```powershell
python -m scripts.bootstrap_demo --skip-fatf
python -m scripts.verify_reproduction
python -m streamlit run app/streamlit_app.py
```

This checks financial/graph/UI functionality, **not** FATF Evidence or Agentic RAG. Do not label it a complete application reproduction. To enable retrieval, rerun bootstrap without `--skip-fatf` (and without `--skip-dense`).

### External dependencies and possible download failures

The normal bootstrap fetches the three official FATF PDFs using the URLs in `src/rag.py` and constructs the BM25 corpus and Chroma dense index. It also downloads a SentenceTransformers model on first run. These sources and model hosts must be reachable. If a PDF URL fails, obtain the relevant PDFs from the official landing pages recorded in `src/rag.py` and save them in `data/raw/fatf/` with the exact filenames:

- `banking_rba_2014.pdf`
- `tbml_indicators_2021.pdf`
- `financial_investigations_2012.pdf`

Then rerun `python -m scripts.bootstrap_demo`. The application can display FATF guidance with no API key once the index is built. Agentic calls require an OpenAI API key and a model enabled for that API account and may incur charges. Copy `.env.example` to `.env` and fill `OPENAI_API_KEY` and `OPENAI_MODEL` locally; never commit `.env`.

### Choosing the active dataset

The application selects the first complete pair of subset+manifest in this order:

1. `data/processed/investigation_subset_2hop_demo.parquet` + `subset_manifest_2hop_demo.json` (original IBM two-hop demo)
2. `data/processed/investigation_subset_expanded.parquet` + `subset_manifest_expanded.json`
3. `data/processed/investigation_subset.parquet` + `subset_manifest.json`
4. `data/demo/processed/investigation_subset.parquet` + `subset_manifest.json` (generated fictional demo)

To override automatic selection:

```powershell
$env:FINCRIME_DATASET = "demo"  # demo | ibm | expanded | twohop | auto
python -m streamlit run app/streamlit_app.py
```

Or set `FINCRIME_SUBSET_PATH` and `FINCRIME_MANIFEST_PATH` **together**. Path selection does not grant access to label files.

## Reproducing the original IBM data path

1. Download **HI-Small_Trans.csv** from the IBM AML dataset published via Kaggle, as linked in `README.md`. Its download conditions and availability are external to this repository.
2. Place it at `data/raw/HI-Small_Trans.csv`. The exact filename, column names and row order matter: transaction IDs are assigned using a 1-based CSV row ordinal.
3. Run:

```powershell
python -m src.ingest
python -m src.prepare_data
python scripts/expand_demo_cases.py --apply
python -m scripts.create_two_hop_demo --account 00701::800A8E530 --apply
$env:FINCRIME_DATASET = "twohop"
python -m streamlit run app/streamlit_app.py
```

The two scripts with `--apply` intentionally refuse to overwrite existing derived files. Their exact outputs depend on the supplied CSV and their documented case coverage constraints. Do not mix a generated demo manifest with an IBM subset.

Published dev/test benchmark JSON files remain **historical artifacts**, not automatic results of running the generated quick-start demo. For the full retrieval experiment, use the exact FATF version and verified gold annotations. The expanded SQL benchmark depends on specific IBM accounts and its manually graded predictions, not generated quick-start accounts. See the existing guides under `docs/`.

## Checks and expected caveats

- `python -m pytest -q` checks implementation contracts; the original local environment reported 34 passing tests before this upgrade, plus non-blocking SQLite datetime-adapter warnings. The added tests separately check generation, configuration and patching.
- `python -m scripts.verify_reproduction --strict-rag` validates the demo Parquet manifest and depth-2 graph, and checks that the FATF corpus and index exist; it is not a full autonomous-agent benchmark.
- `python -m pip check` checks only **installed** package compatibility. A successful bootstrap in a fresh virtual environment is stronger evidence than an existing Anaconda environment.
- `.env`, IBM data, generated demo data, Chroma indices and model caches are ignored by Git to avoid publishing credentials and large/private artifacts.

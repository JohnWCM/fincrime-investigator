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

If `py -3.11` reports **No suitable Python runtime found**, install Python 3.11 first.
You may alternatively test Python 3.13 (`py -3.13 -m venv .venv`) on your
machine, but package compatibility must be verified independently.
For a supported local install with no PowerShell activation, run the virtual
environment's `python.exe` directly (`.\.venv\Scripts\python.exe -m pip ...`).
If PowerShell blocks activation, `Set-ExecutionPolicy -Scope Process RemoteSigned`
changes the policy only for that terminal session.

`--strict-rag` requires external downloads and validates the BM25/dense retrieval
artifacts; it does **not** establish that live LLM tool calls work. Live agent
execution requires your own API key, model access and possible API charges.
The GitHub Actions job intentionally uses `--skip-fatf`, so HTTP 403 errors from the
FATF website cannot break the transaction-data CI checks. **A green Actions result
certifies only the offline demo checks, not PDF availability, FATF retrieval, or
a live LLM run.** Follow the manual full-RAG verification below for those parts.


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

Then rerun `python -m scripts.bootstrap_demo`. The application can display FATF guidance with no API key once the index is built. Agentic calls require an OpenAI API key and a model enabled for that API account and may incur charges. Copy `.env.example` to `.env` and fill `OPENAI_API_KEY` and choose an enabled `OPENAI_MODEL` locally; never commit `.env`.

### FATF HTTP 403: browser-download fallback (verified in a fresh clone)

The FATF website may return **HTTP 403 Forbidden** to Python's automated
`urlopen` downloader, even when the official PDF is available through a web
browser. This is an external access restriction, not a defect in your Python
virtual environment. Do **not** disable TLS verification or use unofficial
mirrors to work around it.

1. Open the official FATF publication pages in a browser and download each PDF:
   - [Banking risk-based approach (2014)](https://www.fatf-gafi.org/en/publications/fatfrecommendations/documents/risk-based-approach-banking-sector.html)
   - [Financial investigations guidance (2012)](https://www.fatf-gafi.org/en/publications/Methodsandtrends/Operationalissues-financialinvestigationsguidance.html)
   - [Trade-based money laundering: risk indicators (2021)](https://www.fatf-gafi.org/en/publications/Methodsandtrends/Trade-based-money-laundering-indicators.html)
2. Save them inside **the newly cloned repository** under `data/raw/fatf/`, with
   these **exact filenames**, respectively:
   `banking_rba_2014.pdf`, `financial_investigations_2012.pdf`, and
   `tbml_indicators_2021.pdf`.
3. Verify all three are real PDFs and not HTML error pages, then ingest and
   build the embedding index:

```powershell
# Run these commands from the repository root, using your virtual environment.
New-Item -ItemType Directory -Force -Path data/raw/fatf | Out-Null
@'
from pathlib import Path
names = ["banking_rba_2014.pdf", "financial_investigations_2012.pdf", "tbml_indicators_2021.pdf"]
for name in names:
    p = Path("data/raw/fatf") / name
    ok = p.is_file() and p.open("rb").read(4) == b"%PDF"
    print(f"{name}: {'OK' if ok else 'MISSING OR INVALID'}")
    if not ok:
        raise SystemExit("Download all three authentic PDFs before ingestion.")
'@ | python -

python -m src.rag ingest
python -m src.rag build-dense
python -m scripts.verify_reproduction --strict-rag
```

If using an unactivated venv, replace `python` above with
`./.venv/Scripts/python.exe` on Windows. Once the three files exist,
`python -m scripts.bootstrap_demo` can also resume setup: the downloader skips
existing filenames. Do not commit the PDFs, generated data, or Chroma index.

**Verification scope:** `--skip-fatf` checks the standalone financial demo.
`--strict-rag` checks the local FATF corpus and that BM25/dense queries return
results. A separate manual search in **FATF Evidence** confirms the Streamlit UI
can access that corpus. Neither check establishes live LLM-agent access or
reproduces historical IBM benchmark scores.

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

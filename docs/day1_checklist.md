# Day 1 completion checklist

- [ ] Repository created from the actual Ironhack template (if supplied) and scaffold files copied without deleting rubric files.
- [ ] Python 3.11 environment installed; no secrets committed.
- [ ] Kaggle file downloaded to `data/raw/HI-Small_Trans.csv`.
- [ ] `python -m src.ingest` finishes; actual row count and label prevalence stored and inspected.
- [ ] `python -m src.prepare_data` finishes; inspect complete subgraph case manifest and any rejected cases.
- [ ] `python -m src.evaluation` produces 24 question specs with 6 holdout questions; annotate independent gold answers later.
- [ ] `python -m src.baseline --account "<ACTUAL_CASE_ID>"` shows evidence IDs and the priority rule.
- [ ] `pytest -q` passes with DuckDB installed.
- [ ] `streamlit run app/streamlit_app.py` shows actual investigated case from IBM subset.
- [ ] Optional after baseline: download FATF PDFs, extract chunks, BM25 and dense search.
- [ ] Commit and push source / notebooks / documentation but NOT transactions, labels, PDFs or secrets.

A checkmark means actually observed successful behavior, not that a file exists.

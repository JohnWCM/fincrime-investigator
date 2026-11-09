# IBM AMLworld HI-Small data card

**Original source:** https://github.com/IBM/AML-Data

**Kaggle distribution:** https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

**License:** IBM states the underlying data are licensed under CDLA-Sharing-1.0, distinct from its code license. The repository must not commit source or derived IBM transaction data without checking redistribution obligations.

**Nature:** Simulated transactions; neither real individuals nor confirmed criminal findings. `Is Laundering` is a simulation label and not proof of guilt.

**Raw schema:** Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering.

**Source fingerprint:** recorded at `data/evaluation/ingest_stats.json` after ingestion; do not invent a digest.

**Composite keys:** `<bank_id>::<account_id>`; treat both as strings to avoid losing leading zeros. The delimiter is an app convention; collisions must be checked if the source uses it.

**Transaction identifiers:** synthetic identifiers, derived from the 1-based row ordinal of the specified CSV. Do not describe them as source-provided IDs.

**Subset:** Deterministic selection of some positive-associated and control roots, followed by complete induced ego subnetworks of one hop. Positive enrichment means investigation-subset prevalence is not representative. Rejected oversized ego subnetworks are tracked, not silently truncated.

**Leakage boundary:** `data/processed/` holds only unlabelled transactions; `data/evaluation/` holds private labels. Production inference should be supplied only with the subset Parquet and read-only financial tool functions.

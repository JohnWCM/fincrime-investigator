# FinCrime Investigator — Repeated Payments & Concentration Alerts

This upgrade adds a **Pattern alerts** tab to the existing **Account Intelligence** page. It does not modify the LangGraph agents, existing SQL benchmarks, the raw dataset, the expanded Parquet subset, or the FATF corpus. No API key and no LLM calls are required.

## Install on Windows (PowerShell)

Run from your repository root:

```powershell
cd "C:\Users\jjwei\final project Ironhack\fincrime-investigator"
Expand-Archive "$HOME\Downloads\fincrime-pattern-alerts-upgrade.zip" -DestinationPath "." -Force
python scripts/install_intelligence_alerts.py
python -m pytest tests/test_intelligence_alerts.py -q
python -m streamlit run app/streamlit_app.py
```

The installer changes only `app/account_intelligence.py`, and creates
`app/account_intelligence.before_pattern_alerts.py.bak` first. It is idempotent.
It does not alter `app/streamlit_app.py` or any existing benchmark scores.
If the tab anchors differ from the original Account Intelligence upgrade,
the installer refuses to change the app; do not delete your existing file.

## How to use

1. Go to **Account Intelligence** > **Pattern alerts**.
2. Choose account `023::800167090` in the sidebar, the full analysis period, and **All currencies** (or `Euro`).
3. Leave the defaults initially: **3** or more identical payments; concentration **40%** or more; at least **3** payments to the counterparty, and **5** external transfers in the entire direction/currency group.
4. Inspect the repeated-payment and concentration tables. Select a row below either table to reveal the actual transaction IDs, dates, formats, counterparties and amounts. Use **Download alert transactions (CSV)** to export the evidence for that row.
5. Change date/currency and thresholds to see how the alert counts respond.

### Expected example from your earlier screenshot

For the *full* account history of `023::800167090`, the existing counterparty drill-down showed ten **outgoing Euro** cheque transactions of **228,651.38** each to `002467::801AC3C40`, totaling **2,286,513.80**. The counterparty ranking showed approximately **52.5%** of outgoing Euro value.

You should see a corresponding repeated-payment group with `transactions=10` and an outgoing concentration alert if the data/filters have not changed. Do not hardcode those values: the upgrade derives them from the Parquet transaction rows. In particular, the percentage denominator is *all positive, external outgoing Euro value* for the account within the selected period; rounding or dataset/filter changes can affect the displayed percentage.

### What each alert means

- **Repeated identical payments:** groups positive external transfers by direction, other account, source-side currency and amount rounded to two decimals. Repeats can be routine invoices, fees, payroll, synthetic-data patterns, etc. Grouping does not combine incoming with outgoing transfers or Euro with USD. The tab shows the number of distinct payment formats as additional context.
- **Counterparty concentration:** flags a counterparty responsible for at least the configured share of *positive external transfer value* in one currency and one direction. The denominator includes **all** qualifying external transfers in that currency and direction, not only the top 30 shown in Counterparties. Self-transfers are excluded. The minimum-count rule suppresses trivial single-transfer 100% shares.
- **Evidence drill-down:** queries original transaction IDs for the selected alert and reports how many rows matched. Records are capped at 5,000 for display/export; the SQL aggregates remain complete. A warning appears if the supporting count does not match the alert's aggregate count.

These are **screening alerts only**, not probabilistic classifications or proof of money laundering. The rules do not use the synthetic laundering labels. Do not interpret repeated amounts as linked funds without other evidence.

## Validation

```powershell
python -m pytest tests/test_intelligence_alerts.py tests/test_account_intelligence.py -q
```

The additional eight regression tests include an in-memory transaction history with ten identical payments totaling 52.50% of same-direction Euro value, a cross-currency control, a self-transfer control, a zero-amount transfer, and transaction-ID evidence verification. The existing six SQL tests also run. The tests use SQLite-compatible SQL for portability; validate on the local DuckDB-backed app as well.

If the new tab doesn't appear, restart Streamlit; the installer edits `app/account_intelligence.py`, not the sidebar configuration (which already has Account Intelligence).

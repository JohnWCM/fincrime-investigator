# Account Intelligence upgrade — FinCrime Investigator

## What it adds

A new **Account Intelligence** page between Investigation and Network, with:

- Complete, parameterized DuckDB account metrics (no transaction-list truncation)
- Date and currency filters; USD and other currencies kept separate
- Daily stacked timeline: external incoming, external outgoing, self-transfers
- Currency-specific incoming/outgoing amount time series
- Overlaid rapid-movement heuristic candidates and their transaction IDs
- Incoming/outgoing counterparty rankings by count or currency-specific amount
- Counterparty concentration *within the same direction and currency*
- Individual counterparty transaction drill-down and CSV export
- Self-transfer currency-pair analysis and original transaction ledger

The page makes **no API/LLM calls**. It reads only the currently selected unlabeled
`SUBSET` Parquet data, using the existing `FinancialTools` DuckDB connection.
Your benchmark code and existing app pages are untouched.

## Install (Windows PowerShell, from project root)

```powershell
cd "C:\Users\jjwei\final project Ironhack\fincrime-investigator"
Expand-Archive "$HOME\Downloads\fincrime-account-intelligence-upgrade.zip" -DestinationPath "." -Force
python scripts/install_account_intelligence.py
python -m pytest tests/test_account_intelligence.py -q
python -m streamlit run app/streamlit_app.py
```

The installer creates:

`app/streamlit_app.before_account_intelligence.py.bak`

It edits only the import, sidebar navigation, and page dispatch in
`app/streamlit_app.py`. **It does not replace your current app file** or change
its expanded dataset paths, dollar-sign formatting, or Benchmark Lab results.
If your local Streamlit code has diverged from the inspected version, the installer
stops with an error rather than guessing where to insert code.

## Verify the page

1. In the Streamlit sidebar choose **Account Intelligence**.
2. Select `001::800058B80`, choose **All currencies**, and the full date range.
3. Confirm **351 unique transactions** (matching your expanded benchmark).
4. Compare the external incoming/outgoing and self-transfer totals; all three
   mutually exclusive categories should add to 351.
5. Change to a specific currency to see nominal amount charts. You should no
   longer see cross-currency sums.
6. Inspect **Counterparties**, select a counterparty and review its transaction IDs.
7. Open **Self-transfers** and **Transaction ledger**; confirm the CSV download.
8. Repeat with `023::800167090` (272 unique transactions).

Note: the account's 351/272 complete transaction totals are visible **only** with
All currencies and the complete date range. Currency/date filters will naturally
reduce the displayed number.

## Rollback

```powershell
Copy-Item "app/streamlit_app.before_account_intelligence.py.bak" "app/streamlit_app.py" -Force
```

Optionally remove `app/account_intelligence.py` and `app/intelligence_data.py`.

## Scope / safety

- No laundering/fraud labels exposed; no suspiciousness percentages or guilt claims.
- A rapid-movement pair is a **heuristic candidate**, not confirmed tracing of funds.
- Account-specific totals are computed over the **selected Parquet**. For the eight
  additional accounts you previously validated full direct-history completeness.
- Currency filters affect the account's sending/receiving currencies separately;
  cross-currency self-transfers are counted just once as unique transactions.
- Ledger downloads are capped at 5,000 rows. SQL aggregates remain complete.
- The ranker's share denominator is external volume of the same direction and
  currency (not a grand total across currencies).
- This uses the dependencies already used by your app: pandas, plotly, streamlit,
  duckdb, pytest for tests.

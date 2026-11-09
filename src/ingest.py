"""Convert unmodified IBM HI-Small CSV to separated unlabelled/labelled Parquet.

Only this offline preparation module is allowed to access the raw laundering label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import duckdb
from src.paths import sql_path

RAW_HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ingest(csv_path: Path, processed_dir: Path, eval_dir: Path) -> dict:
    csv_path = Path(csv_path)
    processed_dir = Path(processed_dir)
    eval_dir = Path(eval_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Download the IBM CSV first: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        observed_header = next(csv.reader(stream))
    if observed_header != RAW_HEADER:
        raise ValueError(f"Unexpected CSV schema. Observed: {observed_header!r}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '3GB'")

    # Explicit positional columns prevent DuckDB/Pandas suffixing the duplicate 'Account'.
    con.execute(f"""
        CREATE TEMP TABLE staged AS
        SELECT
            row_number() OVER () AS source_row_number,
            COALESCE(
                try_strptime(raw_timestamp, '%Y/%m/%d %H:%M'),
                try_strptime(raw_timestamp, '%Y/%m/%d %H:%M:%S'),
                try_strptime(raw_timestamp, '%Y-%m-%d %H:%M:%S')
            ) AS timestamp,
            trim(from_bank_raw) AS from_bank,
            trim(from_account_raw) AS from_account,
            trim(to_bank_raw) AS to_bank,
            trim(to_account_raw) AS to_account,
            try_cast(amount_received_raw AS DOUBLE) AS amount_received,
            trim(receiving_currency_raw) AS receiving_currency,
            try_cast(amount_paid_raw AS DOUBLE) AS amount_paid,
            trim(payment_currency_raw) AS payment_currency,
            trim(payment_format_raw) AS payment_format,
            try_cast(label_raw AS SMALLINT) AS is_laundering,
            amount_received_raw,
            amount_paid_raw
        FROM read_csv(
            {sql_path(csv_path)},
            header = false, skip = 1, delim = ',', parallel = false,
            columns = {{
                'raw_timestamp': 'VARCHAR',
                'from_bank_raw': 'VARCHAR',
                'from_account_raw': 'VARCHAR',
                'to_bank_raw': 'VARCHAR',
                'to_account_raw': 'VARCHAR',
                'amount_received_raw': 'VARCHAR',
                'receiving_currency_raw': 'VARCHAR',
                'amount_paid_raw': 'VARCHAR',
                'payment_currency_raw': 'VARCHAR',
                'payment_format_raw': 'VARCHAR',
                'label_raw': 'VARCHAR'
            }}
        )
    """)
    num_rows, invalid_rows = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE
            timestamp IS NULL OR from_bank IS NULL OR from_account IS NULL OR
            to_bank IS NULL OR to_account IS NULL OR amount_received IS NULL OR
            amount_paid IS NULL OR is_laundering NOT IN (0, 1) OR
            is_laundering IS NULL)
        FROM staged
    """).fetchone()
    if invalid_rows:
        raise ValueError(f"Refusing to export: {invalid_rows} rows have invalid required data")

    # The stage is ordered by its original CSV scan, not sorted by transaction time.
    # Thus the synthetic ID denotes CSV row number, NOT a native IBM transaction ID.
    txn_tmp = processed_dir / 'transactions.tmp.parquet'
    labels_tmp = eval_dir / 'labels.tmp.parquet'
    con.execute(f"""
        COPY (
            SELECT
                'TX-' || lpad(cast(source_row_number AS VARCHAR), 12, '0') AS tx_id,
                timestamp, from_bank, from_account,
                to_bank, to_account,
                from_bank || '::' || from_account AS from_account_id,
                to_bank || '::' || to_account AS to_account_id,
                amount_received, receiving_currency, amount_paid, payment_currency,
                payment_format, amount_received_raw, amount_paid_raw
            FROM staged
        ) TO {sql_path(txn_tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
        COPY (
            SELECT 'TX-' || lpad(cast(source_row_number AS VARCHAR), 12, '0') AS tx_id,
                   is_laundering
            FROM staged
        ) TO {sql_path(labels_tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    by_label = con.execute("""
        SELECT is_laundering, count(*) AS n
        FROM staged GROUP BY is_laundering ORDER BY is_laundering
    """).fetchall()
    min_date, max_date = con.execute("SELECT min(timestamp), max(timestamp) FROM staged").fetchone()
    con.close()
    txn_tmp.replace(processed_dir / 'transactions.parquet')
    labels_tmp.replace(eval_dir / 'labels.parquet')
    summary = {
        'source': 'IBM AML-Data / Kaggle HI-Small_Trans.csv',
        'source_url': 'https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml',
        'license': 'CDLA-Sharing-1.0 (per IBM AML-Data repository)',
        'csv_sha256': sha256_file(csv_path),
        'total_transactions': num_rows,
        'label_counts': {str(label): n for label, n in by_label},
        'laundering_rate': next((n / num_rows for label, n in by_label if label == 1), 0.0),
        'min_timestamp': str(min_date),
        'max_timestamp': str(max_date),
        'transaction_id_method': 'TX- + 1-based CSV row ordinal; immutable only for this exact input file',
        'labels_storage': 'data/evaluation/labels.parquet; excluded from inference tools',
    }
    (eval_dir / 'ingest_stats.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('data/raw/HI-Small_Trans.csv'))
    parser.add_argument('--processed-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--eval-dir', type=Path, default=Path('data/evaluation'))
    args = parser.parse_args()
    print(json.dumps(ingest(args.csv, args.processed_dir, args.eval_dir), indent=2))

if __name__ == '__main__':
    main()

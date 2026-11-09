"""Integration tests use explicitly fabricated, tiny fixtures only.
They do not establish real IBM dataset measurements.
"""
import csv
import json
from pathlib import Path

import pytest
pytest.importorskip('duckdb')

from src.ingest import RAW_HEADER, ingest
from src.prepare_data import prepare_subset
from src.functions import FinancialTools
from src.baseline import investigate
from src.evaluation import build_questions


def test_pipeline_no_label_leakage_and_traceable_baseline(tmp_path):
    raw = tmp_path / 'HI-Small_Trans.csv'
    # Synthetic test-only fixture: selected account 1::P receives funds from A,B,C
    # and sends funds to D,E, causing fan-in/fan-out test alerts.
    rows = [
        ('2022/09/01 10:00','1','A','1','P','100','US Dollar','100','US Dollar','Wire','1'),
        ('2022/09/01 10:05','1','B','1','P','100','US Dollar','100','US Dollar','Wire','0'),
        ('2022/09/01 10:10','1','C','1','P','100','US Dollar','100','US Dollar','Wire','0'),
        ('2022/09/01 10:20','1','P','1','D','100','US Dollar','100','US Dollar','Wire','1'),
        ('2022/09/01 10:21','1','P','1','E','100','US Dollar','100','US Dollar','Wire','0'),
        ('2022/09/01 10:22','1','P','1','F','100','US Dollar','100','US Dollar','Wire','0'),
        ('2022/09/02 10:00','2','J','2','Q','20','Euro','20','Euro','Cheque','0'),
        ('2022/09/02 10:05','2','K','2','Q','20','Euro','20','Euro','Cheque','0'),
        ('2022/09/02 10:10','2','L','2','Q','20','Euro','20','Euro','Cheque','0'),
    ]
    with raw.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(RAW_HEADER)
        w.writerows(rows)

    processed = tmp_path / 'processed'
    evaluations = tmp_path / 'evaluation'
    stats = ingest(raw, processed, evaluations)
    assert stats['total_transactions'] == len(rows)
    assert stats['label_counts']['1'] == 2
    import duckdb
    con = duckdb.connect()
    unlabelled_columns = set(con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{processed / 'transactions.parquet'}')"
    ).df()['column_name'])
    assert 'is_laundering' not in unlabelled_columns
    assert {'from_account_id', 'to_account_id', 'tx_id'} <= unlabelled_columns

    public = prepare_subset(processed/'transactions.parquet', evaluations/'labels.parquet',
                            processed/'investigation_subset.parquet', evaluations,
                            n_positive=1, n_control=1, seed=42)
    assert public['n_cases'] >= 1
    assert public['n_transactions'] >= 3
    # Evaluator question examples use actual chosen case IDs, not fabricated account values.
    questions = build_questions(processed/'subset_manifest.json', evaluations/'questions.jsonl')
    # If only one candidate case is eligible, generator intentionally refuses 24 questions.
    # The fixture includes P and Q, so at least 2 should normally be available.
    assert len(questions) == 24
    assert sum(q['split']=='test' for q in questions) == 6
    assert sum(q['category']=='unsupported' for q in questions) == 4

    with FinancialTools(processed/'investigation_subset.parquet') as tools:
        assert tools.account_exists('1::P')
        report = investigate('1::P', tools)
        assert report['status'] == 'completed'
        assert report['statistics']['total_unique_transactions'] == 6
        assert {'fan_in','fan_out'} <= set(report['observed_pattern_types'])
        assert report['priority'] == 'review'
        assert set(report['supporting_transaction_ids'])
        assert 'is_laundering' not in json.dumps(report).lower()
        assert 'is_laundering' not in json.dumps(tools.get_account_transactions('1::P')).lower()

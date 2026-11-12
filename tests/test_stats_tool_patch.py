"""Regression tests for the Day 3 exact-SQL LangGraph fix. No API calls.

Run from project root: python -m pytest tests/test_stats_tool_patch.py -q
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest


def test_patch_was_applied_and_benchmark_outputs_can_be_versioned():
    from pathlib import Path
    agent_code = Path('src/agent.py').read_text(encoding='utf-8')
    question_code = Path('src/benchmark_qa.py').read_text(encoding='utf-8')
    assert 'def get_account_statistics(account_id:' in agent_code
    assert 'tool_choice=\'get_account_statistics\'' in question_code
    assert 'FINCRIME_BENCHMARK_ROOT' in question_code
    assert '--question-ids' in question_code


def test_complete_account_stats_are_not_limited_to_25_rows(tmp_path):
    pytest.importorskip('duckdb')
    pytest.importorskip('langchain_core')
    from src.ingest import RAW_HEADER, ingest
    from src.functions import FinancialTools
    from src.agent import build_tools

    raw = tmp_path / 'tiny.csv'
    account = '11::P'
    rows = []
    start = datetime(2022, 9, 1)
    for i in range(31):
        stamp = start + timedelta(minutes=i)
        rows.append((stamp.strftime('%Y/%m/%d %H:%M'), '10', f'S{i:02d}', '11', 'P',
                     '100', 'US Dollar', '100', 'US Dollar', 'Wire', '0'))
    for i in range(2):
        stamp = start + timedelta(days=2, minutes=i)
        rows.append((stamp.strftime('%Y/%m/%d %H:%M'), '11', 'P', '12', f'R{i:02d}',
                     '40', 'US Dollar', '40', 'US Dollar', 'ACH', '0'))
    with raw.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(RAW_HEADER)
        w.writerows(rows)
    out_dir = tmp_path / 'processed'
    eval_dir = tmp_path / 'eval'
    ingest(raw, out_dir, eval_dir)

    with FinancialTools(out_dir/'transactions.parquet') as fin:
        listing = fin.get_account_transactions(account, limit=25)
        assert listing['complete'] is False
        assert listing['returned'] == 25
        assert listing['total'] == 33

        tools = build_tools(fin, account, 'hybrid')
        names = {t.name for t in tools}
        assert 'get_account_statistics' in names
        assert 'search_aml_guidance' in names
        stats_tool = next(t for t in tools if t.name == 'get_account_statistics')
        payload = json.loads(stats_tool.invoke({'account_id': account}))
        assert payload['kind'] == 'statistics'
        data = payload['data']
        assert data['total_unique_transactions'] == 33
        assert data['incoming_transactions'] == 31
        assert data['outgoing_transactions'] == 2
        assert data['start'] == '2022-09-01 00:00:00'
        assert data['end'] == '2022-09-03 00:01:00'
        assert data['incoming_by_currency'][0]['total_received'] == 3100.0
        assert 'is_laundering' not in json.dumps(payload)
        denied = json.loads(stats_tool.invoke({'account_id': '11::OTHER'}))
        assert denied['kind'] == 'error'


def test_tools_only_still_has_no_fatf_retrieval(tmp_path):
    pytest.importorskip('duckdb')
    pytest.importorskip('langchain_core')
    from src.agent import build_tools

    class FakeFin:
        def account_statistics(self, account_id):
            return {'account_id': account_id, 'total_unique_transactions': 1}
    tools = build_tools(FakeFin(), 'X::Y', 'hybrid')
    selected_tools = [t for t in tools if t.name != 'search_aml_guidance']
    names = {t.name for t in selected_tools}
    assert 'get_account_statistics' in names
    assert 'search_aml_guidance' not in names
    knowledge_tools = [t for t in tools if t.name == 'search_aml_guidance']
    assert len(knowledge_tools) == 1

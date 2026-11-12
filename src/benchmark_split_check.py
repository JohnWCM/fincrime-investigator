"""Check if development and test account neighborhoods overlap in the chosen subset.

Deterministic, read-only, and does not use laundering labels.
This is a leakage *diagnostic*, not a guarantee of independence.
"""
from __future__ import annotations

import json
from pathlib import Path
from src.benchmark_retrieval import read_jsonl
from src.functions import FinancialTools


def check_splits() -> dict:
    qs = read_jsonl(Path('data/evaluation/questions.jsonl'))
    dev = {q['account_id'] for q in qs if q['split'] == 'dev' and q.get('account_id')}
    test = {q['account_id'] for q in qs if q['split'] == 'test' and q.get('account_id')}
    overlap = sorted(dev & test)
    dev_neighbors = set(dev)
    test_neighbors = set(test)
    with FinancialTools() as fin:
        seeds = sorted(dev | test)
        rows = fin.con.execute('''
            SELECT from_account_id, to_account_id FROM tx
            WHERE from_account_id IN (SELECT UNNEST(?))
               OR to_account_id IN (SELECT UNNEST(?))
        ''', [seeds, seeds]).fetchall()
    for frm, to in rows:
        if frm in dev or to in dev:
            dev_neighbors.update((frm, to))
        if frm in test or to in test:
            test_neighbors.update((frm, to))
    shared = sorted(dev_neighbors & test_neighbors)
    return {
        'dev_root_accounts': len(dev), 'test_root_accounts': len(test),
        'root_account_overlap': overlap,
        'shared_one_hop_neighborhood_accounts': shared[:50],
        'n_shared_one_hop_accounts': len(shared),
        'warning': ('A root-account-disjoint split can still share neighboring accounts/transactions. '
                    'Use component-disjoint cases or explicitly disclose overlap for strict generalization claims.'),
    }


if __name__ == '__main__':
    result = check_splits()
    print(json.dumps(result, indent=2))
    Path('data/evaluation/split_overlap_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

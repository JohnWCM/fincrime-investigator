"""Deterministic Day-1 AML analyst triage, without LLM or reference to labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.functions import FinancialTools


def investigate(account_id: str, tools: FinancialTools) -> dict:
    stats = tools.account_statistics(account_id)
    if not stats['total_unique_transactions']:
        return {
            'account_id': account_id,
            'status': 'insufficient_evidence',
            'reason': 'No transactions found in the selected local dataset.',
            'priority': 'not_assessable',
            'limitations': ['Account may exist in full dataset but not in the selected subset.'],
        }
    detections = tools.detect_transaction_patterns(account_id)
    patterns = detections['patterns']
    kinds = sorted(set(p['pattern'] for p in patterns))
    supporting_ids = sorted(set(
        txid for p in patterns for txid in p['supporting_transaction_ids']
    ))
    # Transparent rule, NOT a calibrated probability or ML score.
    priority = 'review' if len(kinds) >= 2 else 'monitor' if kinds else 'routine'
    report = {
        'account_id': account_id,
        'status': 'completed',
        'investigation_period': {'start': stats['start'], 'end': stats['end']},
        'statistics': stats,
        'pattern_findings': patterns,
        'observed_pattern_types': kinds,
        'supporting_transaction_ids': supporting_ids,
        'priority': priority,
        'priority_rule': 'review: >=2 distinct heuristic pattern types; monitor: 1; routine: 0',
        'limitations': [
            'IBM data are synthetic; labels are withheld from this investigator.',
            'Patterns are heuristic indicators, not proof of a crime.',
            'Daily fan-in/fan-out thresholds are arbitrary working assumptions.',
            'Close-in-time, same-currency transfers do not prove a flow of identical funds.',
            'Subset excludes edges outside selected induced networks.',
        ],
        'recommended_next_steps': [
            'Review the cited transactions, counterparties and dates.',
            'Request permitted contextual information before drawing conclusions.',
        ],
    }
    if not detections['rapid_movement_check_complete']:
        report['limitations'].append('Rapid movement check exceeded pairwise transaction budget.')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--account', required=True, help='Composite ID e.g. BANK::ACCOUNT')
    p.add_argument('--data', type=Path, default=Path('data/processed/investigation_subset.parquet'))
    args = p.parse_args()
    with FinancialTools(args.data) as tools:
        report = investigate(args.account, tools)
    print(json.dumps(report, indent=2, default=str))

if __name__ == '__main__':
    main()

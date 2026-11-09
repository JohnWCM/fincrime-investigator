"""Generate 24 precommitted question specifications without inventing gold facts.

Gold answer/evidence fields are intentionally null until independently reviewed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

KNOWLEDGE = [
    'What does a risk-based approach mean for banks under FATF guidance?',
    'What kinds of risk indicators may suggest trade-based money laundering?',
    'What is the role of financial investigations according to FATF guidance?',
    'How should an institution interpret the presence of one risk indicator?',
    'How can transaction monitoring relate to a risk-based AML approach?',
    'Which sources of evidence can complement financial transaction records in an investigation?',
]
ACCOUNT = [
    'How many distinct transactions touch account {account}?',
    'How many incoming transactions does account {account} have?',
    'How many outgoing transactions does account {account} have?',
    'What is the earliest transaction timestamp for account {account}?',
    'What is the latest transaction timestamp for account {account}?',
    'Report incoming amounts for account {account} by currency.',
    'Report outgoing amounts for account {account} by currency.',
    'Which payment formats are observed for account {account}?',
]
NETWORK = [
    'Does account {account} exhibit daily fan-in activity under the stated threshold?',
    'Does account {account} exhibit daily fan-out activity under the stated threshold?',
    'Find transaction IDs supporting possible rapid fund movement for {account}, if any.',
    'List distinct immediate counterparties of {account}.',
    'Find any reciprocal transfer relationships involving {account}, if present.',
    'What accounts are reachable from {account} in at most two undirected hops within the subset?',
]
UNSUPPORTED = [
    'Has account {account} been proven guilty of money laundering?',
    'Who is the beneficial owner of account {account}?',
    'What is the real-world address of the person controlling account {account}?',
    'Can you guarantee that all activity involving {account} is legitimate?',
]


def build_questions(manifest_path: Path, output: Path) -> list[dict]:
    cases = json.loads(manifest_path.read_text(encoding='utf-8'))['cases']
    if len(cases) < 2:
        raise ValueError('Need at least 2 investigation cases to create evaluation questions')
    accounts = sorted(
        [c['account_id'] for c in cases],
        key=lambda account: hashlib.sha256(('evaluation-v1|' + account).encode()).hexdigest()
    )
    n_holdout_accounts = max(1, len(accounts) // 4)
    dev_accounts = accounts[:-n_holdout_accounts]
    test_accounts = accounts[-n_holdout_accounts:]
    items = []
    def add(category, templates, n_holdout):
        for i, template in enumerate(templates):
            split = 'test' if i >= len(templates) - n_holdout else 'dev'
            pool = test_accounts if split == 'test' else dev_accounts
            account = pool[i % len(pool)] if category != 'knowledge' else None
            items.append({
                'id': f'{category[:3].upper()}-{i+1:02d}',
                'category': category,
                'split': split,
                'account_id': account,
                'question': template.format(account=account),
                'gold_answer': None,
                'gold_transaction_ids': [],
                'gold_document_chunk_ids': [],
                'annotation_status': 'pending_human_verification',
                'expected_behavior': 'abstain_from_unsubstantiated_claims' if category == 'unsupported' else 'answer_with_checked_evidence',
            })
    add('knowledge', KNOWLEDGE, 2)
    add('account', ACCOUNT, 2)
    add('network', NETWORK, 1)
    add('unsupported', UNSUPPORTED, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(i) + '\n' for i in items), encoding='utf-8')
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=Path('data/processed/subset_manifest.json'))
    p.add_argument('--out', type=Path, default=Path('data/evaluation/questions.jsonl'))
    args = p.parse_args()
    items = build_questions(args.manifest, args.out)
    print(f'Generated {len(items)} question specifications; gold answers still need annotation.')

if __name__ == '__main__':
    main()

"""Generate *draft* independent reference answers for precommitted investigation Qs.

Reads ONLY unlabelled investigation_subset.parquet; never reads laundering labels.
Financial facts use direct parameterized DuckDB SQL, not src.baseline output.
Document answers remain unannotated. Human source verification is still required.
Run: python -m src.benchmark_references
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import json
from pathlib import Path

from src.benchmark_retrieval import read_jsonl, write_jsonl

SUBSET = Path('data/processed/investigation_subset.parquet')
QUESTIONS = Path('data/evaluation/questions.jsonl')
OUT = Path('data/evaluation/reference_drafts.jsonl')


def simple_rows(con, query: str, params: list) -> list[dict]:
    res = con.execute(query, params)
    cols = [c[0] for c in res.description]
    return [dict(zip(cols, row)) for row in res.fetchall()]


def account_records(con, account: str) -> list[dict]:
    return simple_rows(con, '''
        SELECT tx_id, timestamp, from_account_id, to_account_id,
               amount_paid, payment_currency, amount_received,
               receiving_currency, payment_format
        FROM tx WHERE from_account_id = ? OR to_account_id = ?
        ORDER BY timestamp, tx_id
    ''', [account, account])


def reference_for_question(con, q: dict) -> dict:
    qid, account = q['id'], q.get('account_id')
    reference = {
        'id': qid, 'category': q['category'], 'split': q['split'],
        'account_id': account, 'question': q['question'],
        'draft_answer': None, 'draft_transaction_ids': [],
        'independent_reference_method': None,
        'annotation_status': 'pending_manual_verification',
    }
    if q['category'] == 'knowledge':
        reference['independent_reference_method'] = 'manual_FATF_PDF_required'
        return reference
    if q['category'] == 'unsupported':
        reference['draft_answer'] = ('Insufficient dataset evidence: the available synthetic transaction '
                                     'records cannot establish criminal guilt, beneficial ownership, '
                                     'a real-world address or a guarantee of legitimacy.')
        reference['independent_reference_method'] = 'known_dataset_scope'
        return reference
    rows = account_records(con, account)
    inbound = [r for r in rows if r['to_account_id'] == account]
    outbound = [r for r in rows if r['from_account_id'] == account]
    ext_in = [r for r in inbound if r['from_account_id'] != account]
    ext_out = [r for r in outbound if r['to_account_id'] != account]
    ids = [r['tx_id'] for r in rows]
    reference['independent_reference_method'] = 'direct_parameterized_DuckDB_SQL'
    if qid == 'ACC-01':
        reference['draft_answer'] = len(rows)
    elif qid == 'ACC-02':
        reference['draft_answer'] = len(inbound)
        reference['draft_transaction_ids'] = [r['tx_id'] for r in inbound]
    elif qid == 'ACC-03':
        reference['draft_answer'] = len(outbound)
        reference['draft_transaction_ids'] = [r['tx_id'] for r in outbound]
    elif qid == 'ACC-04':
        reference['draft_answer'] = str(rows[0]['timestamp']) if rows else None
        reference['draft_transaction_ids'] = [rows[0]['tx_id']] if rows else []
    elif qid == 'ACC-05':
        reference['draft_answer'] = str(rows[-1]['timestamp']) if rows else None
        reference['draft_transaction_ids'] = [rows[-1]['tx_id']] if rows else []
    elif qid in ('ACC-06', 'ACC-07'):
        direction = inbound if qid == 'ACC-06' else outbound
        col = 'receiving_currency' if qid == 'ACC-06' else 'payment_currency'
        val = 'amount_received' if qid == 'ACC-06' else 'amount_paid'
        acc = defaultdict(lambda: {'n': 0, 'amount': 0.0})
        for r in direction:
            item = acc[r[col]]
            item['n'] += 1
            item['amount'] += float(r[val] or 0)
        reference['draft_answer'] = {k: {'n': v['n'], 'amount': round(v['amount'], 2)}
                                     for k, v in sorted(acc.items())}
        reference['draft_transaction_ids'] = [r['tx_id'] for r in direction]
    elif qid == 'ACC-08':
        reference['draft_answer'] = sorted({r['payment_format'] for r in rows})
        reference['draft_transaction_ids'] = ids
    elif qid in ('NET-01', 'NET-02'):
        group_rows = ext_in if qid == 'NET-01' else ext_out
        counterpart_field = 'from_account_id' if qid == 'NET-01' else 'to_account_id'
        day_groups = defaultdict(list)
        for r in group_rows:
            day_groups[str(r['timestamp'].date())].append(r)
        found = []
        for day, txs in sorted(day_groups.items()):
            if len({r[counterpart_field] for r in txs}) >= 3:
                found.append({'day': day, 'distinct_counterparties': len({r[counterpart_field] for r in txs}),
                              'transaction_ids': [r['tx_id'] for r in txs]})
        reference['draft_answer'] = {'pattern_detected': bool(found), 'days': found,
                                     'threshold': 'at least 3 distinct external counterparties per calendar day'}
        reference['draft_transaction_ids'] = sorted({tid for d in found for tid in d['transaction_ids']})
    elif qid == 'NET-03':
        candidates = []
        for i in ext_in:
            if i['amount_received'] is None or i['amount_received'] <= 0:
                continue
            for o in ext_out:
                if o['amount_paid'] is None or i['receiving_currency'] != o['payment_currency']:
                    continue
                gap = o['timestamp'] - i['timestamp']
                if timedelta(0) <= gap <= timedelta(hours=24) and 0.9 <= o['amount_paid'] / i['amount_received'] <= 1.1:
                    candidates.append({'incoming_tx_id': i['tx_id'], 'outgoing_tx_id': o['tx_id'],
                                       'gap_minutes': round(gap.total_seconds() / 60, 2)})
        reference['draft_answer'] = {'candidate_pairs': candidates,
                                     'warning': 'Timestamp/amount matching does not prove common funds'}
        reference['draft_transaction_ids'] = sorted({c[key] for c in candidates for key in ('incoming_tx_id','outgoing_tx_id')})
    elif qid == 'NET-04':
        cps = sorted(({r['from_account_id'] for r in ext_in} | {r['to_account_id'] for r in ext_out}))
        reference['draft_answer'] = cps
        reference['draft_transaction_ids'] = [r['tx_id'] for r in ext_in + ext_out]
    elif qid == 'NET-05':
        incoming_by_cp = defaultdict(list)
        outgoing_by_cp = defaultdict(list)
        for r in ext_in: incoming_by_cp[r['from_account_id']].append(r['tx_id'])
        for r in ext_out: outgoing_by_cp[r['to_account_id']].append(r['tx_id'])
        pairs = [{'counterparty': cp,
                  'incoming_transaction_ids': sorted(incoming_by_cp[cp]),
                  'outgoing_transaction_ids': sorted(outgoing_by_cp[cp])}
                 for cp in sorted(incoming_by_cp.keys() & outgoing_by_cp.keys())]
        reference['draft_answer'] = pairs
        reference['draft_transaction_ids'] = sorted({tid for p in pairs for side in ('incoming_transaction_ids','outgoing_transaction_ids') for tid in p[side]})
    elif qid == 'NET-06':
        # Two undirected steps in the selected dataset; no truncation or graph-tool reuse.
        first_hop = {r['from_account_id'] for r in rows} | {r['to_account_id'] for r in rows}
        first_hop.discard(account)
        second_edges = simple_rows(con, '''
            SELECT from_account_id, to_account_id FROM tx
            WHERE from_account_id IN (SELECT UNNEST(?))
               OR to_account_id IN (SELECT UNNEST(?))
        ''', [sorted(first_hop), sorted(first_hop)]) if first_hop else []
        second_hop = {v for r in second_edges for v in (r['from_account_id'], r['to_account_id'])}
        second_hop.discard(account)
        reference['draft_answer'] = {'within_2_hops_excluding_source': sorted(first_hop | second_hop)}
    else:
        raise ValueError(f'Unsupported question {qid}')
    return reference


def main() -> None:
    import duckdb
    if OUT.exists():
        raise FileExistsError(f'Preserving existing reference annotations at {OUT}. Move it explicitly to regenerate.')
    if not SUBSET.is_file():
        raise FileNotFoundError(SUBSET)
    con = duckdb.connect(database=':memory:')
    try:
        con.execute("CREATE VIEW tx AS SELECT * FROM read_parquet(?)", [str(SUBSET)])
    except Exception:
        # DuckDB cannot parameterise this DDL on all supported versions.
        from src.paths import sql_path
        con.execute(f'CREATE VIEW tx AS SELECT * FROM read_parquet({sql_path(SUBSET)})')
    records = [reference_for_question(con, q) for q in read_jsonl(QUESTIONS)]
    con.close()
    write_jsonl(OUT, records)
    print(f'Wrote {len(records)} independent reference DRAFTS to {OUT}.')
    print('Manually verify facts against Parquet and FATF PDFs; these are not automatically approved gold labels.')

if __name__ == '__main__':
    main()

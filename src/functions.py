"""Read-only financial evidence tools. NO references to the laundering label file.

This module is the boundary exposed to the LLM on Day 2. It only reads
unlabelled Parquet using fixed parameterized SQL, never arbitrary model SQL.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb

from src.paths import sql_path

DEFAULT_SUBSET = Path('data/processed/investigation_subset.parquet')


class FinancialTools:
    def __init__(self, subset_path: Path = DEFAULT_SUBSET):
        self.subset_path = Path(subset_path)
        if not self.subset_path.is_file():
            raise FileNotFoundError(f"Missing {self.subset_path}: run src.prepare_data")
        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f'CREATE VIEW tx AS SELECT * FROM read_parquet({sql_path(self.subset_path)})')

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _records(self, query: str, params: list[Any]):
        result = self.con.execute(query, params)
        cols = [c[0] for c in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def account_exists(self, account_id: str) -> bool:
        return bool(self.con.execute("""
            SELECT EXISTS(SELECT 1 FROM tx
            WHERE from_account_id = ? OR to_account_id = ?)
        """, [account_id, account_id]).fetchone()[0])

    def get_account_transactions(self, account_id: str, limit: int = 500) -> dict:
        if not (1 <= limit <= 5000):
            raise ValueError('limit must be between 1 and 5000')
        total = self.con.execute("""
            SELECT count(*) FROM tx
            WHERE from_account_id = ? OR to_account_id = ?
        """, [account_id, account_id]).fetchone()[0]
        records = self._records("""
            SELECT tx_id, timestamp, from_account_id, to_account_id,
                   amount_received, receiving_currency,
                   amount_paid, payment_currency, payment_format
            FROM tx WHERE from_account_id = ? OR to_account_id = ?
            ORDER BY timestamp, tx_id LIMIT ?
        """, [account_id, account_id, limit])
        for rec in records:
            rec['timestamp'] = str(rec['timestamp'])
        return {'account_id': account_id, 'total': total,
                'returned': len(records), 'complete': len(records) == total,
                'transactions': records}

    def account_statistics(self, account_id: str) -> dict:
        basic = self.con.execute("""
            SELECT count(*), min(timestamp), max(timestamp),
                   count(*) FILTER (WHERE to_account_id = ?),
                   count(*) FILTER (WHERE from_account_id = ?)
            FROM tx WHERE from_account_id = ? OR to_account_id = ?
        """, [account_id] * 4).fetchone()
        incoming = self._records("""
            SELECT receiving_currency AS currency, count(*) AS n,
                   sum(amount_received) AS total_received
            FROM tx WHERE to_account_id = ?
            GROUP BY receiving_currency ORDER BY receiving_currency
        """, [account_id])
        outgoing = self._records("""
            SELECT payment_currency AS currency, count(*) AS n,
                   sum(amount_paid) AS total_paid
            FROM tx WHERE from_account_id = ?
            GROUP BY payment_currency ORDER BY payment_currency
        """, [account_id])
        return {
            'account_id': account_id, 'total_unique_transactions': basic[0],
            'start': str(basic[1]) if basic[1] else None,
            'end': str(basic[2]) if basic[2] else None,
            'incoming_transactions': basic[3], 'outgoing_transactions': basic[4],
            'incoming_by_currency': incoming, 'outgoing_by_currency': outgoing,
            'self_transfers_counted_in_both_directions': True,
            'currency_warning': 'Do not add nominal amounts across different currencies.'
        }

    def detect_transaction_patterns(
        self, account_id: str, distinct_counterparty_threshold: int = 3,
        rapid_hours: int = 24, max_pair_rows: int = 2000,
    ) -> dict:
        if not 2 <= distinct_counterparty_threshold <= 50:
            raise ValueError('distinct_counterparty_threshold must be 2..50')
        if not 1 <= rapid_hours <= 72:
            raise ValueError('rapid_hours must be 1..72')
        patterns: list[dict] = []
        for kind, source_field, counterparty_field, own_field in (
            ('fan_in', 'to_account_id', 'from_account_id', 'from_account_id'),
            ('fan_out', 'from_account_id', 'to_account_id', 'to_account_id'),
        ):
            rows = self._records(f"""
                SELECT cast(date_trunc('day', timestamp) AS DATE) AS day,
                       count(*) AS transactions,
                       count(DISTINCT {counterparty_field}) AS distinct_counterparties,
                       list(tx_id ORDER BY timestamp, tx_id) AS evidence_ids
                FROM tx WHERE {source_field} = ? AND {own_field} <> ?
                GROUP BY 1
                HAVING count(DISTINCT {counterparty_field}) >= ?
                ORDER BY day
            """, [account_id, account_id, distinct_counterparty_threshold])
            for row in rows:
                patterns.append({
                    'pattern': kind, 'day': str(row['day']),
                    'transactions': row['transactions'],
                    'distinct_counterparties': row['distinct_counterparties'],
                    'supporting_transaction_ids': row['evidence_ids'][:10],
                    'evidence_complete': len(row['evidence_ids']) <= 10,
                    'interpretation': 'Unusual-looking concentration of counterparties; requires context, not a criminal finding.',
                })

        incoming = self._records("""
            SELECT tx_id, timestamp, amount_received AS amount, receiving_currency AS currency
            FROM tx WHERE to_account_id = ? AND from_account_id <> ?
            ORDER BY timestamp, tx_id LIMIT ?
        """, [account_id, account_id, max_pair_rows + 1])
        outgoing = self._records("""
            SELECT tx_id, timestamp, amount_paid AS amount, payment_currency AS currency
            FROM tx WHERE from_account_id = ? AND to_account_id <> ?
            ORDER BY timestamp, tx_id LIMIT ?
        """, [account_id, account_id, max_pair_rows + 1])
        rapid_checked = len(incoming) <= max_pair_rows and len(outgoing) <= max_pair_rows
        if rapid_checked:
            # Same-currency, approximate 1:1 flow pairing. This is only a
            # candidate, not proof that the exact same funds moved onward.
            for src in incoming:
                if src['amount'] is None or src['amount'] <= 0:
                    continue
                for dst in outgoing:
                    if dst['currency'] != src['currency'] or dst['amount'] is None:
                        continue
                    dt = dst['timestamp'] - src['timestamp']
                    if timedelta(seconds=0) <= dt <= timedelta(hours=rapid_hours):
                        ratio = dst['amount'] / src['amount']
                        if 0.9 <= ratio <= 1.1:
                            patterns.append({
                                'pattern': 'possible_rapid_movement',
                                'gap_minutes': dt.total_seconds() / 60,
                                'currency': src['currency'],
                                'amount_received': src['amount'], 'amount_paid': dst['amount'],
                                'supporting_transaction_ids': [src['tx_id'], dst['tx_id']],
                                'interpretation': 'Approximate same-currency flow; provenance of funds is not established.'
                            })
                            break
                if sum(p['pattern'] == 'possible_rapid_movement' for p in patterns) >= 10:
                    break
        return {
            'account_id': account_id, 'patterns': patterns,
            'rapid_movement_check_complete': rapid_checked,
            'note': None if rapid_checked else 'Rapid movement not evaluated: too many account transactions for pairwise test.',
            'thresholds': {'daily_distinct_counterparties': distinct_counterparty_threshold,
                           'rapid_hours': rapid_hours, 'value_ratio_bounds': [0.9, 1.1]},
        }

    def explore_transaction_network(
        self, account_id: str, hops: int = 1, max_edges: int = 10000
    ) -> dict:
        if hops not in (1, 2):
            raise ValueError('Day-1 network exploration supports hops=1 or 2')
        if not 1 <= max_edges <= 20000:
            raise ValueError('max_edges must be 1..20000')
        nodes = {account_id}
        all_edges: dict[str, dict] = {}
        frontier = {account_id}
        for _ in range(hops):
            if not frontier:
                break
            batch = self._records("""
                SELECT tx_id, from_account_id, to_account_id
                FROM tx
                WHERE from_account_id IN (SELECT unnest(?))
                   OR to_account_id IN (SELECT unnest(?))
                LIMIT ?
            """, [sorted(frontier), sorted(frontier), max_edges + 1])
            if len(batch) > max_edges or len(all_edges) + len(batch) > max_edges:
                return {'account_id': account_id, 'hops': hops, 'complete': False,
                        'error': 'Edge budget exceeded; no incomplete graph or derived count returned.'}
            frontier_next = set()
            for edge in batch:
                all_edges[edge['tx_id']] = edge
                frontier_next.update([edge['from_account_id'], edge['to_account_id']])
            frontier = frontier_next - nodes
            nodes |= frontier_next
        return {'account_id': account_id, 'hops': hops, 'complete': True,
                'node_count': len(nodes), 'edge_count': len(all_edges),
                'nodes': sorted(nodes), 'edges': list(all_edges.values())}

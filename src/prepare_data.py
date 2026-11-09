"""Offline, label-aware selection of bounded *complete* one-hop ego subnetworks.

The label file is used only to find informative demo cases and later evaluate;
never imported by src.functions or by the application.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import duckdb
from src.paths import sql_path


def prepare_subset(
    transactions: Path, labels: Path, output: Path, eval_dir: Path,
    n_positive: int = 8, n_control: int = 8, seed: int = 42,
    max_seed_degree: int = 200, max_edges_per_case: int = 20000,
    max_nodes_per_case: int = 2000,
) -> dict:
    if not transactions.exists() or not labels.exists():
        raise FileNotFoundError('Run python -m src.ingest first.')
    output.parent.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet({sql_path(transactions)})")
    con.execute(f"CREATE VIEW eval_labels AS SELECT * FROM read_parquet({sql_path(labels)})")
    con.execute("""
        CREATE TEMP TABLE activity AS
        SELECT account_id, count(*) AS degree FROM (
            SELECT from_account_id AS account_id FROM tx
            UNION ALL SELECT to_account_id AS account_id FROM tx
        ) GROUP BY account_id
    """)
    con.execute("""
        CREATE TEMP TABLE label_positive_accounts AS
        SELECT DISTINCT account_id FROM (
            SELECT t.from_account_id AS account_id
            FROM tx t JOIN eval_labels l USING (tx_id) WHERE l.is_laundering = 1
            UNION
            SELECT t.to_account_id AS account_id
            FROM tx t JOIN eval_labels l USING (tx_id) WHERE l.is_laundering = 1
        )
    """)
    # Use deterministic sorted candidate pools. Positive selection is deliberately
    # enriched and MUST NEVER be claimed to represent production class prevalence.
    max_candidates = max(40, 12 * (n_positive + n_control))
    positives = [r[0] for r in con.execute("""
        SELECT p.account_id FROM label_positive_accounts p
        JOIN activity a USING (account_id)
        WHERE a.degree BETWEEN 3 AND ?
        ORDER BY md5(p.account_id || ?) LIMIT ?
    """, [max_seed_degree, str(seed), max_candidates]).fetchall()]
    controls = [r[0] for r in con.execute("""
        SELECT a.account_id FROM activity a
        LEFT JOIN label_positive_accounts p USING (account_id)
        WHERE p.account_id IS NULL AND a.degree BETWEEN 3 AND ?
        ORDER BY md5(a.account_id || ?) LIMIT ?
    """, [max_seed_degree, str(seed), max_candidates]).fetchall()]

    cases = []
    selected_txids: set[str] = set()
    rejected = 0
    for stratum, candidates, quota in (
        ('label_positive_seed', positives, n_positive),
        ('control_seed', controls, n_control),
    ):
        accepted = 0
        for account_id in candidates:
            if accepted >= quota:
                break
            # If LIMIT+1 is exceeded, reject ENTIRE case; never keep arbitrary edges.
            star = con.execute("""
                SELECT tx_id, from_account_id, to_account_id FROM tx
                WHERE from_account_id = ? OR to_account_id = ?
                LIMIT ?
            """, [account_id, account_id, max_edges_per_case + 1]).fetchall()
            if len(star) > max_edges_per_case:
                rejected += 1
                continue
            node_ids = sorted({account_id} | {v for t in star for v in t[1:]})
            if len(node_ids) > max_nodes_per_case:
                rejected += 1
                continue
            # All edges *among* the account and its immediate counterparties.
            # This is a complete node-induced ego subgraph, not sampled rows.
            edges = con.execute("""
                SELECT tx_id FROM tx
                WHERE from_account_id IN (SELECT unnest(?))
                  AND to_account_id IN (SELECT unnest(?))
                LIMIT ?
            """, [node_ids, node_ids, max_edges_per_case + 1]).fetchall()
            if len(edges) > max_edges_per_case:
                rejected += 1
                continue
            txids = {r[0] for r in edges}
            selected_txids.update(txids)
            cases.append({'account_id': account_id, 'case_edge_count': len(txids),
                          'counterparty_nodes': len(node_ids) - 1})
            accepted += 1
    if not cases:
        raise ValueError('No complete ego subnetworks within limits: change max_edges_per_case.')

    con.execute('CREATE TEMP TABLE selected_ids (tx_id VARCHAR PRIMARY KEY)')
    con.executemany('INSERT INTO selected_ids VALUES (?)', [(i,) for i in sorted(selected_txids)])
    temp_out = output.with_name(output.stem + '.tmp.parquet')
    con.execute(f"""
        COPY (
            SELECT t.* FROM tx t JOIN selected_ids s USING (tx_id)
            ORDER BY t.timestamp, t.tx_id
        ) TO {sql_path(temp_out)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    subset_total, subset_labelled = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE l.is_laundering = 1)
        FROM selected_ids s JOIN eval_labels l USING (tx_id)
    """).fetchone()
    con.close()
    temp_out.replace(output)

    # No private labels or sampling stratum in the public inference manifest.
    public = {
        'source': 'IBM AML HI-Small', 'subset_method': 'complete one-hop node-induced ego subnetworks',
        'seed': seed, 'cases': cases, 'n_cases': len(cases), 'n_transactions': subset_total,
        'case_selection_warning': 'Deliberately enriched; not representative of real-world prevalence',
        'limits': {'max_seed_degree': max_seed_degree,
                   'max_edges_per_case': max_edges_per_case,
                   'max_nodes_per_case': max_nodes_per_case},
    }
    (output.parent / 'subset_manifest.json').write_text(json.dumps(public, indent=2), encoding='utf-8')
    private = {
        'n_subset_transactions': subset_total,
        'n_labelled_laundering_transactions': subset_labelled,
        'fraction_labelled': subset_labelled / subset_total,
        'positive_seed_accounts': [c['account_id'] for c in cases if c['account_id'] in positives],
        'control_seed_accounts': [c['account_id'] for c in cases if c['account_id'] in controls],
        'rejected_cases_due_to_size': rejected,
    }
    (eval_dir / 'subset_selection_private.json').write_text(json.dumps(private, indent=2), encoding='utf-8')
    return public


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-edges-per-case', type=int, default=20000)
    args = p.parse_args()
    result = prepare_subset(
        Path('data/processed/transactions.parquet'),
        Path('data/evaluation/labels.parquet'),
        Path('data/processed/investigation_subset.parquet'),
        Path('data/evaluation'),
        seed=args.seed, max_edges_per_case=args.max_edges_per_case,
    )
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

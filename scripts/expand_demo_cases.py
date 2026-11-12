"""Expanding the FinCrime Investigator demo subset with complete one-hop ego networks.

Run from the repository root:
    python scripts/expand_demo_cases.py           # preview, no writes
    python scripts/expand_demo_cases.py --apply   # build separate expanded files

Uses only unlabeled transactions. Never reads labels.parquet or raw Is Laundering.
Keeps the existing case subset and manifest untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

TARGET_ACCOUNTS = [
    "001::800058B80",      # 351 raw transactions
    "014::80006FCE0",      # 297
    "012::8000F7C60",      # 288
    "015::803037BE0",      # 286
    "01292::8004203B0",    # 283
    "010::800058920",      # 279
    "0117::8000C5D00",     # 273
    "023::800167090",      # 272
]


def sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write expanded data and public manifest")
    parser.add_argument("--max-edges-per-case", type=int, default=20000)
    parser.add_argument("--max-nodes-per-case", type=int, default=2000)
    parser.add_argument("--max-total-rows", type=int, default=150000)
    args = parser.parse_args()

    base = Path("data/processed")
    source = base / "transactions.parquet"
    original = base / "investigation_subset.parquet"
    original_manifest = base / "subset_manifest.json"
    expanded = base / "investigation_subset_expanded.parquet"
    expanded_manifest = base / "subset_manifest_expanded.json"
    for path in (source, original, original_manifest):
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run the original data pipeline first.")
    if args.max_edges_per_case < 1 or args.max_nodes_per_case < 1:
        raise SystemExit("Size limits must be positive")
    if args.apply and (expanded.exists() or expanded_manifest.exists()):
        raise SystemExit("Expanded output already exists. Move/back up both expanded files before rerunning.")

    public = json.loads(original_manifest.read_text(encoding="utf-8"))
    existing_cases = [dict(c) for c in public.get("cases", [])]
    existing_account_ids = {c["account_id"] for c in existing_cases}
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '3GB'")
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet({sql_path(source)})")
    con.execute(f"CREATE VIEW original_subset AS SELECT * FROM read_parquet({sql_path(original)})")
    con.execute("CREATE TEMP TABLE selected_ids (tx_id VARCHAR PRIMARY KEY)")
    con.execute("INSERT INTO selected_ids SELECT DISTINCT tx_id FROM original_subset")
    original_count = con.execute("SELECT count(*) FROM selected_ids").fetchone()[0]
    accepted = []
    rejected = []

    for account in TARGET_ACCOUNTS:
        if account in existing_account_ids:
            print(f"SKIP {account}: already present in original manifest")
            continue
        # Complete star or reject: never sample the first N edges for a case.
        star = con.execute("""
            SELECT tx_id, from_account_id, to_account_id
              FROM tx
             WHERE from_account_id = ? OR to_account_id = ?
             LIMIT ?
        """, [account, account, args.max_edges_per_case + 1]).fetchall()
        if not star:
            rejected.append({"account_id": account, "reason": "not found in source"})
            print(f"REJECT {account}: no matching transactions")
            continue
        if len(star) > args.max_edges_per_case:
            rejected.append({"account_id": account, "reason": "star edge limit exceeded"})
            print(f"REJECT {account}: more than {args.max_edges_per_case} direct transactions")
            continue
        nodes = sorted({account} | {node for row in star for node in row[1:]})
        if len(nodes) > args.max_nodes_per_case:
            rejected.append({"account_id": account, "reason": "counterparty node limit exceeded"})
            print(f"REJECT {account}: too many counterparties ({len(nodes)-1})")
            continue
        # Preserve the complete *node-induced* subgraph, just like src.prepare_data.
        edge_rows = con.execute("""
            SELECT tx_id FROM tx
             WHERE from_account_id IN (SELECT unnest(?))
               AND to_account_id IN (SELECT unnest(?))
             LIMIT ?
        """, [nodes, nodes, args.max_edges_per_case + 1]).fetchall()
        if len(edge_rows) > args.max_edges_per_case:
            rejected.append({"account_id": account, "reason": "induced edge limit exceeded"})
            print(f"REJECT {account}: more than {args.max_edges_per_case} induced edges")
            continue
        edge_ids = sorted({row[0] for row in edge_rows})
        con.executemany("INSERT INTO selected_ids VALUES (?) ON CONFLICT DO NOTHING",
                        [(txid,) for txid in edge_ids])
        accepted.append({
            "account_id": account,
            "case_edge_count": len(edge_ids),
            "counterparty_nodes": len(nodes) - 1,
            "full_direct_transaction_count": len(star),
            "case_source": "unlabelled_activity_expansion",
        })
        print(f"ACCEPT {account}: {len(star)} direct tx; "
              f"{len(nodes)-1} counterparties; {len(edge_ids)} induced tx")

    expanded_count = con.execute("SELECT count(*) FROM selected_ids").fetchone()[0]
    print(f"\nOriginal rows: {original_count:,}")
    print(f"Expanded rows: {expanded_count:,} (+{expanded_count - original_count:,})")
    print(f"New cases accepted: {len(accepted)}; rejected: {len(rejected)}")
    if rejected:
        print("Rejected details:", json.dumps(rejected, indent=2))
    if not accepted:
        raise SystemExit("No new cases were accepted; no files written.")
    if expanded_count > args.max_total_rows:
        raise SystemExit(f"Expanded row count exceeds max-total-rows={args.max_total_rows}; no files written")
    if not args.apply:
        print("\nPREVIEW ONLY. Run with --apply to save separate expanded files.")
        con.close()
        return

    tmp = expanded.with_name(expanded.stem + ".tmp.parquet")
    con.execute(f"""
        COPY (
            SELECT t.* FROM tx AS t
              JOIN selected_ids AS s USING (tx_id)
             ORDER BY t.timestamp, t.tx_id
        ) TO {sql_path(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    actual_rows = con.execute(f"SELECT count(*) FROM read_parquet({sql_path(tmp)})").fetchone()[0]
    if actual_rows != expanded_count:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Source inconsistency: expected {expanded_count}, wrote {actual_rows}. No files promoted.")
    # Verify complete directly involving transactions for each focal account.
    for case in accepted:
        aid = case["account_id"]
        full = con.execute("""SELECT count(*) FROM tx
                              WHERE from_account_id=? OR to_account_id=?""", [aid, aid]).fetchone()[0]
        subset = con.execute(f"""SELECT count(*) FROM read_parquet({sql_path(tmp)})
                                 WHERE from_account_id=? OR to_account_id=?""", [aid, aid]).fetchone()[0]
        if full != subset:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"INCOMPLETE {aid}: raw {full}, expanded subset {subset}")
    tmp.replace(expanded)
    manifest = dict(public)
    manifest.update({
        "subset_method": "union of complete one-hop node-induced ego subnetworks; baseline plus activity-selected cases",
        "cases": existing_cases + accepted,
        "n_cases": len(existing_cases) + len(accepted),
        "n_transactions": expanded_count,
        "added_cases": [c["account_id"] for c in accepted],
        "case_selection_warning": "Synthetic, selectively chosen demonstration cases; not a representative sample of AML activity",
        "expansion": {
            "selection_basis": "raw transaction activity only; no laundering labels used",
            "original_transaction_count": original_count,
            "added_case_count": len(accepted),
            "max_edges_per_case": args.max_edges_per_case,
            "max_nodes_per_case": args.max_nodes_per_case,
            "rejected": rejected,
        },
    })
    expanded_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    con.close()
    print(f"\nSAVED {expanded}")
    print(f"SAVED {expanded_manifest}")
    print("Existing files were not modified. Update Streamlit SUBSET/MANIFEST paths to use expanded outputs.")


if __name__ == "__main__":
    main()

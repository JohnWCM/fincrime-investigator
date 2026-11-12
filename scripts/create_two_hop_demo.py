"""Find and construct one COMPLETE, bounded two-hop network for a demo case.

Run from the FinCrime Investigator repository root:
    python scripts/create_two_hop_demo.py
    python scripts/create_two_hop_demo.py --apply
    python scripts/create_two_hop_demo.py --account 021575::80491D7A0 --apply

Only the UNLABELLED data/processed/transactions.parquet is read. Existing
subsets, manifests and benchmark files are not modified. A candidate is rejected
rather than truncating its 2-hop network to fit an edge limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

BASE = Path("data/processed")
RAW = BASE / "transactions.parquet"
EXISTING = BASE / "investigation_subset_expanded.parquet"
MANIFEST = BASE / "subset_manifest_expanded.json"
OUTPUT = BASE / "investigation_subset_2hop_demo.parquet"
OUTPUT_MANIFEST = BASE / "subset_manifest_2hop_demo.json"


def sql_path(p: Path) -> str:
    return "'" + p.resolve().as_posix().replace("'", "''") + "'"


def network_for_account(con: duckdb.DuckDBPyConnection, account: str, limit: int) -> dict:
    # The existing subset contains each selected case's direct transactions.
    first_rows = con.execute("""
        SELECT from_account_id, to_account_id
        FROM existing WHERE from_account_id = ? OR to_account_id = ?
    """, [account, account]).fetchall()
    if not first_rows:
        return {"account_id": account, "status": "no_direct_transactions"}
    first_nodes = {account} | {n for pair in first_rows for n in pair}
    con.execute("DELETE FROM hop1")
    con.executemany("INSERT INTO hop1 VALUES (?)", [(n,) for n in sorted(first_nodes)])
    # Includes ALL transactions touching a first-hop node, or rejects the case.
    rows = con.execute("""
        SELECT tx_id, from_account_id, to_account_id
        FROM raw_tx
        WHERE from_account_id IN (SELECT node_id FROM hop1)
           OR to_account_id IN (SELECT node_id FROM hop1)
        LIMIT ?
    """, [limit + 1]).fetchall()
    if len(rows) > limit:
        return {"account_id": account, "status": "edge_limit_exceeded",
                "one_hop_nodes": len(first_nodes), "min_edges": len(rows)}
    all_nodes = {n for _, sender, receiver in rows for n in (sender, receiver)} | first_nodes
    extra = all_nodes - first_nodes
    return {"account_id": account, "status": "eligible" if extra else "no_new_nodes",
            "one_hop_nodes": len(first_nodes), "two_hop_nodes": len(all_nodes),
            "additional_nodes": len(extra), "two_hop_edges": len(rows),
            "edge_ids": [r[0] for r in rows]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="Write separate two-hop demo files")
    p.add_argument("--account", help="Force one existing case instead of automatically choosing")
    p.add_argument("--max-two-hop-edges", type=int, default=1200,
                   help="Reject networks larger than this; Streamlit has a 3000-edge graph budget")
    args = p.parse_args()
    if not 10 <= args.max_two_hop_edges <= 1400:
        p.error("--max-two-hop-edges must be 10..1400 to fit the current Streamlit graph budget")
    for needed in (RAW, EXISTING, MANIFEST):
        if not needed.is_file():
            raise SystemExit(f"Missing {needed}. Run data preparation first.")
    if args.apply and (OUTPUT.exists() or OUTPUT_MANIFEST.exists()):
        raise SystemExit("Demo output already exists; back up/move BOTH demo files before another --apply.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candidates = [c["account_id"] for c in manifest["cases"]]
    if args.account:
        if args.account not in candidates:
            raise SystemExit("--account must match an account in subset_manifest_expanded.json")
        candidates = [args.account]
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"CREATE VIEW raw_tx AS SELECT * FROM read_parquet({sql_path(RAW)})")
    con.execute(f"CREATE VIEW existing AS SELECT * FROM read_parquet({sql_path(EXISTING)})")
    con.execute("CREATE TEMP TABLE hop1 (node_id VARCHAR PRIMARY KEY)")
    reports = []
    for account in candidates:
        result = network_for_account(con, account, args.max_two_hop_edges)
        reports.append(result)
        print(f"{account:27s}  {result['status']:20s} "
              f"1hop={result.get('one_hop_nodes', '-')} "
              f"2hop={result.get('two_hop_nodes', '-')} "
              f"new={result.get('additional_nodes', '-')} "
              f"edges={result.get('two_hop_edges', result.get('min_edges', '-'))}")
    eligible = [r for r in reports if r["status"] == "eligible"]
    if not eligible:
        con.close()
        raise SystemExit("No bounded complete two-hop cases found under this limit. "
                         "Do not claim depth-2 coverage from the current subset.")
    # Favor a legible live demo: additional hop-2 nodes, but not an immense graph.
    eligible.sort(key=lambda r: (-min(r["additional_nodes"], 30),
                                 abs(r["two_hop_nodes"] - 45), r["two_hop_edges"]))
    chosen = eligible[0]
    print("\nRECOMMENDED DEMO ACCOUNT:", chosen["account_id"])
    print(f"  One-hop nodes: {chosen['one_hop_nodes']}; two-hop nodes: {chosen['two_hop_nodes']}; "
          f"additional two-hop nodes: {chosen['additional_nodes']}; edges: {chosen['two_hop_edges']}")
    if not args.apply:
        print("\nPreview only. Run again with --apply to write TWO NEW files (not overwrite your existing dataset).")
        con.close()
        return

    con.execute("CREATE TEMP TABLE selected_ids (tx_id VARCHAR PRIMARY KEY)")
    con.execute("INSERT INTO selected_ids SELECT DISTINCT tx_id FROM existing")
    con.executemany("INSERT INTO selected_ids VALUES (?) ON CONFLICT DO NOTHING",
                    [(tx_id,) for tx_id in chosen["edge_ids"]])
    expected_rows = con.execute("SELECT count(*) FROM selected_ids").fetchone()[0]
    tmp = OUTPUT.with_name(OUTPUT.stem + ".tmp.parquet")
    try:
        con.execute(f"""
            COPY (
                SELECT tx.* FROM raw_tx tx JOIN selected_ids s USING (tx_id)
                ORDER BY tx.timestamp, tx.tx_id
            ) TO {sql_path(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        written_rows = con.execute(
            f"SELECT count(*) FROM read_parquet({sql_path(tmp)})").fetchone()[0]
        if written_rows != expected_rows:
            raise RuntimeError(f"Output mismatch: expected {expected_rows}, wrote {written_rows}")
        # Verify that our chosen account has gained genuine second-hop nodes.
        from src.functions import FinancialTools
        with FinancialTools(tmp) as fin:
            one = fin.explore_transaction_network(chosen["account_id"], hops=1, max_edges=3000)
            two = fin.explore_transaction_network(chosen["account_id"], hops=2, max_edges=3000)
        if not one.get("complete") or not two.get("complete"):
            raise RuntimeError("Network query exceeded UI edge budget. Demo not promoted.")
        actual_new = len(set(two["nodes"]) - set(one["nodes"]))
        if actual_new < 1:
            raise RuntimeError("No new nodes in the actual two-hop query. Demo not promoted.")
        # Existing focal-account direct history must remain identical.
        for case in manifest["cases"]:
            account = case["account_id"]
            old_count = con.execute("""SELECT count(*) FROM existing
                                      WHERE from_account_id=? OR to_account_id=?""",
                                    [account, account]).fetchone()[0]
            new_count = con.execute(f"""SELECT count(*) FROM read_parquet({sql_path(tmp)})
                                      WHERE from_account_id=? OR to_account_id=?""",
                                    [account, account]).fetchone()[0]
            if old_count != new_count:
                raise RuntimeError(f"Existing account history changed for {account}: {old_count} -> {new_count}")
        tmp.replace(OUTPUT)
        out_manifest = dict(manifest)
        out_manifest["n_transactions"] = written_rows
        out_manifest["subset_method"] = ("original expanded one-hop case subset plus complete, bounded "
                                         "two-hop edges for one separately identified demo account")
        out_manifest["two_hop_demo"] = {
            "account_id": chosen["account_id"], "one_hop_nodes": one["node_count"],
            "two_hop_nodes": two["node_count"], "new_nodes": actual_new,
            "two_hop_transaction_edges": two["edge_count"],
            "warning": "Complete two-hop neighborhood only for this focal account; "
                       "this is a deliberately selected demonstration subset, not the whole transaction graph."
        }
        OUTPUT_MANIFEST.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
        print(f"\nSAVED {OUTPUT}\nSAVED {OUTPUT_MANIFEST}")
        print(f"VERIFIED: {chosen['account_id']}: "
              f"1hop={one['node_count']}, 2hop={two['node_count']}, new={actual_new}")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()

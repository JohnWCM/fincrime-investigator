"""Tests for a bootstrappable fresh clone (no network or API credentials needed)."""
from __future__ import annotations
import csv
from pathlib import Path

from app.data_config import is_generated_demo, resolve_data_paths
from scripts.bootstrap_demo import FOCAL, HEADER, make_rows, write_demo_csv


def test_demo_generator_deterministic(tmp_path: Path):
    assert make_rows() == make_rows()
    assert len(make_rows()) == 168
    target = tmp_path / "HI-Small_Trans.csv"
    n = write_demo_csv(target)
    with target.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER
    assert len(rows) == n + 1
    assert rows[0][2] == rows[0][4] == "Account"
    assert all(row[-1] == "0" for row in rows[1:])
    assert any(row[1] == "D00" and row[2] == "CENTER" for row in rows[1:])
    assert FOCAL == "D00::CENTER"


def test_data_path_auto_and_explicit_override(tmp_path: Path):
    root = tmp_path
    first = root / "data/processed/investigation_subset.parquet"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"fixture")
    (first.parent / "subset_manifest.json").write_text("{}")
    assert resolve_data_paths(root, {})[0] == first
    demo = root / "data/demo/processed/investigation_subset.parquet"
    demo.parent.mkdir(parents=True)
    demo.write_bytes(b"fixture")
    (demo.parent / "subset_manifest.json").write_text("{}")
    assert resolve_data_paths(root, {"FINCRIME_DATASET": "demo"})[0] == demo
    assert is_generated_demo(demo)
    assert resolve_data_paths(root, {"FINCRIME_SUBSET_PATH": "data/demo/processed/investigation_subset.parquet",
                                     "FINCRIME_MANIFEST_PATH": "data/demo/processed/subset_manifest.json"})[0] == demo


def test_manual_paths_require_pair(tmp_path: Path):
    import pytest
    with pytest.raises(ValueError, match="together"):
        resolve_data_paths(tmp_path, {"FINCRIME_SUBSET_PATH": "a.parquet"})


def test_patch_idempotent_and_valid_syntax():
    import ast
    from scripts.apply_reproducibility_patch import patch_app
    src = '''from app.ui_helpers import x\nROOT = Path(".")\nSUBSET = ROOT / "data/processed/investigation_subset.parquet"\nMANIFEST = ROOT / "data/processed/subset_manifest.json"\ndef caption_note() -> None:\n    pass\ndef benchmark_lab() -> None:\n    pass\n'''
    changed = patch_app(src)
    assert changed == patch_app(changed)
    ast.parse(changed)
    assert "resolve_data_paths(ROOT)" in changed


def test_generated_fixture_through_real_duckdb(tmp_path: Path):
    """Integration test exercised on a fresh install (skips absent optional DuckDB)."""
    import pytest
    pytest.importorskip("duckdb")
    from app.intelligence_alerts import repeated_payment_alerts
    from app.intelligence_data import account_bounds
    from src.functions import FinancialTools
    from src.ingest import ingest

    raw = tmp_path / "source.csv"
    write_demo_csv(raw)
    processed = tmp_path / "processed"
    ingest(raw, processed, tmp_path / "evaluation")
    with FinancialTools(processed / "transactions.parquet") as fin:
        assert fin.account_exists(FOCAL)
        network1 = fin.explore_transaction_network(FOCAL, hops=1)
        network2 = fin.explore_transaction_network(FOCAL, hops=2)
        assert network1["complete"] and network2["complete"]
        assert network2["node_count"] > network1["node_count"]
        first, last = account_bounds(fin, FOCAL)
        alerts = repeated_payment_alerts(fin, FOCAL, first, last, None)
        assert any(x["transactions"] >= 10 and x["unit_amount"] == 1250 for x in alerts)

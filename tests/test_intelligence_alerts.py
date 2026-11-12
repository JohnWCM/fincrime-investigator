"""SQL tests for repeated payments and concentrated counterparties (no API)."""
from __future__ import annotations

import ast
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.intelligence_alerts import (  # noqa: E402
    alert_transaction_evidence, concentration_alerts, repeated_payment_alerts,
)

A = "023::800167090"
B = "002467::801AC3C40"
C = "011::OTHER"
D = "USD::COUNTERPARTY"
START = date(2022, 9, 1)
END = date(2022, 9, 10)


class FakeFin:
    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.con.execute("""
        CREATE TABLE tx (
            tx_id TEXT, timestamp TEXT,
            from_account_id TEXT, to_account_id TEXT,
            amount_received REAL, receiving_currency TEXT,
            amount_paid REAL, payment_currency TEXT,
            payment_format TEXT
        )""")
        data = []
        for i in range(10):
            data.append((f"R{i:02}", f"2022-09-{i+1:02} 13:00:00", A, B,
                         228651.38, "Euro", 228651.38, "Euro", "Cheque"))
        # The total positive external outgoing EUR is 4,355,264.38,
        # giving the repeated B group a 52.50% value share.
        data.append(("OTH", "2022-09-10 14:00:00", A, C, 2068750.58, "Euro", 2068750.58, "Euro", "Wire"))
        data.extend([
            ("USD1", "2022-09-03 12:00:00", A, D, 228651.38, "US Dollar", 228651.38, "US Dollar", "Wire"),
            ("SELF1", "2022-09-03 14:00:00", A, A, 228651.38, "Euro", 228651.38, "Euro", "Wire"),
            ("INC1", "2022-09-04 12:00:00", C, A, 500, "Euro", 500, "Euro", "Wire"),
            ("INC2", "2022-09-04 13:00:00", D, A, 228651.38, "US Dollar", 228651.38, "US Dollar", "Wire"),
            ("ZERO", "2022-09-04 14:00:00", A, C, 0, "Euro", 0, "Euro", "Wire"),
        ])
        self.con.executemany("INSERT INTO tx VALUES (?,?,?,?,?,?,?,?,?)", data)

    def _records(self, query, params):
        cursor = self.con.execute(query, params)
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def test_repeated_payment_group_detected_and_self_transfer_excluded():
    rows = repeated_payment_alerts(FakeFin(), A, START, END, None)
    assert len(rows) == 1
    group = rows[0]
    assert (group["direction"], group["counterparty"], group["currency"]) == ("Outgoing", B, "Euro")
    assert group["unit_amount"] == 228651.38
    assert group["transactions"] == 10
    assert group["total_amount"] == 2286513.80
    assert group["payment_formats"] == 1


def test_concentration_uses_complete_external_same_currency_denom():
    rows = concentration_alerts(FakeFin(), A, START, END, None)
    assert len(rows) == 1
    group = rows[0]
    assert group["counterparty"] == B
    assert group["flow_transactions"] == 11  # 10 B + 1 C; excludes zero and self
    assert group["transactions"] == 10
    assert group["flow_amount"] == 4355264.38
    assert group["share_pct"] == 52.50


def test_default_minimum_suppresses_one_off_incoming_100_pct():
    rows = concentration_alerts(FakeFin(), A, START, END, "Euro")
    assert all(row["direction"] == "Outgoing" for row in rows)
    assert all(row["counterparty"] != A for row in rows)


def test_thresholds_change_alerts_without_modifying_dataset():
    fin = FakeFin()
    assert not repeated_payment_alerts(fin, A, START, END, "US Dollar", min_repetitions=2)
    assert not concentration_alerts(fin, A, START, END, "Euro", min_share_pct=60)
    assert not concentration_alerts(fin, A, START, END, "Euro", min_counterparty_transactions=11)
    assert not concentration_alerts(fin, A, START, date(2022, 9, 2), "Euro", min_direction_transactions=5)


def test_repeated_group_supports_exact_transaction_evidence():
    fin = FakeFin()
    rows, count = alert_transaction_evidence(fin, A, START, END,
                                              direction="Outgoing", counterparty=B,
                                              currency="Euro", unit_amount=228651.38)
    assert count == 10
    assert {row["tx_id"] for row in rows} == {f"R{i:02}" for i in range(10)}
    rows, count = alert_transaction_evidence(fin, A, START, END,
                                              direction="Outgoing", counterparty=C,
                                              currency="Euro")
    assert count == 1  # zero nominal values excluded
    assert rows[0]["tx_id"] == "OTH"


def test_alert_scope_does_not_pull_other_currency_same_amount():
    fin = FakeFin()
    rows, total = alert_transaction_evidence(fin, A, START, END,
                                              direction="Outgoing", counterparty=D,
                                              currency="US Dollar", unit_amount=228651.38)
    assert total == 1 and [r["tx_id"] for r in rows] == ["USD1"]
    assert not repeated_payment_alerts(fin, A, START, END, "US Dollar")


def test_invalid_values_rejected_before_query_execution():
    fin = FakeFin()
    with pytest.raises(ValueError):
        repeated_payment_alerts(fin, A, START, END, None, min_repetitions=1)
    with pytest.raises(ValueError):
        concentration_alerts(fin, A, START, END, None, min_share_pct=0)
    with pytest.raises(ValueError):
        alert_transaction_evidence(fin, A, START, END,
                                   direction="Self-transfers", counterparty=B, currency="Euro")


def test_installer_adds_one_tab_and_is_idempotent(tmp_path):
    from scripts.install_intelligence_alerts import install

    source_root = Path(__file__).resolve().parents[1]
    test_app = tmp_path / "app"
    test_app.mkdir(parents=True)

    # The live app has usually already been patched by the time tests run.
    # Begin with a pristine copy, not that already-installed source.
    backup = source_root / "app" / "account_intelligence.before_pattern_alerts.py.bak"
    live = source_root / "app" / "account_intelligence.py"
    original = (backup if backup.is_file() else live).read_text(encoding="utf-8")

    # Allow this test to run even in a checkout without the backup.
    if "from app.intelligence_alerts_view import alerts_tab" in original:
        original = original.replace(
            "from app.intelligence_alerts_view import alerts_tab\n", "", 1
        )
        original = original.replace(
            "tab_timeline, tab_cp, tab_alerts, tab_self, tab_ledger = st.tabs([\n"
            "            \"Activity timeline\", \"Counterparties\", \"Pattern alerts\", \"Self-transfers\", \"Transaction ledger\"\n"
            "        ])",
            "tab_timeline, tab_cp, tab_self, tab_ledger = st.tabs([\n"
            "            \"Activity timeline\", \"Counterparties\", \"Self-transfers\", \"Transaction ledger\"\n"
            "        ])",
            1,
        )
        original = original.replace(
            "        with tab_alerts:\n"
            "            alerts_tab(fin, account_id, start, end, selected_currency)\n"
            "        with tab_self:",
            "        with tab_self:",
            1,
        )

    (test_app / "account_intelligence.py").write_text(original, encoding="utf-8")
    for name in ("intelligence_data.py", "intelligence_alerts.py", "intelligence_alerts_view.py"):
        (test_app / name).write_bytes((source_root / "app" / name).read_bytes())

    # Test first-time install, then idempotency on the second invocation.
    assert "installed" in install(tmp_path).lower()
    patched = (test_app / "account_intelligence.py").read_text(encoding="utf-8")
    ast.parse(patched)
    assert patched.count('"Pattern alerts", "Self-transfers"') == 1
    assert patched.count("alerts_tab(fin, account_id, start, end, selected_currency)") == 1
    assert "already installed" in install(tmp_path).lower()
    assert (test_app / "account_intelligence.before_pattern_alerts.py.bak").exists()

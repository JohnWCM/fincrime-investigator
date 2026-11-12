"""Local SQL regression suite. Uses stdlib SQLite for core SQL; no secrets needed."""
from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.intelligence_data import (
    counterparty_summary, overview, rank_counterparties,
    self_transfer_summary, timeline, transaction_ledger,
)

A = "001::800058B80"
B = "BANK::B"
C = "BANK::C"
D = "BANK::D"
START, END = date(2022, 9, 1), date(2022, 9, 3)


class FakeFin:
    """Only for in-memory tests: mimic the FinancialTools._records interface."""
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
        data = [
            ("T1", "2022-09-01 08:00:00", B, A, 100, "USD", 100, "USD", "Wire"),
            ("T2", "2022-09-01 09:00:00", A, C, 60, "USD", 60, "USD", "Wire"),
            # Self-transfer with currency conversion: incoming USD, outgoing EUR.
            ("T3", "2022-09-02 09:00:00", A, A, 25, "USD", 20, "EUR", "Wire"),
            ("T4", "2022-09-02 09:05:00", A, A, 50, "USD", 50, "USD", "Wire"),
            ("T5", "2022-09-03 09:00:00", A, D, 10, "EUR", 10, "EUR", "Wire"),
            ("T6", "2022-09-03 09:05:00", C, A, 30, "EUR", 30, "EUR", "Wire"),
        ]
        self.con.executemany("INSERT INTO tx VALUES (?,?,?,?,?,?,?,?,?)", data)

    def _records(self, query, params):
        cur = self.con.execute(query, params)
        columns = [c[0] for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def test_all_currency_counts_are_disjoint():
    d = overview(FakeFin(), A, START, END, None)
    assert d["unique_transactions"] == 6
    assert (d["external_incoming"], d["external_outgoing"], d["self_transfers"]) == (2, 2, 2)
    assert (d["incoming"], d["outgoing"]) == (4, 4)
    assert d["incoming_amount"] is None  # Never sum multiple currencies
    assert d["outgoing_amount"] is None


def test_usd_currency_conversion_not_double_counted():
    d = overview(FakeFin(), A, START, END, "USD")
    assert d["unique_transactions"] == 4
    assert d["external_incoming"] == 1
    assert d["external_outgoing"] == 1
    assert d["self_transfers"] == 2
    assert d["incoming"] == 3  # T1, T3, T4
    assert d["outgoing"] == 2  # T2, T4, not EUR-paid T3
    assert d["incoming_amount"] == 175
    assert d["outgoing_amount"] == 110


def test_eur_currency_filter():
    d = overview(FakeFin(), A, START, END, "EUR")
    assert d["unique_transactions"] == 3  # T3, T5, T6
    assert d["self_transfers"] == 1
    assert d["incoming"] == 1
    assert d["outgoing"] == 2
    assert d["outgoing_amount"] == 30


def test_counterparties_external_only_and_per_currency():
    rows = counterparty_summary(FakeFin(), A, START, END, None)
    assert len(rows) == 4
    assert all(r["counterparty"] != A for r in rows)
    usd_in = rank_counterparties(rows, "Incoming", "Amount")
    assert usd_in[0]["counterparty"] == B
    assert usd_in[0]["currency"] == "USD"
    assert usd_in[0]["amount"] == 100
    assert usd_in[0]["share_of_currency_flow_pct"] == 100


def test_self_breakdown_and_drill_down():
    rows = self_transfer_summary(FakeFin(), A, START, END, "USD")
    assert sum(r["transactions"] for r in rows) == 2
    transactions, count = transaction_ledger(FakeFin(), A, START, END, "USD",
                                              direction="Self-transfers")
    assert count == 2
    assert {r["tx_id"] for r in transactions} == {"T3", "T4"}
    txs, count = transaction_ledger(FakeFin(), A, START, END, "USD",
                                     direction="Incoming", counterparty=B)
    assert count == 1
    assert txs[0]["tx_id"] == "T1"


def test_timeline_and_date_filter():
    rows = timeline(FakeFin(), A, START, END, None)
    assert sum(r["external_incoming"] for r in rows) == 2
    assert sum(r["external_outgoing"] for r in rows) == 2
    assert sum(r["self_transfers"] for r in rows) == 2
    day_two = overview(FakeFin(), A, date(2022, 9, 2), date(2022, 9, 2), None)
    assert day_two["unique_transactions"] == 2

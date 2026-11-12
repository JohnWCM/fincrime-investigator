"""Deterministic, read-only pattern alerts for FinCrime Account Intelligence.

Alerts are screening heuristics, never evidence of guilt. They use only the
unlabeled transaction table, restrict the date/account scope, exclude
self-transfers, and keep currency and flow direction separate.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from app.intelligence_data import _scoped_query

# The SQL includes only parameterized values (the two fixed direction branches
# are literals); neither account names nor user thresholds enter SQL text.
_FLOW_SQL = """
, external_flows AS (
    SELECT 'Outgoing' AS direction, to_account_id AS counterparty,
           payment_currency AS currency, amount_paid AS amount,
           tx_id, timestamp, payment_format
    FROM relevant
    WHERE is_from AND NOT is_to AND paid_selected
    UNION ALL
    SELECT 'Incoming' AS direction, from_account_id AS counterparty,
           receiving_currency AS currency, amount_received AS amount,
           tx_id, timestamp, payment_format
    FROM relevant
    WHERE is_to AND NOT is_from AND received_selected
)
"""


def repeated_payment_alerts(
    fin: Any, account_id: str, first_day: date, last_day: date,
    currency: str | None, min_repetitions: int = 3,
    limit: int = 200,
) -> list[dict]:
    """Groups repeated *positive* nominal values, within counterparty/direction/currency.

    One group can include multiple payment formats; those formats can be
    examined in the underlying transaction evidence.
    """
    if not 2 <= min_repetitions <= 100:
        raise ValueError("min_repetitions must be between 2 and 100")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    return fin._records(cte + _FLOW_SQL + """
        SELECT direction, counterparty, currency,
               ROUND(amount, 2) AS unit_amount,
               COUNT(*) AS transactions,
               ROUND(SUM(amount), 2) AS total_amount,
               MIN(timestamp) AS first_seen,
               MAX(timestamp) AS last_seen,
               COUNT(DISTINCT payment_format) AS payment_formats
        FROM external_flows
        WHERE amount > 0 AND ROUND(amount, 2) > 0
        GROUP BY direction, counterparty, currency, ROUND(amount, 2)
        HAVING COUNT(*) >= ?
        ORDER BY transactions DESC, total_amount DESC, counterparty
        LIMIT ?
    """, [*params, min_repetitions, limit])


def concentration_alerts(
    fin: Any, account_id: str, first_day: date, last_day: date,
    currency: str | None, min_share_pct: float = 40.0,
    min_counterparty_transactions: int = 3,
    min_direction_transactions: int = 5,
    limit: int = 200,
) -> list[dict]:
    """Share of *positive external value* by direction AND currency.

    The percentage denominator is the complete scoped directional flow,
    never the visible top-ranked counterparties or a multi-currency sum.
    A minimum flow size prevents 1/1 transfers from triggering by default.
    """
    if not 0 < min_share_pct <= 100:
        raise ValueError("min_share_pct must be in (0, 100]")
    if not 1 <= min_counterparty_transactions <= 100000:
        raise ValueError("Invalid minimum counterparty transaction count")
    if not 1 <= min_direction_transactions <= 100000:
        raise ValueError("Invalid minimum directional transaction count")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    return fin._records(cte + _FLOW_SQL + """
        , positive_flows AS (
            SELECT * FROM external_flows WHERE amount > 0
        ), grouped AS (
            SELECT direction, currency, counterparty,
                   COUNT(*) AS transactions,
                   SUM(amount) AS total_amount,
                   MIN(timestamp) AS first_seen,
                   MAX(timestamp) AS last_seen
            FROM positive_flows
            GROUP BY direction, currency, counterparty
        ), with_totals AS (
            SELECT *,
                   SUM(total_amount) OVER (PARTITION BY direction, currency) AS flow_amount,
                   SUM(transactions) OVER (PARTITION BY direction, currency) AS flow_transactions
            FROM grouped
        )
        SELECT direction, counterparty, currency, transactions,
               ROUND(total_amount, 2) AS total_amount,
               ROUND(flow_amount, 2) AS flow_amount,
               flow_transactions,
               ROUND(100.0 * total_amount / NULLIF(flow_amount, 0), 2) AS share_pct,
               first_seen, last_seen
        FROM with_totals
        WHERE transactions >= ? AND flow_transactions >= ?
          AND flow_amount > 0 AND 100.0 * total_amount / flow_amount >= ?
        ORDER BY share_pct DESC, total_amount DESC, counterparty
        LIMIT ?
    """, [*params, min_counterparty_transactions,
          min_direction_transactions, min_share_pct, limit])


def alert_transaction_evidence(
    fin: Any, account_id: str, first_day: date, last_day: date,
    *, direction: str, counterparty: str, currency: str,
    unit_amount: float | None = None, limit: int = 5000,
) -> tuple[list[dict], int]:
    """Return transaction IDs and details supporting one alert.

    Supplying unit_amount filters a repeated-payment group; without it,
    the results cover the selected counterparty/currency/direction.
    """
    if direction not in ("Outgoing", "Incoming"):
        raise ValueError("Direction must be Outgoing or Incoming")
    if not counterparty or not currency:
        raise ValueError("Both counterparty and currency are required")
    if not 1 <= limit <= 5000:
        raise ValueError("limit must be in [1, 5000]")
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    if direction == "Outgoing":
        predicate = "is_from AND NOT is_to AND paid_selected AND to_account_id = ? AND payment_currency = ?"
        amount_field = "amount_paid"
    else:
        predicate = "is_to AND NOT is_from AND received_selected AND from_account_id = ? AND receiving_currency = ?"
        amount_field = "amount_received"
    params = [*params, counterparty, currency]
    predicate += f" AND {amount_field} > 0"
    if unit_amount is not None:
        if float(unit_amount) <= 0:
            raise ValueError("Unit amount must be positive")
        predicate += f" AND ROUND({amount_field}, 2) = ?"
        params.append(round(float(unit_amount), 2))
    total = fin._records(cte + f"SELECT COUNT(*) AS n FROM relevant WHERE {predicate}", params)[0]["n"]
    rows = fin._records(cte + f"""
        SELECT tx_id, timestamp, from_account_id, to_account_id,
               amount_received, receiving_currency, amount_paid,
               payment_currency, payment_format
        FROM relevant WHERE {predicate}
        ORDER BY timestamp DESC, tx_id DESC
        LIMIT ?
    """, [*params, limit])
    return rows, int(total)

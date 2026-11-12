"""Read-only, SQL-grounded account intelligence for the unlabeled FinCrime subset.

All account/currency/date values are parameters. There is no access to the
private AML label file. Monetary amounts are never summed across currencies.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any


def account_bounds(fin: Any, account_id: str) -> tuple[date, date] | None:
    rows = fin._records("""
        SELECT min(CAST(timestamp AS DATE)) AS first_day,
               max(CAST(timestamp AS DATE)) AS last_day
        FROM tx WHERE from_account_id = ? OR to_account_id = ?
    """, [account_id, account_id])
    if not rows or rows[0]["first_day"] is None:
        return None
    return rows[0]["first_day"], rows[0]["last_day"]


def account_currencies(fin: Any, account_id: str) -> list[str]:
    rows = fin._records("""
        SELECT DISTINCT currency FROM (
            SELECT payment_currency AS currency FROM tx WHERE from_account_id = ?
            UNION ALL
            SELECT receiving_currency AS currency FROM tx WHERE to_account_id = ?
        ) AS c
        WHERE currency IS NOT NULL AND CAST(currency AS VARCHAR) <> ''
        ORDER BY currency
    """, [account_id, account_id])
    return [str(row["currency"]) for row in rows]


def _scoped_query(
    account_id: str,
    first_day: date,
    last_day: date,
    currency: str | None,
) -> tuple[str, list[Any]]:
    """Generate a fixed, parameterized CTE for one account and date window.

    A self-transfer with payment currency EUR and receiving currency USD can
    appear in a USD view. It counts as USD incoming but NOT USD outgoing.
    """
    if first_day > last_day:
        raise ValueError("Start date is after end date")
    if (last_day - first_day).days > 3650:
        raise ValueError("Date range is unexpectedly long")
    start = datetime.combine(first_day, time.min)
    end_exclusive = datetime.combine(last_day + timedelta(days=1), time.min)
    sql = """
        WITH scoped AS (
            SELECT tx_id, timestamp, from_account_id, to_account_id,
                   amount_received, receiving_currency, amount_paid,
                   payment_currency, payment_format,
                   (from_account_id = ?) AS is_from,
                   (to_account_id = ?) AS is_to,
                   (CAST(? AS VARCHAR) IS NULL OR payment_currency = ?) AS paid_selected,
                   (CAST(? AS VARCHAR) IS NULL OR receiving_currency = ?) AS received_selected
            FROM tx
            WHERE (from_account_id = ? OR to_account_id = ?)
              AND timestamp >= ? AND timestamp < ?
        ), relevant AS (
            SELECT * FROM scoped
            WHERE (CAST(? AS VARCHAR) IS NULL OR
                 (is_from AND paid_selected) OR (is_to AND received_selected))
        )
    """
    parameters: list[Any] = [account_id, account_id, currency, currency,
                             currency, currency, account_id, account_id,
                             start, end_exclusive, currency]
    return sql, parameters


def overview(fin: Any, account_id: str, first_day: date,
             last_day: date, currency: str | None) -> dict:
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    result = fin._records(cte + """
        SELECT count(*) AS unique_transactions,
               count(*) FILTER (WHERE is_to AND received_selected) AS incoming,
               count(*) FILTER (WHERE is_from AND paid_selected) AS outgoing,
               count(*) FILTER (WHERE is_to AND NOT is_from) AS external_incoming,
               count(*) FILTER (WHERE is_from AND NOT is_to) AS external_outgoing,
               count(*) FILTER (WHERE is_from AND is_to) AS self_transfers,
               sum(amount_received) FILTER (WHERE is_to AND received_selected) AS incoming_amount,
               sum(amount_paid) FILTER (WHERE is_from AND paid_selected) AS outgoing_amount
        FROM relevant
    """, params)[0]
    # Totals across mixed currencies are not meaningful, even when the
    # underlying SQL would produce a number. Hide them in all-currency mode.
    if currency is None:
        result["incoming_amount"] = None
        result["outgoing_amount"] = None
    return result


def timeline(fin: Any, account_id: str, first_day: date,
             last_day: date, currency: str | None) -> list[dict]:
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    return fin._records(cte + """
        SELECT CAST(timestamp AS DATE) AS day,
               count(*) FILTER (WHERE is_to AND NOT is_from) AS external_incoming,
               count(*) FILTER (WHERE is_from AND NOT is_to) AS external_outgoing,
               count(*) FILTER (WHERE is_from AND is_to) AS self_transfers,
               sum(amount_received) FILTER (WHERE is_to AND received_selected) AS incoming_amount,
               sum(amount_paid) FILTER (WHERE is_from AND paid_selected) AS outgoing_amount
        FROM relevant
        GROUP BY 1 ORDER BY 1
    """, params)


def counterparty_summary(fin: Any, account_id: str, first_day: date,
                         last_day: date, currency: str | None) -> list[dict]:
    """External counterparties only; one row per direction and currency."""
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    return fin._records(cte + """
        SELECT 'Incoming' AS direction,
               from_account_id AS counterparty,
               receiving_currency AS currency,
               count(*) AS transactions,
               sum(amount_received) AS amount,
               max(timestamp) AS latest
        FROM relevant
        WHERE is_to AND NOT is_from AND received_selected
        GROUP BY from_account_id, receiving_currency
        UNION ALL
        SELECT 'Outgoing' AS direction,
               to_account_id AS counterparty,
               payment_currency AS currency,
               count(*) AS transactions,
               sum(amount_paid) AS amount,
               max(timestamp) AS latest
        FROM relevant
        WHERE is_from AND NOT is_to AND paid_selected
        GROUP BY to_account_id, payment_currency
    """, params)


def self_transfer_summary(fin: Any, account_id: str, first_day: date,
                          last_day: date, currency: str | None) -> list[dict]:
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    return fin._records(cte + """
        SELECT payment_currency, receiving_currency,
               count(*) AS transactions,
               sum(amount_paid) AS outgoing_amount,
               sum(amount_received) AS incoming_amount
        FROM relevant
        WHERE is_from AND is_to
        GROUP BY payment_currency, receiving_currency
        ORDER BY transactions DESC, payment_currency, receiving_currency
    """, params)


def transaction_ledger(fin: Any, account_id: str, first_day: date,
                       last_day: date, currency: str | None,
                       direction: str = "All", counterparty: str | None = None,
                       limit: int = 5000) -> tuple[list[dict], int]:
    """Filtered transaction records with explicit completeness and row cap.

    'Incoming' / 'Outgoing' here denote external transfers only;
    'Self-transfers' selects self-transfers, avoiding double counting.
    """
    if not 1 <= limit <= 5000:
        raise ValueError("Transaction row limit must be between 1 and 5000")
    conditions = {
        "All": "TRUE",
        "Incoming": "(is_to AND NOT is_from AND received_selected)",
        "Outgoing": "(is_from AND NOT is_to AND paid_selected)",
        "Self-transfers": "(is_from AND is_to)",
    }
    if direction not in conditions:
        raise ValueError("Unsupported transaction direction")
    if counterparty is not None and direction not in ("Incoming", "Outgoing"):
        raise ValueError("Counterparty filter requires Incoming or Outgoing")
    cte, params = _scoped_query(account_id, first_day, last_day, currency)
    clause = conditions[direction]
    if counterparty is not None:
        clause += " AND " + ("from_account_id = ?" if direction == "Incoming" else "to_account_id = ?")
        params = [*params, counterparty]
    total = fin._records(cte + f"SELECT count(*) AS n FROM relevant WHERE {clause}", params)[0]["n"]
    records = fin._records(cte + f"""
        SELECT tx_id, timestamp, from_account_id, to_account_id,
               amount_received, receiving_currency, amount_paid,
               payment_currency, payment_format
        FROM relevant WHERE {clause}
        ORDER BY timestamp DESC, tx_id DESC LIMIT ?
    """, [*params, limit])
    return records, int(total)


def rapid_movement_evidence(fin: Any, account_id: str, first_day: date,
                            last_day: date, currency: str | None) -> list[dict]:
    """Existing heuristic findings, limited to matching date and currency.

    Flag is based on matching amounts/timing, not funds provenance.
    """
    result = fin.detect_transaction_patterns(account_id)
    patterns = [p for p in result.get("patterns", [])
                if p.get("pattern") == "possible_rapid_movement"
                and (currency is None or p.get("currency") == currency)]
    if not patterns:
        return []
    ids = sorted({tid for p in patterns for tid in p.get("supporting_transaction_ids", [])})
    if not ids:
        return []
    event_times = {
        row["tx_id"]: row["timestamp"]
        for row in fin._records("""
            SELECT tx_id, timestamp FROM tx WHERE tx_id IN (SELECT unnest(?))
        """, [ids])
    }
    kept = []
    for p in patterns:
        event_ids = p.get("supporting_transaction_ids", [])
        if len(event_ids) < 2 or event_ids[1] not in event_times:
            continue
        event_date = event_times[event_ids[1]].date()
        if first_day <= event_date <= last_day:
            kept.append({
                "Day": event_date,
                "Incoming tx": event_ids[0],
                "Outgoing tx": event_ids[1],
                "Gap (min)": round(float(p.get("gap_minutes", 0)), 1),
                "Currency": p.get("currency"),
                "Incoming amount": p.get("amount_received"),
                "Outgoing amount": p.get("amount_paid"),
            })
    return kept


def rank_counterparties(rows: list[dict], direction: str,
                        sort_by: str = "Transactions", limit: int = 10) -> list[dict]:
    """Compute per-(flow,currency) shares; never combine currencies."""
    subset = [dict(r) for r in rows if r["direction"] == direction]
    denominators: dict[str, float] = {}
    for row in subset:
        currency = str(row["currency"])
        denominators[currency] = denominators.get(currency, 0.0) + float(row["amount"] or 0)
    for row in subset:
        amount = float(row["amount"] or 0)
        denom = denominators[str(row["currency"])]
        row["share_of_currency_flow_pct"] = round(100 * amount / denom, 1) if denom > 0 else None
    key = "amount" if sort_by == "Amount" else "transactions"
    subset.sort(key=lambda r: (float(r.get(key) or 0), int(r.get("transactions") or 0)), reverse=True)
    return subset[:limit]

"""Evidence-linked Streamlit presentation for deterministic AML screening alerts."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from app.intelligence_alerts import (
    alert_transaction_evidence,
    concentration_alerts,
    repeated_payment_alerts,
)


def _display_table(rows: list[dict], columns: dict[str, str]) -> None:
    frame = pd.DataFrame(rows)
    frame = frame[list(columns)].rename(columns=columns)
    for col in ("Amount per transfer", "Total value", "Direction value"):
        if col in frame.columns:
            frame[col] = frame[col].map(lambda x: f"{float(x):,.2f}")
    if "Share of value (%)" in frame.columns:
        frame["Share of value (%)"] = frame["Share of value (%)"].map(lambda x: f"{float(x):.2f}%")
    st.dataframe(frame, hide_index=True, use_container_width=True)


def _evidence(
    fin: Any, account_id: str, start: date, end: date,
    rows: list[dict], *, group: str,
) -> None:
    if not rows:
        return
    choices = list(range(len(rows)))

    def label(i: int) -> str:
        r = rows[i]
        prefix = f"{r['direction']} · {r['counterparty']} · {r['currency']}"
        if group == "repeat":
            return f"{prefix} · {r['transactions']} × {float(r['unit_amount']):,.2f}"
        return f"{prefix} · {float(r['share_pct']):.2f}% · {r['transactions']} transfers"

    st.markdown("##### Supporting transaction evidence")
    selected = st.selectbox("Select an alert to inspect", choices,
                            format_func=label, key=f"int_alert_{group}_{account_id}")
    item = rows[selected]
    transactions, total = alert_transaction_evidence(
        fin, account_id, start, end,
        direction=item["direction"], counterparty=item["counterparty"],
        currency=item["currency"],
        unit_amount=item["unit_amount"] if group == "repeat" else None,
    )
    st.caption(f"{total:,} matching transaction IDs; displaying {len(transactions):,} most recent.")
    if total != item["transactions"]:
        st.warning("The alert count and transaction evidence do not reconcile. Review the dataset and filters.")
    if transactions:
        st.dataframe(pd.DataFrame(transactions), hide_index=True, use_container_width=True)
        st.download_button(
            "Download alert transactions (CSV)",
            pd.DataFrame(transactions).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{group}_alert_transactions.csv", mime="text/csv",
            key=f"int_alert_export_{group}_{account_id}",
        )
    if len(transactions) < total:
        st.warning("The exported evidence is limited to 5,000 rows; the SQL aggregation is complete.")


def alerts_tab(fin: Any, account_id: str, start: date,
               end: date, currency: str | None) -> None:
    """No LLM or API calls. Uses page-level account, date and currency filters."""
    st.markdown("#### Pattern alerts")
    st.caption("Deterministic screening rules for external transfers only. "
               "Alerts indicate observations for review, never proof of illicit activity.")
    a, b, c = st.columns(3)
    with a:
        minimum_repeats = st.slider("Minimum identical payments", 2, 10, 3,
                            key=f"int_alert_min_repeats_{account_id}")
    with b:
        threshold = st.slider("Concentration threshold (%)", 10, 95, 40, step=5,
                              key=f"int_alert_share_{account_id}")
    with c:
        min_cp = st.slider("Min. payments to counterparty", 1, 10, 3,
                           key=f"int_alert_cpcount_{account_id}")

    repeats = repeated_payment_alerts(fin, account_id, start, end, currency,
                                       min_repetitions=minimum_repeats)
    concentrated = concentration_alerts(
        fin, account_id, start, end, currency,
        min_share_pct=float(threshold), min_counterparty_transactions=min_cp,
        min_direction_transactions=5,
    )
    x, y = st.columns(2)
    x.metric("Repeated payment groups", len(repeats))
    y.metric("Concentrated counterparty flows", len(concentrated))
    st.caption("Defaults: ≥3 identical positive payments and ≥40% of a same-direction, "
               "same-currency external flow. Concentration requires ≥5 external transfers "
               "in that flow; all thresholds are adjustable screening assumptions. "
               "At most 200 alert groups per type are displayed.")

    st.markdown("##### Repeated identical payments")
    if repeats:
        _display_table(repeats, {
            "direction": "Direction", "counterparty": "Counterparty",
            "currency": "Currency", "unit_amount": "Amount per transfer",
            "transactions": "Count", "total_amount": "Total value",
            "payment_formats": "Payment formats", "first_seen": "First seen",
            "last_seen": "Last seen",
        })
        _evidence(fin, account_id, start, end, repeats, group="repeat")
    else:
        st.info("No repeated external-payment groups meet the current threshold.")

    st.markdown("##### Counterparty concentration")
    if concentrated:
        _display_table(concentrated, {
            "direction": "Direction", "counterparty": "Counterparty",
            "currency": "Currency", "transactions": "Count",
            "total_amount": "Total value", "flow_amount": "Direction value",
            "share_pct": "Share of value (%)",
            "flow_transactions": "Direction transactions",
        })
        _evidence(fin, account_id, start, end, concentrated, group="concentration")
    else:
        st.info("No external counterparties meet the concentration criteria.")
    st.caption("Percentages are calculated from positive external transferred value in each "
               "direction and currency, not all currencies combined. Repetition can reflect "
               "legitimate recurring payments; concentration may reflect ordinary supplier relationships. "
               "Neither alert is a risk probability or a finding of money laundering.")

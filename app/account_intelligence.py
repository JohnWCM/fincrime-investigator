"""Streamlit Account Intelligence page for the selected synthetic case account."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.intelligence_alerts_view import alerts_tab
from app.intelligence_data import (
    account_bounds,
    account_currencies,
    counterparty_summary,
    overview,
    rank_counterparties,
    rapid_movement_evidence,
    self_transfer_summary,
    timeline,
    transaction_ledger,
)
from src.functions import FinancialTools


def _amount(value: object) -> str:
    try:
        return f"{float(value or 0):,.2f}"
    except (ValueError, TypeError):
        return "—"


def _download_csv(label: str, rows: list[dict], filename: str, key: str) -> None:
    if rows:
        data = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(label, data=data, file_name=filename, mime="text/csv", key=key)


def _account_dates(fin: FinancialTools, account_id: str) -> tuple[date, date] | None:
    return account_bounds(fin, account_id)


def _timeline_tab(fin: FinancialTools, account_id: str, start: date,
                  end: date, currency: str | None) -> None:
    st.markdown("#### Daily transaction activity")
    rows = timeline(fin, account_id, start, end, currency)
    if not rows:
        st.info("No transactions in the selected period and currency.")
        return
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df["day"])
    # Show zero-activity days, not just days with transactions.
    days = pd.date_range(start, end, freq="D")
    df = df.set_index("day").reindex(days).rename_axis("day").reset_index()
    cols = ["external_incoming", "external_outgoing", "self_transfers"]
    df[cols] = df[cols].fillna(0).astype(int)
    daily = df.melt(id_vars=["day"], value_vars=cols,
                    var_name="Flow category", value_name="Transactions")
    labels = {"external_incoming": "External incoming",
              "external_outgoing": "External outgoing", "self_transfers": "Self-transfers"}
    daily["Flow category"] = daily["Flow category"].map(labels)
    fig = px.bar(daily, x="day", y="Transactions", color="Flow category",
                 barmode="stack", title="Daily activity · non-overlapping categories",
                 labels={"day": "Date"})
    fig.update_layout(legend_title_text="", hovermode="x unified", height=420,
                      margin=dict(l=10, r=15, b=15, t=55))

    with st.spinner("Checking the existing rapid-movement heuristic..."):
        try:
            rapid = rapid_movement_evidence(fin, account_id, start, end, currency)
        except Exception as exc:
            rapid = []
            st.warning(f"Pattern overlay unavailable: {type(exc).__name__}: {exc}")
    if rapid:
        marker_df = pd.DataFrame(rapid)
        marker_days = pd.to_datetime(marker_df["Day"])
        day_totals = df[cols].sum(axis=1)
        heights = dict(zip(df["day"].dt.normalize(), day_totals))
        fig.add_trace(go.Scatter(
            x=marker_days,
            y=[heights.get(d.normalize(), 0) + max(1, max(day_totals) * .06)
               for d in marker_days],
            mode="markers", marker=dict(symbol="diamond", size=12),
            name="Rapid-movement candidate",
            customdata=marker_df[["Incoming tx", "Outgoing tx"]].to_numpy(),
            hovertemplate="Candidate on %{x|%Y-%m-%d}<br>Incoming: %{customdata[0]}<br>Outgoing: %{customdata[1]}<extra></extra>",
        ))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("External incoming, external outgoing and self-transfers are mutually exclusive. "
               "Diamond markers show heuristic rapid-movement candidates, not confirmed suspicious activity.")

    if currency is not None:
        st.markdown(f"#### Daily amounts · {currency}")
        amount_df = df[["day", "incoming_amount", "outgoing_amount"]].copy()
        amount_df[["incoming_amount", "outgoing_amount"]] = (
            amount_df[["incoming_amount", "outgoing_amount"]].fillna(0)
        )
        amount_long = amount_df.melt(id_vars="day", var_name="Flow", value_name="Amount")
        amount_long["Flow"] = amount_long["Flow"].map({
            "incoming_amount": "Incoming amount", "outgoing_amount": "Outgoing amount"})
        amount_chart = px.line(amount_long, x="day", y="Amount", color="Flow", markers=True,
                               title=f"Daily nominal flows in {currency}")
        amount_chart.update_layout(height=330, margin=dict(l=10, r=15, b=15, t=55))
        st.plotly_chart(amount_chart, use_container_width=True)
        st.caption("Incoming and outgoing totals include self-transfers in their respective currency; "
                   "these flows must not be added or interpreted as a net position without reconciliation.")
    else:
        st.info("Select an individual currency to display monetary time series. "
                "Amounts in different currencies are never added together.")

    st.markdown("#### Rapid-movement evidence")
    if rapid:
        st.dataframe(pd.DataFrame(rapid), hide_index=True, use_container_width=True)
        st.caption("This heuristic pairs similarly sized, same-currency incoming and outgoing "
                   "transfers within a time window. It does not trace the identity of funds.")
        _download_csv("Download candidate pairs (CSV)", rapid,
                      "rapid_movement_candidates.csv", f"rapid_csv_{account_id}")
    else:
        st.caption("No rapid-movement candidates returned for the selected filters. "
                   "Absence of a heuristic match does not establish absence of risk.")


def _counterparties_tab(fin: FinancialTools, account_id: str, start: date,
                        end: date, currency: str | None) -> None:
    st.markdown("#### Counterparty ranking")
    st.caption("Only transfers to or from other accounts are ranked. Self-transfers appear in a separate tab. "
               "Each row has its own direction and currency; values are never combined across currencies.")
    rows = counterparty_summary(fin, account_id, start, end, currency)
    if not rows:
        st.info("No external counterparties in the selected filters.")
        return
    x, y = st.columns(2)
    with x:
        sort_by = st.selectbox("Rank counterparties by", ["Transactions", "Amount"],
                               disabled=currency is None, key=f"int_rank_{account_id}")
        if currency is None:
            sort_by = "Transactions"
    with y:
        top_n = st.slider("Number of counterparties", min_value=5, max_value=30,
                          value=10, step=5, key=f"int_top_{account_id}")
    tables = st.columns(2)
    for col, direction in zip(tables, ("Incoming", "Outgoing")):
        ranked = rank_counterparties(rows, direction, sort_by=sort_by, limit=top_n)
        with col:
            st.markdown(f"**{direction} — top {top_n}**")
            if ranked:
                display = pd.DataFrame(ranked).rename(columns={
                    "counterparty": "Counterparty", "currency": "Currency",
                    "transactions": "Transactions", "amount": "Amount",
                    "share_of_currency_flow_pct": "Share of flow (%)", "latest": "Last seen",
                })
                st.dataframe(display[["Counterparty", "Currency", "Transactions", "Amount",
                                      "Share of flow (%)", "Last seen"]],
                             hide_index=True, use_container_width=True)
                st.caption("Share denominator: all external transfers in this direction and currency, "
                           "not merely the visible top counterparties.")
            else:
                st.info("No external transfers in this direction.")

    st.markdown("#### Drill down into a counterparty")
    direction = st.radio("Transaction direction", ["Outgoing", "Incoming"],
                         horizontal=True, key=f"int_cpdir_{account_id}")
    choices = sorted({row["counterparty"] for row in rows if row["direction"] == direction})
    if not choices:
        st.info(f"No {direction.lower()} counterparties in this selection.")
        return
    counterparty = st.selectbox("Counterparty account", choices,
                                key=f"int_cp_{account_id}_{direction}")
    ledger, total = transaction_ledger(fin, account_id, start, end, currency,
                                       direction=direction, counterparty=counterparty)
    st.caption(f"{total:,} matching transactions for {counterparty}; displaying {len(ledger):,}.")
    if ledger:
        st.dataframe(pd.DataFrame(ledger), hide_index=True, use_container_width=True)
        _download_csv("Download counterparty transactions (CSV)", ledger,
                      "counterparty_transactions.csv", f"int_cp_csv_{account_id}_{direction}")
    if len(ledger) < total:
        st.warning("Ledger output is capped at 5,000 rows. Counts and rankings above remain complete.")


def _self_tab(fin: FinancialTools, account_id: str, start: date,
              end: date, currency: str | None) -> None:
    st.markdown("#### Self-transfer breakdown")
    st.caption("A self-transfer has the same sender and recipient account. It counts once as a "
               "unique transaction but can count once in each directional total.")
    rows = self_transfer_summary(fin, account_id, start, end, currency)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Payment and receiving currency are displayed separately; different-currency "
                   "values are not netted or summed.")
    else:
        st.info("No self-transfers in the current date and currency filters.")
    ledger, total = transaction_ledger(fin, account_id, start, end, currency,
                                       direction="Self-transfers")
    st.markdown("#### Supporting transactions")
    st.caption(f"{total:,} self-transfers; displaying {len(ledger):,} most recent.")
    if ledger:
        st.dataframe(pd.DataFrame(ledger), hide_index=True, use_container_width=True)
        _download_csv("Download self-transfers (CSV)", ledger,
                      "self_transfers.csv", f"int_self_csv_{account_id}")


def account_intelligence_view(account_id: str, subset_path: Path) -> None:
    """Render full Account Intelligence page; no LLM calls and no label access."""
    st.subheader("Account Intelligence")
    st.caption("Explainable transaction activity and counterparties · DuckDB aggregates · "
               "synthetic unlabeled data · no automatic LLM/API calls")
    if not Path(subset_path).is_file():
        st.error(f"Cannot find the selected transaction dataset: {subset_path}")
        return
    with FinancialTools(Path(subset_path)) as fin:
        bounds = _account_dates(fin, account_id)
        if bounds is None:
            st.warning("This account has no records in the selected dataset.")
            return
        currencies = account_currencies(fin, account_id)
        st.markdown(f"**Selected account:** `{account_id}`")
        date_col, currency_col = st.columns([2.3, 1])
        with date_col:
            picked = st.date_input("Analysis period", value=bounds, min_value=bounds[0],
                                   max_value=bounds[1],
                                   key=f"int_period_{account_id}")
        if not isinstance(picked, (tuple, list)) or len(picked) != 2:
            st.info("Select both the start and end dates to display the analysis.")
            return
        start, end = picked
        if start > end:
            st.error("The start date must not be after the end date.")
            return
        with currency_col:
            selection = st.selectbox("Currency", ["All currencies", *currencies],
                                     index=0,
                                     key=f"int_currency_{account_id}")
        selected_currency = None if selection == "All currencies" else selection
        stats = overview(fin, account_id, start, end, selected_currency)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique transactions", f"{stats['unique_transactions']:,}")
        c2.metric("External incoming", f"{stats['external_incoming']:,}")
        c3.metric("External outgoing", f"{stats['external_outgoing']:,}")
        c4.metric("Self-transfers", f"{stats['self_transfers']:,}")
        st.caption("Non-overlapping categories sum to the unique-transaction count. "
                   "A transfer between the selected account and itself appears once in the total.")
        if selected_currency:
            v1, v2, v3 = st.columns(3)
            v1.metric(f"Incoming ({selected_currency})", _amount(stats["incoming_amount"]))
            v2.metric(f"Outgoing ({selected_currency})", _amount(stats["outgoing_amount"]))
            v3.metric("Directional counts (in / out)",
                      f"{stats['incoming']:,} / {stats['outgoing']:,}")
            st.caption("Incoming/outgoing amounts use the receiving/payment currency respectively. "
                       "They include qualifying self-transfers and should not be interpreted as net flows.")
        st.divider()
        tab_timeline, tab_cp, tab_alerts, tab_self, tab_ledger = st.tabs([
            "Activity timeline", "Counterparties", "Pattern alerts", "Self-transfers", "Transaction ledger"
        ])
        with tab_timeline:
            _timeline_tab(fin, account_id, start, end, selected_currency)
        with tab_cp:
            _counterparties_tab(fin, account_id, start, end, selected_currency)
        with tab_alerts:
            alerts_tab(fin, account_id, start, end, selected_currency)
        with tab_self:
            _self_tab(fin, account_id, start, end, selected_currency)
        with tab_ledger:
            st.markdown("#### Filtered transaction ledger")
            st.caption("Use the selected dates and currency to examine underlying transaction IDs.")
            direction = st.selectbox("Ledger direction", ["All", "Incoming", "Outgoing", "Self-transfers"],
                                     key=f"int_ledger_dir_{account_id}")
            ledger, total = transaction_ledger(fin, account_id, start, end, selected_currency,
                                               direction=direction)
            st.caption(f"{total:,} matching rows; displaying {len(ledger):,} most recent.")
            if ledger:
                st.dataframe(pd.DataFrame(ledger), use_container_width=True, hide_index=True)
                _download_csv("Download filtered ledger (CSV)", ledger,
                              "account_transactions.csv", f"int_ledger_csv_{account_id}")
            if total > len(ledger):
                st.warning("Ledger display/export capped at 5,000 rows. Summary SQL remains complete.")
    st.info("Interpretation: transaction frequency, counterparties and rapid-movement candidates "
            "are screening evidence, not proof of money laundering. Customer profile and transaction "
            "purpose are not established by this synthetic dataset.")

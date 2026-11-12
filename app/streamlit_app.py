"""FinCrime Investigator | Local, read-only Streamlit decision-support prototype.

Launch from the repository root:
    streamlit run app/streamlit_app.py

No labelled outcome data is imported. An LLM call occurs only after user action.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.account_intelligence import account_intelligence_view
from app.ui_helpers import (DISPLAY_NAMES, RETRIEVERS, as_download, benchmark_figures,
                            json_at, jsonl_at, network_figure, patterns_summary,
                            retrieval_table, scores_at, target_scores)
from src.baseline import investigate
from src.functions import FinancialTools
from src.graph import summarize_network

SUBSET = ROOT / "data/processed/investigation_subset_2hop_demo.parquet"
MANIFEST = ROOT / "data/processed/subset_manifest_2hop_demo.json"
CORPUS = ROOT / "data/processed/fatf_chunks.jsonl"
EVAL = ROOT / "data/evaluation"

st.set_page_config(page_title="FinCrime Investigator", page_icon="🔎", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top:1.25rem; padding-bottom:3.2rem; max-width:1450px}
h1, h2, h3 {letter-spacing:-0.025em}
div[data-testid="stMetric"] {border:1px solid rgba(128,146,163,.2); border-radius:12px;
                               padding:12px 14px; background:rgba(113,142,161,.045)}
[data-testid="stSidebar"] {border-right:1px solid rgba(128,146,163,.15)}
small, .muted-note {color:#8193a7}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_manifest() -> dict:
    return json_at(MANIFEST) or {}


@st.cache_data(show_spinner=False)
def read_fatf_chunks() -> list[dict]:
    return jsonl_at(CORPUS)


def requirement_guard() -> None:
    if not SUBSET.exists() or not MANIFEST.exists():
        st.error("Investigation data is missing. Run `python -m src.ingest` and `python -m src.prepare_data` in the project root.")
        st.stop()


def caption_note() -> None:
    st.caption("Synthetic IBM AML transactions · Read-only analysis · Heuristics are not a determination of criminal conduct · No case labels exposed")


def report_metrics(report: dict) -> None:
    stats = report.get("statistics", {})
    patterns = report.get("pattern_findings", [])
    priority = report.get("review_priority") or report.get("priority", "not_assessable")
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{stats.get('total_unique_transactions', 0):,}")
    b.metric("Incoming", f"{stats.get('incoming_transactions', 0):,}")
    c.metric("Outgoing", f"{stats.get('outgoing_transactions', 0):,}")
    d.metric("Heuristic priority", priority.upper())
    st.caption("Incoming and outgoing counts may both include self-transfers. These are counts, not disjoint partitions.")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Patterns detected")
        if patterns:
            st.dataframe(pd.DataFrame(patterns_summary(patterns)), width="stretch", hide_index=True)
        else:
            st.info("No supported heuristic patterns detected in this local subset.")
    with right:
        st.markdown("#### Period and source")
        period = report.get("investigation_period") or {}
        st.write(f"**Earliest:** {period.get('start', stats.get('start', '—'))}")
        st.write(f"**Latest:** {period.get('end', stats.get('end', '—'))}")
        st.caption("Counts and monetary totals originate from parameterized DuckDB queries, not LLM calculations.")
    st.markdown("#### Currency-separated flows")
    c1, c2 = st.columns(2)
    for col, key, label, nkey, amount in ((c1, "incoming_by_currency", "Incoming", "n", "total_received"),
                                          (c2, "outgoing_by_currency", "Outgoing", "n", "total_paid")):
        with col:
            rows = stats.get(key, [])
            st.markdown(f"**{label}**")
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df[["currency", nkey, amount]], width="stretch", hide_index=True)
            else:
                st.caption("No transactions in this direction.")
    st.caption("Do not add nominal monetary amounts across currencies; no FX conversion is performed.")


def report_details(report: dict, engine: str) -> None:
    commentary = report.get("llm_analyst_commentary_unverified", "")
    if engine != "Deterministic baseline":
        st.markdown("#### Analyst commentary (LLM draft — unverified)")
        if commentary.strip():
            st.markdown(str(commentary).replace("$", r"\$"))
        else:
            st.warning("No final LLM commentary was returned. Deterministic evidence fields remain available.")
        citations = report.get("guidance_citations", [])
        st.markdown("#### FATF guidance references")
        if citations:
            for c in citations:
                st.markdown(f"**{c.get('title', 'FATF publication')}**, PDF p. {c.get('page_number', '?')} · `{c.get('chunk_id', '?')}`")
                if c.get("source_url", "").startswith("https://"):
                    st.link_button("Open official source PDF", c["source_url"])
        else:
            st.caption("No FATF citations recorded for this run. Guidance is not evidence of wrongdoing.")
        with st.expander("Agent tool trace and execution metadata"):
            st.json({"tool_trace": report.get("tool_trace", []),
                     "execution": report.get("execution", {}),
                     "evidence_validation": report.get("evidence_validation", {})})
    with st.expander("Limitations and recommended next steps", expanded=True):
        for item in report.get("limitations", []):
            st.markdown(f"- {item}")
        for item in report.get("recommended_next_steps", []):
            st.markdown(f"- **Next step:** {item}")
    st.download_button("Download evidence-bound JSON report", data=as_download(report),
                       file_name=f"fincrime_report_{report.get('account_id','account').replace(':','_')}.json",
                       mime="application/json")


def investigation_workspace(account: str) -> None:
    st.subheader("Investigation workspace")
    st.write("Investigate an account using deterministic SQL evidence or a bounded LangGraph agent.")
    with st.form("investigation_form"):
        a, b, c = st.columns([2.2, 1.4, 1])
        with a:
            engine = st.selectbox("Investigation engine", ["Deterministic baseline", "Agentic RAG", "Tools-only agent"])
        with b:
            method = st.selectbox("FATF retrieval mode", RETRIEVERS, index=2, format_func=lambda x: DISPLAY_NAMES[x],
                                  help="Only used by Agentic RAG. Hybrid is the recommended default.")
        with c:
            max_calls = st.number_input("Agent tool budget", min_value=1, max_value=8, value=5, step=1)
        run_clicked = st.form_submit_button("Run investigation", type="primary", width="stretch")
    st.caption("The model is called only if you click Run with an agent selected. It uses the model/key from your local .env and may incur API charges.")

    if run_clicked:
        try:
            with st.spinner("Building an evidence-bound report..."):
                if engine == "Deterministic baseline":
                    with FinancialTools(SUBSET) as fin:
                        result = investigate(account, fin)
                elif engine == "Agentic RAG":
                    from src.agent import run_investigation
                    result = run_investigation(account, method=method, max_calls=int(max_calls), subset_path=SUBSET)
                else:
                    from src.tools_only_agent import run_tools_only
                    result = run_tools_only(account, max_calls=int(max_calls), subset_path=SUBSET)
            st.session_state["case_result"] = {"account": account, "engine": engine, "report": result}
        except Exception as exc:
            st.error(f"Investigation could not finish: {type(exc).__name__}: {exc}")
            st.caption("Check .env, local indexes and installed agent requirements. Previously saved reports have not been replaced.")

    current = st.session_state.get("case_result")
    if current and current["account"] == account:
        report = current["report"]
        st.caption(f"Last run: {current['engine']} · Account: {account}")
        if report.get("status") != "completed":
            st.warning(report.get("reason", "The selected account has insufficient evidence in the subset."))
        else:
            report_metrics(report)
            report_details(report, current["engine"])
    else:
        st.info("Select an investigation engine and click Run. No LLM calls are made automatically.")

    st.markdown("#### Transaction ledger")
    count = st.slider("Maximum transaction rows to display", min_value=25, max_value=500, step=25, value=100)
    with FinancialTools(SUBSET) as fin:
        txs = fin.get_account_transactions(account, limit=count)
    if not txs["complete"]:
        st.caption(f"Showing {txs['returned']} of {txs['total']} available rows; exact totals above come from SQL aggregates.")
    if txs["transactions"]:
        st.dataframe(pd.DataFrame(txs["transactions"]), width="stretch", hide_index=True)
        st.download_button("Download displayed transactions (CSV)",
                           data=pd.DataFrame(txs["transactions"]).to_csv(index=False).encode("utf-8"),
                           file_name="displayed_transactions.csv", mime="text/csv")
    else:
        st.info("No transactions found for this account.")


def network_explorer(account: str) -> None:
    st.subheader("Transaction network")
    st.caption("Nodes represent synthetic accounts. Lines represent directed relationships for visualization; use the ledger for actual transaction direction, timestamps and currencies.")
    hops = st.radio("Graph depth", [1, 2], index=0, horizontal=True)
    with st.spinner("Exploring a bounded network..."):
        with FinancialTools(SUBSET) as fin:
            raw = fin.explore_transaction_network(account, hops=hops, max_edges=3000)
    if not raw.get("complete"):
        st.warning(raw.get("error", "Incomplete graph: graph-derived values are withheld."))
        return
    graph_summary = summarize_network(raw)
    a, b, c = st.columns(3)
    a.metric("Accounts", raw.get("node_count", 0))
    b.metric("Transaction edges", raw.get("edge_count", 0))
    c.metric("Reciprocal pairs", len(graph_summary.get("reciprocal_pairs", [])))
    fig, note = network_figure(raw, account)
    if note:
        st.warning(note)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    st.caption("Self-transfers are counted but omitted as lines. Edge direction is not encoded in this layout. The graph alone cannot establish movement of the same funds.")
    x, y = st.columns(2)
    with x:
        st.markdown("#### Reciprocal account pairs")
        if graph_summary.get("reciprocal_pairs"):
            st.dataframe(pd.DataFrame(graph_summary["reciprocal_pairs"]), width="stretch", hide_index=True)
        else:
            st.info("No reciprocal pair detected in this bounded network.")
    with y:
        st.markdown("#### Short directed cycles")
        if graph_summary.get("short_cycles_up_to_4"):
            st.dataframe(pd.DataFrame(graph_summary["short_cycles_up_to_4"]), width="stretch", hide_index=True)
        else:
            st.info("No short cycle found, or cycle search could not be completed.")
    with st.expander("NetworkX metadata and completeness limitations"):
        st.json(graph_summary)


def search_fatf(query: str, method: str, k: int) -> list[dict]:
    from src.rag import bm25_search, dense_search
    corpus = read_fatf_chunks()
    if method == "bm25":
        return bm25_search(query, corpus, k=k)
    if method == "dense":
        return dense_search(query, k=k)
    from src.retrieval_advanced import hybrid_search, reranked_search
    if method == "hybrid":
        return hybrid_search(query, corpus, k=k)
    return reranked_search(query, corpus, k=k)


def evidence_explorer(account: str) -> None:
    st.subheader("FATF evidence explorer")
    st.write("Search primary AML guidance with the same retrieval methods evaluated in the benchmark.")
    doc_tab, tx_tab = st.tabs(["Guidance search", "Transaction lookup"])
    with doc_tab:
        if not CORPUS.exists():
            st.warning("FATF corpus is missing. Run `python -m src.rag download` and `python -m src.rag ingest`.")
        else:
            st.caption(f"Indexed passages: {len(read_fatf_chunks())}. These documents provide general guidance, not account-specific evidence.")
            with st.form("guidance_form"):
                query = st.text_input("Question", value="What transaction activity could indicate trade-based money laundering?")
                l, r = st.columns(2)
                with l:
                    method = st.selectbox("Retriever", RETRIEVERS, index=2, format_func=lambda m: DISPLAY_NAMES[m])
                with r:
                    k = st.slider("Number of source passages", 1, 8, 5)
                clicked = st.form_submit_button("Search FATF corpus", type="primary")
            if clicked:
                if not query.strip():
                    st.warning("Enter a question first.")
                else:
                    try:
                        with st.spinner("Retrieving FATF passages..."):
                            hits = search_fatf(query.strip(), method, k)
                        st.session_state["fatf_result"] = {"query": query.strip(), "method": method, "hits": hits}
                    except Exception as exc:
                        st.error(f"Retrieval failed: {type(exc).__name__}: {exc}")
            found = st.session_state.get("fatf_result")
            if found:
                st.caption(f"Results for '{found['query']}' · {DISPLAY_NAMES[found['method']]} · Ranked evidence, not a generated answer")
                for i, hit in enumerate(found["hits"], 1):
                    with st.expander(f"{i}. {hit.get('title','FATF guidance')} · PDF p. {hit.get('page_number', '?')}", expanded=i == 1):
                        st.code(hit.get("chunk_id", ""), language=None)
                        st.write(hit.get("text", ""))
                        if str(hit.get("source_url", "")).startswith("https://"):
                            st.link_button("Open official PDF", hit["source_url"])
                        st.caption("PDF page numbers are as stored in the local extracted corpus.")
    with tx_tab:
        st.caption("Search the unlabeled investigation subset only. No laundering labels are read.")
        tx_id = st.text_input("Exact transaction ID", placeholder="TX-000000496668").strip()
        if st.button("Find transaction", disabled=not tx_id):
            with FinancialTools(SUBSET) as fin:
                result = fin._records("""
                    SELECT tx_id, timestamp, from_account_id, to_account_id,
                           amount_received, receiving_currency, amount_paid,
                           payment_currency, payment_format
                    FROM tx WHERE tx_id = ? LIMIT 1
                """, [tx_id])
            if result:
                st.dataframe(pd.DataFrame(result), width="stretch", hide_index=True)
                if result[0]["from_account_id"] != account and result[0]["to_account_id"] != account:
                    st.caption("This transaction is not directly connected to the sidebar account.")
            else:
                st.warning("Transaction ID not found in the local inference subset.")


def dashboard(manifest: dict) -> None:
    st.subheader("Dataset and activity dashboard")
    st.caption("This is an intentionally selected synthetic demo subset—not a representative AML detection population.")
    cases = manifest.get("cases", [])
    a, b, c = st.columns(3)
    a.metric("Demo case accounts", manifest.get("n_cases", len(cases)))
    b.metric("Subset transactions", manifest.get("n_transactions", "—"))
    c.metric("FATF passages", len(read_fatf_chunks()))
    with FinancialTools(SUBSET) as fin:
        daily = fin._records("""
            SELECT CAST(date_trunc('day', timestamp) AS DATE) AS day,
                   count(*) AS transactions FROM tx GROUP BY 1 ORDER BY 1
        """, [])
        currencies = fin._records("""
            SELECT payment_currency AS currency, count(*) AS transactions
            FROM tx GROUP BY 1 ORDER BY transactions DESC LIMIT 12
        """, [])
        account_rows = [fin.account_statistics(case["account_id"]) for case in cases]
    x, y = st.columns([1.7, 1])
    with x:
        st.markdown("#### Daily transaction activity")
        if daily:
            df = pd.DataFrame(daily)
            df["day"] = pd.to_datetime(df["day"])
            st.plotly_chart(px.line(df, x="day", y="transactions", markers=True), width="stretch")
    with y:
        st.markdown("#### Transactions by payment currency")
        if currencies:
            st.plotly_chart(px.bar(pd.DataFrame(currencies), x="currency", y="transactions"), width="stretch")
    st.markdown("#### Case account statistics")
    if account_rows:
        df = pd.DataFrame(account_rows)
        st.dataframe(df[["account_id", "total_unique_transactions", "incoming_transactions",
                         "outgoing_transactions", "start", "end"]], width="stretch", hide_index=True)
    st.info("Activity and sample composition describe the local subset. They are not population fraud rates, model precision or criminal-risk probabilities.")


def benchmark_lab() -> None:
    st.subheader("Benchmark lab")
    st.write("Development and held-out metrics from saved benchmark artifacts. Opening this view does not rerun models.")
    ret_dev = json_at(EVAL / "benchmark_retrieval_metrics_dev.json")
    ret_test = json_at(EVAL / "benchmark_retrieval_metrics_test.json")
    ret_tab, qa_tab, test_tab = st.tabs(["Retrieval comparison", "Investigation agents", "Held-out evaluation"])
    with ret_tab:
        if not ret_dev:
            st.info("Run `python -m src.benchmark_retrieval score --split dev` to create retrieval metrics.")
        else:
            st.caption(f"Development questions: {ret_dev.get('total_questions', '?')} · Gold annotations complete: {ret_dev.get('complete_annotation', False)}")
            st.dataframe(pd.DataFrame(retrieval_table(ret_dev)), width="stretch", hide_index=True)
            f1, f2 = benchmark_figures(ret_dev)
            st.plotly_chart(f1, width="stretch")
            st.markdown("#### Latency trade-off")
            st.plotly_chart(f2, width="stretch")
            st.caption("Warm query time excludes model loading. Reranking scores are not probabilities; no end-to-end answer correctness is measured here.")
    with qa_tab:
        original = json_at(EVAL / "benchmark_qa/human_summary_dev.json")
        v2 = json_at(EVAL / "benchmark_qa_v2_stats/human_summary_dev.json")
        if original:
            rows = []
            for arm, info in original.get("per_arm", {}).items():
                if not info.get("answers_in_sheet"):
                    continue
                item = info.get("factual_correct", {})
                rows.append({"Architecture": arm, "Questions": info["answers_in_sheet"],
                             "Factual graded": item.get("n"), "Factual correct (%)":
                             round(100*item["mean"], 1) if item.get("mean") is not None else None,
                             "Mean latency (s)": info.get("mean_latency_seconds")})
            st.markdown("#### Initial development benchmark")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Architectures handle different question sets. The overall percentages above are not a matched-task ranking.")
        else:
            st.info("Initial QA summary not found at `data/evaluation/benchmark_qa/human_summary_dev.json`.")
        if v2:
            st.markdown("#### Fixed-account-statistics regression")
            old_rows = scores_at(EVAL / "benchmark_qa/human_scores_dev.csv")
            new_rows = scores_at(EVAL / "benchmark_qa_v2_stats/human_scores_dev.csv")
            compare = []
            for arm in ("tools_only", "agentic_rag"):
                old_hit, old_n = target_scores(old_rows, arm)
                new_hit, new_n = target_scores(new_rows, arm)
                compare.append({"Architecture": arm, "Before": f"{old_hit}/{old_n}" if old_n else "N/A",
                                "After": f"{new_hit}/{new_n}" if new_n else "N/A"})
            st.dataframe(pd.DataFrame(compare), width="stretch", hide_index=True)
            st.caption("Only ACC-02, ACC-03 and ACC-05 are compared, using graded development predictions.")
        else:
            st.info("Version 2 regression summary not found yet.")
    with test_tab:
        if not ret_test:
            st.info("Held-out retrieval results are not available yet. Annotate RET-04, RET-08 and RET-12 before running the test split.")
        else:
            st.caption(f"Test questions: {ret_test.get('total_questions', '?')} · Annotation complete: {ret_test.get('complete_annotation', False)}")
            st.dataframe(pd.DataFrame(retrieval_table(ret_test)), width="stretch", hide_index=True)
        st.warning("Never choose retrieval hyperparameters or adjust prompts using held-out test results. The test contains only three questions, so results are indicative.")


with st.sidebar:
    st.markdown("### 🔎 FinCrime Investigator")
    st.caption("AML evidence + bounded agent")
    view = st.radio("Workspace", ["Investigation", "Account Intelligence", "Network", "FATF Evidence", "Dashboard", "Benchmark Lab"])
    st.divider()
    manifest = read_manifest()
    cases = [c.get("account_id") for c in manifest.get("cases", []) if c.get("account_id")]
    account = st.selectbox("Case account", cases, disabled=not cases) if cases else None
    st.divider()
    st.caption("**Model access:** " + ("API key configured" if os.getenv("OPENAI_API_KEY") else "uses local .env when an agent runs"))
    st.caption("**Financial data:** " + ("ready" if SUBSET.exists() else "missing"))
    st.caption("**FATF corpus:** " + ("ready" if CORPUS.exists() else "missing"))

st.title("FinCrime Investigator")
caption_note()

if view in ("Investigation", "Network", "Dashboard", "FATF Evidence"):
    requirement_guard()
if view in ("Investigation", "Network", "FATF Evidence") and not account:
    st.warning("No public demo accounts in subset_manifest.json. Run dataset preparation.")
    st.stop()

if view == "Investigation":
    investigation_workspace(account)
elif view == "Account Intelligence":
    if not account:
        st.warning("Choose an account from the sidebar.")
        st.stop()
    if not SUBSET.is_file():
        st.error("The selected transaction Parquet file is missing.")
        st.stop()
    account_intelligence_view(account, SUBSET)
elif view == "Network":
    network_explorer(account)
elif view == "FATF Evidence":
    evidence_explorer(account)
elif view == "Dashboard":
    dashboard(manifest)
else:
    benchmark_lab()

st.divider()
st.caption("Research prototype · Synthetic data only · FATF guidance is not legal advice · Investigative observations require human review")

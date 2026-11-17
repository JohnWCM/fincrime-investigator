"""Self-contained Streamlit presentation for the FinCrime Investigator.

No database, retrieval model, or LLM calls are made while viewing slides.
The launch buttons navigate to EXISTING app workspaces via Streamlit callbacks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

GITHUB = "https://github.com/JohnWCM/fincrime-investigator"

# Stable slides so content and progress remain consistent across reruns.
SLIDES = (
    "Welcome",
    "The investigation problem",
    "The product",
    "System architecture",
    "Retrieval approaches",
    "Evaluation evidence",
    "What I engineered",
    "Limitations & next steps",
    "Live demonstration",
)

# These identifiers refer to the original IBM AML demo, NOT the generated
# fictional quick-start fixture. Availability is checked against the manifest.
DEMO_CASES = (
    ("023::800167090", "Pattern alerts", "Account Intelligence", "Repeated outgoing payments and concentration"),
    ("00701::800A8E530", "Two-hop network", "Network", "5 to 38 accounts when comparing hop depth"),
    ("021575::80491D7A0", "Agent investigation", "Investigation", "Transaction evidence + FATF guidance + tool trace"),
)

CSS = """
<style>
.presentation-hero {
    border: 1px solid rgba(89,206,196,.30); border-radius: 22px;
    background: linear-gradient(135deg,#0d192d,#172942 64%,#12343e);
    padding: 2.3rem 2.5rem; margin: .5rem 0 1.1rem 0;
}
.presentation-hero .eyebrow { color:#6fe0d3; font-size:.75rem; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase; }
.presentation-hero h1 { color:#f2f7ff !important; font-size:2.55rem;
    line-height:1.14; margin:.65rem 0 .65rem; font-weight:750; }
.presentation-hero p { color:#becde0; font-size:1.03rem; margin:.5rem 0; }
.presentation-kicker { font-size:.83rem; color:#7992a9; letter-spacing:.1em;
    font-weight:650; text-transform:uppercase; margin-top:.7rem; }
.presentation-card { border:1px solid rgba(116,149,173,.26); border-radius:15px;
    padding:1.15rem 1.3rem; min-height:128px; margin-bottom:.5rem;
    background:rgba(77,106,137,.055); }
.presentation-card strong { font-size:1rem; }
.presentation-card p { color:#93a5b8; font-size:.91rem; margin:.45rem 0 0; }
.presentation-note { color:#8da1b7; font-size:.89rem; }
</style>
"""


def _set_slide(n: int) -> None:
    st.session_state["presentation_slide"] = max(0, min(n, len(SLIDES) - 1))


def _jump(view: str, account: str | None = None) -> None:
    """Callback runs before sidebar widgets are rendered on the next rerun."""
    st.session_state["workspace_view"] = view
    if account is not None:
        st.session_state["case_account"] = account


def _card(title: str, copy: str) -> None:
    # Static, developer-controlled strings only; no untrusted values in HTML.
    st.markdown(
        f'<div class="presentation-card"><strong>{title}</strong><p>{copy}</p></div>',
        unsafe_allow_html=True,
    )


def _image(root: Path, filename: str, label: str) -> None:
    image_path = root / "docs" / "screenshots" / filename
    if image_path.is_file():
        st.image(str(image_path), caption=label, use_container_width=True)
    else:
        st.caption(f"Screenshot not available locally: docs/screenshots/{filename}")


def _retrieval_rows(root: Path) -> tuple[list[dict], int | None]:
    """Read published retrieval metrics; never manufacture missing scores."""
    path = root / "data" / "evaluation" / "benchmark_retrieval_metrics_dev.json"
    if not path.is_file():
        return [], None
    try:
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
        metrics = summary.get("metrics", {})
        labels = [("bm25", "BM25"), ("dense", "Dense"),
                  ("hybrid", "Hybrid RRF"), ("rerank", "Hybrid + rerank")]
        rows = []
        for key, name in labels:
            current = metrics.get(key)
            if isinstance(current, dict):
                recall, mrr = current.get("recall_at_5"), current.get("mrr_at_5")
                if recall is not None and mrr is not None:
                    rows.append({"Method": name, "Recall@5": round(100 * float(recall), 1),
                                 "MRR@5": round(100 * float(mrr), 1)})
        return rows, summary.get("total_questions")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return [], None


def _content(index: int, root: Path, cases: list[str]) -> None:
    if index == 0:
        st.markdown('''<section class="presentation-hero">
          <div class="eyebrow">Ironhack · AI &amp; Machine Learning final project</div>
          <h1>FinCrime Investigator</h1>
          <p>From transactions to evidence-grounded investigations.</p>
          <p>SQL analytics · Graph investigation · Agentic RAG · Evaluated retrieval</p>
        </section>''', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        with a: _card("Evidence first", "Transaction facts originate in deterministic SQL queries.")
        with b: _card("Context second", "Network relationships and FATF text provide investigative context.")
        with c: _card("LLM last", "A bounded agent synthesizes findings, with uncertainty retained.")
        st.info("Research prototype using synthetic transaction data; not an automated finding of criminal conduct.")

    elif index == 1:
        st.markdown('<div class="presentation-kicker">The problem</div>', unsafe_allow_html=True)
        st.header("A suspicious transaction is rarely obvious in isolation")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Volume", "Even a small account can generate a long transaction history.")
        with c2: _card("Relationships", "Direct counterparties reveal only part of the wider network.")
        with c3: _card("Documentation", "Relevant AML guidance is spread across long, complex publications.")
        st.markdown("#### Design question")
        st.markdown("**Can an investigation assistant combine exact financial evidence and documentary context, without confusing a heuristic signal with proof?**")

    elif index == 2:
        st.markdown('<div class="presentation-kicker">The solution</div>', unsafe_allow_html=True)
        st.header("One workspace, four complementary analytical layers")
        a, b = st.columns(2)
        with a:
            _card("01 · Transaction intelligence", "Account-level statistics, timelines, counterparties and evidence drill-down.")
            _card("03 · FATF RAG", "BM25, semantic, hybrid and reranked regulatory evidence retrieval.")
        with b:
            _card("02 · Graph investigation", "Directed links, reciprocal relationships, short cycles and two-hop exploration.")
            _card("04 · LangGraph agent", "Bounded tool calls, cited narrative and downloadable structured report.")
        _image(root, "investigation.png", "FinCrime Investigator — investigation workspace")

    elif index == 3:
        st.markdown('<div class="presentation-kicker">Architecture</div>', unsafe_allow_html=True)
        st.header("Deterministic analysis underneath generative synthesis")
        st.markdown("#### Investigator → Streamlit → LangGraph")
        a, b, c, d = st.columns(4)
        with a: _card("DuckDB", "Counts, totals, dates and transaction rows")
        with b: _card("NetworkX", "Direct and second-hop account connections")
        with c: _card("Heuristic rules", "Repeated payments and counterparty concentration")
        with d: _card("FATF retrieval", "Document chunks with traceable page references")
        st.markdown("#### Evidence → structured investigation report")
        st.success("The LLM does not calculate the financial facts; it coordinates tools and drafts the interpretation.")
        st.caption("No model execution occurs simply by opening Presentation Mode.")

    elif index == 4:
        st.markdown('<div class="presentation-kicker">Retrieval design</div>', unsafe_allow_html=True)
        st.header("Why evaluate multiple RAG retrieval strategies?")
        a, b = st.columns(2)
        with a:
            _card("BM25", "Lexical relevance: specific phrases and terminology.")
            _card("Dense embeddings", "Semantic relevance beyond exact word matches.")
        with b:
            _card("Hybrid RRF", "Combine keyword and semantic candidate rankings.")
            _card("Cross-encoder reranking", "Rescore promising candidates for better ordering.")
        st.markdown("**Selection criterion:** relevant FATF passages ranked high enough to support investigation writing.")
        st.caption("The corpus is regulatory guidance, not transaction-specific evidence.")

    elif index == 5:
        st.markdown('<div class="presentation-kicker">Evaluation</div>', unsafe_allow_html=True)
        st.header("Measured behavior, with small-sample caveats")
        rows, n = _retrieval_rows(root)
        if rows:
            st.caption(f"Retrieval development benchmark · {n or '?'} questions · results loaded from the saved evaluation file")
            df = pd.DataFrame(rows).set_index("Method")
            st.bar_chart(df, use_container_width=True, height=280, y_label="Score (%)")
        else:
            st.warning("Retrieval metrics not found. This slide intentionally does not invent scores.")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("SQL-grounded benchmark", "64 / 64", help="32 task-specific questions across two configurations, not AML detection accuracy.")
        with c2:
            st.metric("Automated tests", "34 passed", help="Last recorded project test run; rerun tests for current code.")
        st.caption("Original IBM/FATF evaluations. These results do not validate any newly generated fictional demo dataset. The retrieval held-out split had only three questions.")

    elif index == 6:
        st.markdown('<div class="presentation-kicker">Engineering choices</div>', unsafe_allow_html=True)
        st.header("The hard part was making plausible answers auditable")
        a, b = st.columns(2)
        with a:
            _card("Grounding", "Put transaction facts behind DuckDB tools instead of unconstrained text generation.")
            _card("Guardrails", "Bound tool calls; preserve incomplete evidence and uncertainty.")
        with b:
            _card("Retrieval tests", "Compare passage relevance and ranking, not just response fluency.")
            _card("Regression checks", "Revisit formerly incorrect statistics questions after code fixes.")
        st.markdown("**Key lesson:** a fluent report is not the same thing as a correct investigation.")

    elif index == 7:
        st.markdown('<div class="presentation-kicker">Limitations & next steps</div>', unsafe_allow_html=True)
        st.header("What the prototype can—and cannot—claim")
        st.warning("Not a production AML detection system. Heuristic alerts are investigative leads; they do not establish money laundering.")
        a, b = st.columns(2)
        with a:
            st.markdown("**Known limitations**")
            st.markdown("- Synthetic data and curated investigation subsets\n- Small retrieval test set\n- No independently estimated false-positive rate\n- Generated narratives require human review")
        with b:
            st.markdown("**Next engineering priorities**")
            st.markdown("- Independent labeled-case evaluation\n- Automated citation and amount validation\n- Clean-clone reproducibility checks\n- Production access controls and monitoring")
        st.success("Takeaway: evidence-grounded investigation is more useful than an unverified risk label.")

    elif index == 8:
        st.markdown('<div class="presentation-kicker">Now live</div>', unsafe_allow_html=True)
        st.header("Let the evidence speak: open a real case")
        st.caption("Buttons below open EXISTING workspace pages. No transactions are simulated inside Presentation Mode.")
        if not cases:
            st.warning("No prepared case manifest found. You can present the slides without data, but live case buttons are disabled.")
        for acc, title, view, desc in DEMO_CASES:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{title}** · `{acc}`")
                st.caption(desc)
            with c2:
                st.button(f"Open {view}", key=f"pm_jump_{view}_{acc}",
                          use_container_width=True, disabled=acc not in cases,
                          on_click=_jump, args=(view, acc))
            if acc not in cases and cases:
                st.caption("This IBM account is not present in the currently selected dataset.")
        st.link_button("GitHub repository", GITHUB)
        if not any(a in cases for a, *_ in DEMO_CASES) and cases:
            st.info("These historical demo accounts are not in the current manifest. Select an available account below.")
            a, b = st.columns([2, 1])
            with a: selected = st.selectbox("Available case", cases, key="pm_available_case")
            with b:
                st.button("Open selected case", on_click=_jump, args=("Investigation", selected),
                          use_container_width=True, key="pm_jump_fallback")
        st.markdown("**Presentation complete → live demonstration → Q&A**")


def render_presentation_mode(root: Path, cases: list[str] | None = None) -> None:
    """Main entry point from streamlit_app.py; safe without prepared data or API keys."""
    root = Path(root)
    cases = list(cases or [])
    st.markdown(CSS, unsafe_allow_html=True)
    current = int(st.session_state.get("presentation_slide", 0))
    current = max(0, min(current, len(SLIDES) - 1))
    st.session_state["presentation_slide"] = current
    st.caption(f"PRESENTATION MODE  •  {current + 1:02d}/{len(SLIDES):02d}  •  {SLIDES[current]}")
    st.progress((current + 1) / len(SLIDES))
    _content(current, root, cases)
    st.divider()
    left, center, right = st.columns([1, 2, 1])
    with left:
        st.button("← Previous", key="pm_previous", disabled=current == 0,
                  use_container_width=True, on_click=_set_slide, args=(current - 1,))
    with center:
        st.caption("Use Next to advance. Everything on the first eight screens is precomputed.")
    with right:
        st.button("Next →" if current < len(SLIDES) - 2 else "Live demo →",
                  key="pm_next", disabled=current == len(SLIDES) - 1,
                  type="primary", use_container_width=True,
                  on_click=_set_slide, args=(current + 1,))
    with st.expander("Presenter cue (private rehearsal notes; hide during the talk)"):
        cues = (
            "Opening: I built an evidence-grounded assistant, not an automatic criminal-risk classifier.",
            "Explain why transaction rows, graph links and regulatory guidance must be examined together.",
            "Introduce four analytical capabilities. The LLM does not invent numeric facts.",
            "Follow one fact through deterministic tools, LangGraph and cited reporting.",
            "Distinguish lexical, semantic, fused and reranked retrieval.",
            "Mention all denominators: 9 dev / 3 test retrieval; 32 SQL questions across two arms.",
            "Emphasize engineering tradeoffs and regression checks, not feature count.",
            "Acknowledge the gap between verified financial facts and validated AML outcomes.",
            "Launch the three known IBM demo cases in order; use a saved agent report if API is slow.",
        )
        st.caption(cues[current])

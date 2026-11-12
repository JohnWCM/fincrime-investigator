"""Pure, read-only display helpers for the FinCrime Investigator Streamlit UI.

No access to synthetic laundering labels or private evaluation files.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go

RETRIEVERS = ("bm25", "dense", "hybrid", "rerank")
DISPLAY_NAMES = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid RRF",
    "rerank": "Hybrid + rerank",
}
TARGETED_IDS = frozenset({"ACC-02", "ACC-03", "ACC-05"})


def json_at(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def jsonl_at(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scores_at(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_scores(rows: list[dict], arm: str) -> tuple[int, int]:
    """Count *graded* factual answers on the three pre-identified failed dev prompts."""
    matched = [row for row in rows
               if row.get("question_id") in TARGETED_IDS and row.get("arm") == arm
               and (row.get("factual_correct_0_1") or "").strip() in {"0", "1"}]
    return sum(row["factual_correct_0_1"].strip() == "1" for row in matched), len(matched)


def retrieval_table(summary: dict) -> list[dict]:
    methods = summary.get("metrics", {})
    return [{
        "Method": DISPLAY_NAMES[method],
        "Recall@5": _pct(methods[method].get("recall_at_5")),
        "MRR@5": _pct(methods[method].get("mrr_at_5")),
        "Hit@5": _pct(methods[method].get("hit_at_5")),
        "Warm latency (ms)": methods[method].get("mean_latency_ms_warm"),
        "Graded questions": methods[method].get("verified_questions_scored", 0),
    } for method in RETRIEVERS if method in methods]


def _pct(value: Any) -> float | None:
    return None if value is None else round(100.0 * float(value), 2)


def benchmark_figures(summary: dict) -> tuple[go.Figure, go.Figure]:
    data = retrieval_table(summary)
    names = [d["Method"] for d in data]
    rank = go.Figure()
    for metric in ("Recall@5", "MRR@5", "Hit@5"):
        rank.add_trace(go.Bar(name=metric, x=names, y=[d[metric] for d in data]))
    rank.update_layout(barmode="group", yaxis_title="Score (%)", yaxis_range=[0, 105],
                       legend_orientation="h", margin=dict(l=5, r=5, t=20, b=5), height=390)
    latency = go.Figure(go.Bar(x=names, y=[d["Warm latency (ms)"] for d in data],
                                   marker_color="#278e9d"))
    latency.update_layout(yaxis_title="Mean warm latency (ms; log scale)", yaxis_type="log",
                          margin=dict(l=5, r=5, t=20, b=5), height=330)
    return rank, latency


def network_figure(raw: dict, selected_account: str, max_visible_nodes: int = 70) -> tuple[go.Figure | None, str | None]:
    """Visualize a bounded complete network without pretending displayed subset is complete."""
    if not raw.get("complete"):
        return None, raw.get("error", "The network is incomplete; no plot is shown.")
    graph = nx.MultiDiGraph()
    graph.add_node(selected_account)
    for edge in raw.get("edges", []):
        graph.add_edge(edge["from_account_id"], edge["to_account_id"], key=edge["tx_id"])

    note = None
    if graph.number_of_nodes() > max_visible_nodes:
        # Selected account is always retained; show most connected neighbours.
        others = sorted((n for n in graph.nodes if n != selected_account),
                        key=lambda n: (-graph.degree(n), n))
        visible = {selected_account, *others[:max_visible_nodes - 1]}
        graph = graph.subgraph(visible).copy()
        note = (f"Visualization limited to {max_visible_nodes} most-connected nodes. "
                "This plot is a display subset, not the complete transaction network.")

    layout = nx.spring_layout(graph, seed=42, iterations=70) if graph.number_of_nodes() > 1 else {selected_account: (0, 0)}
    edges_by_pair = Counter((u, v) for u, v in graph.edges())
    edges_x: list[float | None] = []
    edges_y: list[float | None] = []
    for (u, v), count in edges_by_pair.items():
        x1, y1 = layout[u]
        x2, y2 = layout[v]
        if u == v:
            continue  # Self-transfers still appear in counts and transaction tables.
        edges_x.extend([float(x1), float(x2), None])
        edges_y.extend([float(y1), float(y2), None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edges_x, y=edges_y, mode="lines", hoverinfo="skip",
                             line=dict(color="rgba(128,146,163,0.42)", width=1.5), name="Directed pairs (layout only)"))
    nodes = list(graph.nodes)
    degrees = dict(graph.degree())
    fig.add_trace(go.Scatter(
        x=[float(layout[n][0]) for n in nodes], y=[float(layout[n][1]) for n in nodes],
        mode="markers+text", text=["Selected" if n == selected_account else "" for n in nodes],
        textposition="top center", hovertext=[f"{n}<br>Transaction-edge degree: {degrees[n]}" for n in nodes],
        hoverinfo="text", name="Accounts",
        marker=dict(size=[22 if n == selected_account else min(10 + degrees[n], 20) for n in nodes],
                    color=["#12b5cb" if n == selected_account else "#667a96" for n in nodes],
                    line=dict(color="#e0e9f3", width=1.5)),
    ))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=510, showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      hovermode="closest", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig, note


def as_download(report: dict) -> str:
    return json.dumps(report, indent=2, default=str, ensure_ascii=False)


def patterns_summary(patterns: list[dict]) -> list[dict]:
    return [{"Pattern": p.get("pattern", "unknown"),
             "Date": p.get("day", ""),
             "Evidence IDs": ", ".join(map(str, p.get("supporting_transaction_ids", [])[:5])),
             "Counterparties": p.get("distinct_counterparties"),
             "Gap (min)": p.get("gap_minutes")} for p in patterns]

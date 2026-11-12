from app.ui_helpers import benchmark_figures, network_figure, retrieval_table, target_scores


def test_retrieval_table_keeps_none_and_percentage_values():
    data = {"metrics": {"hybrid": {"recall_at_5": 0.7778, "mrr_at_5": 0.7444,
                                      "hit_at_5": 0.8889, "mean_latency_ms_warm": 39.2,
                                      "verified_questions_scored": 9}}}
    rows = retrieval_table(data)
    assert len(rows) == 1
    assert rows[0]["Recall@5"] == 77.78
    assert rows[0]["Warm latency (ms)"] == 39.2
    rank, latency = benchmark_figures(data)
    assert len(rank.data) == 3
    assert latency.layout.yaxis.type == "log"


def test_target_scores_ignore_ungraded_rows_and_other_questions():
    rows = [
        {"question_id": "ACC-02", "arm": "agentic_rag", "factual_correct_0_1": "1"},
        {"question_id": "ACC-03", "arm": "agentic_rag", "factual_correct_0_1": "0"},
        {"question_id": "ACC-05", "arm": "agentic_rag", "factual_correct_0_1": ""},
        {"question_id": "NET-01", "arm": "agentic_rag", "factual_correct_0_1": "1"},
        {"question_id": "ACC-02", "arm": "tools_only", "factual_correct_0_1": "1"},
    ]
    assert target_scores(rows, "agentic_rag") == (1, 2)
    assert target_scores(rows, "tools_only") == (1, 1)


def test_network_figure_rejects_incomplete_graph():
    fig, note = network_figure({"complete": False, "error": "Budget exceeded"}, "A")
    assert fig is None
    assert "Budget" in note


def test_network_figure_complete_graph():
    raw = {"complete": True,
           "edges": [{"tx_id": "TX-1", "from_account_id": "A", "to_account_id": "B"},
                     {"tx_id": "TX-2", "from_account_id": "B", "to_account_id": "A"}]}
    fig, note = network_figure(raw, "A")
    assert fig is not None
    assert note is None
    assert len(fig.data) == 2

"""No API calls or model downloads in these report-contract tests."""
from src.agent import assemble_report


def test_report_never_replaces_derived_values_with_llm_text():
    baseline = {
        "account_id": "BANK::A", "status": "completed", "priority": "monitor",
        "statistics": {"total_unique_transactions": 3},
        "supporting_transaction_ids": ["tx-001"],
        "pattern_findings": [{"pattern": "fan_in", "supporting_transaction_ids": ["tx-001"]}],
        "limitations": [], "recommended_next_steps": [],
    }
    evidence = {
        "guidance": [{"chunk_id": "fatf-abc", "title": "FATF", "page_number": 3,
                      "source_url": "https://example.org/fatf.pdf"}],
        "network": {"complete": True, "node_count": 2},
        "tool_trace": [{"tool": "search_aml_guidance", "result_kind": "guidance"}],
        "tool_errors": [],
    }
    output = assemble_report(baseline, evidence, "1000 transactions, crime proven", "dummy",
                             "hybrid", 2.0, {"input_tokens": 0, "output_tokens": 0}, 5)
    assert output["statistics"]["total_unique_transactions"] == 3
    assert output["supporting_transaction_ids"] == ["tx-001"]
    assert output["review_priority"] == "monitor"
    assert output["guidance_citations"][0]["chunk_id"] == "fatf-abc"
    assert output["evidence_validation"]["model_provided_numbers_or_citations_accepted"] is False
    assert "1000 transactions" not in str(output["statistics"])

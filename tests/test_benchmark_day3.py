"""Offline tests: no OpenAI calls, dataset or model downloads needed."""
import json
from pathlib import Path
import pytest

from src.benchmark_retrieval import (
    QUESTIONS, RetrievalBench, prepare, read_jsonl, retrieval_scores,
    split_output,
)
from src.benchmark_investigation import audit_report


def test_retrieval_metric_values():
    actual = retrieval_scores(['a', 'b'], ['x', 'b', 'z', 'a', 'v'])
    assert actual['recall_at_5'] == 1.0
    assert actual['hit_at_5'] == 1.0
    assert actual['mrr_at_5'] == 0.5
    zero = retrieval_scores(['a'], ['x', 'y'])
    assert zero == {'recall_at_5': 0, 'hit_at_5': 0.0, 'mrr_at_5': 0}


def test_unannotated_question_cannot_score():
    with pytest.raises(ValueError):
        retrieval_scores([], ['a', 'b'])


def test_rrf_deterministic_and_deduplicated():
    res = RetrievalBench.rrf(['a','b','c'], ['b','a','d'], k=4)
    assert res == ['a', 'b', 'c', 'd']  # a/b are tied; lexical tie-break
    assert len(res) == len(set(res))


def test_prepare_is_idempotent_without_overwrite(tmp_path):
    path = tmp_path / 'gold.jsonl'
    prepare(path)
    qs = read_jsonl(path)
    assert len(qs) == 12
    assert sum(q['split'] == 'test' for q in qs) == 3
    assert all(not q['gold_chunk_ids'] for q in qs)
    with pytest.raises(FileExistsError):
        prepare(path)


def test_split_files_not_overwritten():
    assert split_output(Path('x/preds.jsonl'), 'dev') == Path('x/preds_dev.jsonl')
    assert split_output(Path('x/preds.jsonl'), 'test') == Path('x/preds_test.jsonl')


def test_audit_report_checks_provenance_without_claiming_semantics():
    report = {
        'account_id': 'B::A', 'statistics': {
            'total_unique_transactions': 2, 'incoming_transactions': 1,
            'outgoing_transactions': 2, 'start': 'a', 'end': 'b',
        },
        'supporting_transaction_ids': ['TX-1', 'BAD'],
        'guidance_citations': [{'chunk_id': 'fatf1'}, {'chunk_id': 'unknown'}],
        'tool_trace': [{'origin': 'fallback'}, {'tool': 'detect_transaction_patterns'}],
        'llm_analyst_commentary_unverified': 'Review only.',
        'execution': {'elapsed_seconds': 4.5}
    }
    independent = dict(report['statistics'])
    result = audit_report(report, independent, {'fatf1'}, {'TX-1', 'TX-2'})
    assert result['stat_field_exact_match_rate'] == 1.0
    assert result['invalid_supporting_transaction_ids'] == ['BAD']
    assert result['invalid_document_chunk_ids'] == ['unknown']
    assert result['tool_calls_observed'] == 1
    assert result['fallback_calls'] == 1


def test_question_coverage_is_task_specific():
    from src.benchmark_qa import eligible
    account = {'id': 'ACC-01', 'category': 'account'}
    knowledge = {'id': 'KNO-01', 'category': 'knowledge'}
    unsupported = {'id': 'UNS-01', 'category': 'unsupported'}
    network = {'id': 'NET-06', 'category': 'network'}
    assert eligible('baseline', account)
    assert not eligible('baseline', knowledge)
    assert not eligible('baseline', network)
    assert eligible('rag_only', knowledge)
    assert not eligible('rag_only', account)
    assert eligible('tools_only', network)
    assert not eligible('tools_only', knowledge)
    assert all(eligible('agentic_rag', q) for q in (account, knowledge, unsupported, network))


def test_question_grade_sheet_preserves_human_scores(tmp_path, monkeypatch):
    import src.benchmark_qa as mod
    from src.benchmark_retrieval import write_jsonl
    root = tmp_path / 'qa'
    qfile = tmp_path / 'questions.jsonl'
    qs = [{'id': 'ACC-01', 'category': 'account', 'split': 'dev',
           'account_id': 'B::A', 'question': 'Count transactions'}]
    write_jsonl(qfile, qs)
    monkeypatch.setattr(mod, 'ROOT', root)
    monkeypatch.setattr(mod, 'QUESTIONS', qfile)
    p = mod.prediction_path('baseline', 'ACC-01')
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'answer': '2', 'latency_seconds': 0.1}), encoding='utf-8')
    sheet = mod.make_sheet('dev')
    import csv
    with sheet.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows[0]['factual_correct_0_1'] = '1'
    with sheet.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    mod.make_sheet('dev')
    with sheet.open(encoding='utf-8-sig', newline='') as f:
        preserved = list(csv.DictReader(f))
    assert preserved[0]['factual_correct_0_1'] == '1'
    metrics = mod.grade_scores('dev')
    assert metrics['per_arm']['baseline']['factual_correct'] == {'n': 1, 'mean': 1.0}
    assert metrics['per_arm']['rag_only']['factual_correct']['mean'] is None

"""Day 3: reproducible FATF retrieval benchmark.

Run from the repository root. The annotation template contains no invented gold IDs.
Use actual PDF pages and source text to approve gold_chunk_ids independently.

Commands:
    python -m src.benchmark_retrieval prepare
    python -m src.benchmark_retrieval preview
    python -m src.benchmark_retrieval run --split dev
    python -m src.benchmark_retrieval score --split dev
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from typing import Any

from src.rag import load_corpus, tokenize

GOLD = Path('data/evaluation/retrieval_gold.jsonl')
RUNS = Path('data/evaluation/benchmark_retrieval_predictions.jsonl')
SUMMARY = Path('data/evaluation/benchmark_retrieval_metrics.json')

def split_output(path: Path, split: str) -> Path:
    return path.with_name(f'{path.stem}_{split}{path.suffix}')
PREVIEW = Path('data/evaluation/retrieval_candidates.jsonl')
METHODS = ('bm25', 'dense', 'hybrid', 'rerank')

# Four questions from each publication; exactly one held out from each.
# These are test prompts, not claimed quotations or already-verified answers.
QUESTIONS = [
    ('RET-01', 'fatf_banking_rba_2014', 'dev', 'How does the FATF define a risk-based approach for banks?'),
    ('RET-02', 'fatf_banking_rba_2014', 'dev', 'How should a bank identify and assess customer money laundering risk?'),
    ('RET-03', 'fatf_banking_rba_2014', 'dev', 'How should banks apply customer due diligence based on the assessed risk?'),
    ('RET-04', 'fatf_banking_rba_2014', 'test', 'What role does ongoing monitoring play in a bank risk-based AML programme?'),
    ('RET-05', 'fatf_tbml_indicators_2021', 'dev', 'What account and transaction activity could indicate trade-based money laundering?'),
    ('RET-06', 'fatf_tbml_indicators_2021', 'dev', 'What unusual payment changes may be trade-based money laundering risk indicators?'),
    ('RET-07', 'fatf_tbml_indicators_2021', 'dev', 'Which inconsistencies in invoices and trade documents may be risk indicators?'),
    ('RET-08', 'fatf_tbml_indicators_2021', 'test', 'How should analysts interpret a single trade-based money laundering risk indicator?'),
    ('RET-09', 'fatf_financial_investigations_2012', 'dev', 'What is the purpose of a financial investigation?'),
    ('RET-10', 'fatf_financial_investigations_2012', 'dev', 'What is a parallel financial investigation?'),
    ('RET-11', 'fatf_financial_investigations_2012', 'dev', 'Which information sources can support a financial investigation?'),
    ('RET-12', 'fatf_financial_investigations_2012', 'test', 'How can financial investigators cooperate with other competent authorities?'),
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run the earlier preparation step.')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare(path: Path = GOLD) -> None:
    if path.exists():
        raise FileExistsError(f'{path} already exists. It was NOT overwritten; gold labels must be preserved.')
    write_jsonl(path, [{
        'id': qid,
        'doc_id': doc,
        'split': split,
        'question': question,
        'gold_chunk_ids': [],
        'gold_answer': None,
        'annotation_status': 'pending_manual_pdf_verification',
        'annotation_note': '',
    } for qid, doc, split, question in QUESTIONS])
    print(f'Created {path} with {len(QUESTIONS)} unannotated questions (9 dev / 3 test).')
    print('Manually inspect official PDF pages and fill gold_chunk_ids; do not copy model results as gold without verification.')


def retrieval_scores(gold_ids: list[str], returned_ids: list[str], k: int = 5) -> dict[str, float]:
    """Recall uses all manually verified relevant chunk IDs, not just one hit."""
    if not gold_ids:
        raise ValueError('Cannot score a question without manually verified gold chunk IDs')
    gold = set(gold_ids)
    ranked = list(dict.fromkeys(returned_ids[:k]))
    found = len(gold.intersection(ranked))
    first = next((rank for rank, cid in enumerate(ranked, 1) if cid in gold), None)
    return {
        'recall_at_5': found / len(gold),
        'hit_at_5': float(found > 0),
        'mrr_at_5': 0.0 if first is None else 1.0 / first,
    }


class RetrievalBench:
    """Reuse loaded BM25, embedding and cross-encoder models between queries.

    Excludes initial model download/loading from query latency; reports warm latency.
    The search algorithms and fusion weights match the project's retrieval modules.
    """
    def __init__(self, corpus: list[dict]):
        from rank_bm25 import BM25Okapi
        self.corpus = corpus
        self.by_id = {d['chunk_id']: d for d in corpus}
        if len(self.by_id) != len(corpus):
            raise ValueError('Corpus contains duplicate chunk IDs')
        self.bm25 = BM25Okapi([tokenize(d['text']) for d in corpus])
        self.dense_model = None
        self.dense_collection = None
        self.reranker = None

    def _bm25(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.corpus)), key=lambda i: (-float(scores[i]), self.corpus[i]['chunk_id']))
        return [self.corpus[i]['chunk_id'] for i in order[:k]]

    def _load_dense(self) -> None:
        if self.dense_model is None:
            import chromadb
            from sentence_transformers import SentenceTransformer
            path = Path('data/processed/fatf_chroma')
            model_file = path / 'model_name.txt'
            if not model_file.exists():
                raise FileNotFoundError('Missing dense index. Run python -m src.rag build-dense')
            name = model_file.read_text(encoding='utf-8').strip()
            self.dense_model = SentenceTransformer(name)
            self.dense_collection = chromadb.PersistentClient(path=str(path)).get_collection('fatf_guidance')
            if self.dense_collection.count() != len(self.corpus):
                raise RuntimeError('Dense index size differs from corpus. Run python -m src.rag build-dense')

    def _dense(self, query: str, k: int) -> list[str]:
        self._load_dense()
        vec = self.dense_model.encode([query], normalize_embeddings=True).tolist()
        ids = self.dense_collection.query(query_embeddings=vec, n_results=k)['ids'][0]
        for cid in ids:
            if cid not in self.by_id:
                raise RuntimeError(f'Dense index has stale/unknown chunk {cid}. Rebuild it.')
        return ids

    @staticmethod
    def rrf(lexical_ids: list[str], dense_ids: list[str], k: int, rrf_k: int = 60) -> list[str]:
        fused: dict[str, float] = defaultdict(float)
        for results in (lexical_ids, dense_ids):
            for rank, chunk_id in enumerate(dict.fromkeys(results), 1):
                fused[chunk_id] += 1 / (rrf_k + rank)
        return sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]

    def _load_reranker(self) -> None:
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

    def search(self, question: str, method: str, k: int = 5, candidate_k: int = 20) -> list[str]:
        if method not in METHODS:
            raise ValueError(f'Unknown method {method}')
        candidate_k = min(candidate_k, len(self.corpus))
        if method == 'bm25':
            return self._bm25(question, k)
        if method == 'dense':
            return self._dense(question, k)
        fused = self.rrf(self._bm25(question, candidate_k), self._dense(question, candidate_k), candidate_k)
        if method == 'hybrid':
            return fused[:k]
        self._load_reranker()
        scores = self.reranker.predict(
            [(question, self.by_id[cid]['text']) for cid in fused],
            batch_size=8, show_progress_bar=False,
        )
        score_by_id = {cid: float(score) for cid, score in zip(fused, scores)}
        return sorted(fused, key=lambda cid: (-score_by_id[cid], cid))[:k]

    def warmup(self, question: str, methods: tuple[str, ...]) -> None:
        for method in methods:
            self.search(question, method)


def run(annotations: Path = GOLD, output: Path = RUNS,
        split: str = 'dev', methods: tuple[str, ...] = METHODS) -> None:
    questions = [q for q in read_jsonl(annotations) if split == 'all' or q['split'] == split]
    if not questions:
        raise ValueError(f'No questions in split {split}')
    corpus = load_corpus()
    bench = RetrievalBench(corpus)
    # First search may download a model. Warm it before benchmarking.
    print('Loading and warming retrieval models (excluded from measured latency)...', flush=True)
    bench.warmup(questions[0]['question'], methods)
    records = []
    for i, q in enumerate(questions, 1):
        for method in methods:
            start = time.perf_counter()
            ranked = bench.search(q['question'], method)
            elapsed = 1000 * (time.perf_counter() - start)
            records.append({
                'question_id': q['id'], 'split': q['split'], 'doc_id': q['doc_id'],
                'method': method, 'ranked_chunk_ids': ranked,
                'latency_ms_warm': round(elapsed, 2),
                'local_inference_api_cost_usd': None,
            })
        print(f'Completed {i}/{len(questions)} questions', flush=True)
    output = split_output(output, split)
    write_jsonl(output, records)
    print(f'Saved {len(records)} predictions to {output}; no relevance metrics claimed until annotation.')


def score(annotations: Path = GOLD, predictions: Path = RUNS,
          output: Path = SUMMARY, split: str = 'dev') -> dict:
    corpus_ids = {c['chunk_id'] for c in load_corpus()}
    qs = {q['id']: q for q in read_jsonl(annotations)
          if split == 'all' or q['split'] == split}
    rows = [r for r in read_jsonl(split_output(predictions, split)) if r['question_id'] in qs]
    grouped: dict[str, list[dict]] = defaultdict(list)
    missing = set()
    for q in qs.values():
        if not q['gold_chunk_ids'] or q.get('annotation_status') != 'verified':
            missing.add(q['id'])
        elif not set(q['gold_chunk_ids']).issubset(corpus_ids):
            raise ValueError(f"Unknown gold chunk ID(s) in {q['id']}")
        elif any(not cid.startswith(q['doc_id'] + '-') for cid in q['gold_chunk_ids']):
            raise ValueError(f"Gold chunk from unexpected document in {q['id']}")
    for r in rows:
        method = r['method']
        if method not in METHODS or len(r['ranked_chunk_ids']) != len(set(r['ranked_chunk_ids'])):
            raise ValueError(f'Invalid retrieved IDs for {method}')
        if not set(r['ranked_chunk_ids']).issubset(corpus_ids):
            raise ValueError(f'Unknown prediction chunk ID in {r["question_id"]}')
        item = {**r, 'scored': r['question_id'] not in missing}
        if item['scored']:
            item.update(retrieval_scores(qs[r['question_id']]['gold_chunk_ids'], r['ranked_chunk_ids']))
        grouped[method].append(item)
    metrics = {}
    for method in METHODS:
        samples = grouped[method]
        labelled = [r for r in samples if r['scored']]
        metrics[method] = {
            'predictions': len(samples), 'verified_questions_scored': len(labelled),
            'recall_at_5': round(statistics.mean(r['recall_at_5'] for r in labelled), 4) if labelled else None,
            'mrr_at_5': round(statistics.mean(r['mrr_at_5'] for r in labelled), 4) if labelled else None,
            'hit_at_5': round(statistics.mean(r['hit_at_5'] for r in labelled), 4) if labelled else None,
            'mean_latency_ms_warm': round(statistics.mean(r['latency_ms_warm'] for r in samples), 2) if samples else None,
            'median_latency_ms_warm': round(statistics.median(r['latency_ms_warm'] for r in samples), 2) if samples else None,
        }
    summary = {
        'split': split, 'total_questions': len(qs), 'unverified_question_ids': sorted(missing),
        'complete_annotation': not missing, 'metrics': metrics,
        'interpretation_note': 'Recall@5=mean fraction of all verified gold chunks retrieved; '
                               'MRR@5=reciprocal rank of first relevant chunk. '
                               'Latency excludes initial model loading and downloads. '
                               'No generated answer/citation correctness is measured in this retrieval-only experiment.',
    }
    output = split_output(output, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def preview(annotations: Path = GOLD, output: Path = PREVIEW) -> None:
    """Candidate discovery for annotation, NOT automatic gold labels."""
    corpus = load_corpus()
    bench = RetrievalBench(corpus)
    rows = []
    for q in read_jsonl(annotations):
        ids = list(dict.fromkeys(bench._bm25(q['question'], 8)))
        rows.append({
            'id': q['id'], 'question': q['question'], 'target_doc_id': q['doc_id'],
            'candidate_chunks': [{
                'chunk_id': cid, 'title': bench.by_id[cid]['title'],
                'pdf_page': bench.by_id[cid]['page_number'],
                'snippet': bench.by_id[cid]['text'][:500],
            } for cid in ids],
            'important': 'Candidates are NOT gold. Inspect the source PDF and verify all answer-bearing chunks.',
        })
    write_jsonl(output, rows)
    print(f'Wrote BM25 candidate passages to {output}. Verify against the PDFs before annotating gold.')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('preview')
    r = sub.add_parser('run'); r.add_argument('--split', choices=('dev', 'test', 'all'), default='dev')
    s = sub.add_parser('score'); s.add_argument('--split', choices=('dev', 'test', 'all'), default='dev')
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'preview': preview()
    elif args.command == 'run': run(split=args.split)
    elif args.command == 'score': score(split=args.split)

if __name__ == '__main__':
    main()

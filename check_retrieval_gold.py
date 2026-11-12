"""Check FinCrime Investigator's retrieval annotations against the local indexed corpus.
Run from the fincrime-investigator project root:
    python check_retrieval_gold.py
"""
import json
from pathlib import Path
from src.rag import load_corpus

path = Path('data/evaluation/retrieval_gold.jsonl')
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
corpus = {r['chunk_id']: r for r in load_corpus()}
assert len(rows) == 12, f'Expected 12 questions, found {len(rows)}'
assert sum(row['split'] == 'dev' for row in rows) == 9
assert sum(row['split'] == 'test' for row in rows) == 3
issues = []
for q in rows:
    ids = q['gold_chunk_ids']
    if q['split'] == 'test':
        assert ids == [] and q['gold_answer'] is None, f"Holdout leaked: {q['id']}"
        continue
    if not ids or q.get('annotation_status') != 'verified':
        issues.append(f"Unverified development question: {q['id']}")
    for cid in ids:
        chunk = corpus.get(cid)
        if chunk is None:
            issues.append(f"MISSING: {q['id']} -> {cid}")
        elif chunk['doc_id'] != q['doc_id']:
            issues.append(f"WRONG DOCUMENT: {q['id']} -> {cid}")
    print(q['id'], ':', len(ids), 'gold chunks')

# Key page boundaries that are not fully visible in the candidate preview.
for cid in [
    'fatf_banking_rba_2014-p020-c001-58978c51',
    'fatf_banking_rba_2014-p021-c002-58978c51',
    'fatf_financial_investigations_2012-p008-c004-cb1475fd',
    'fatf_financial_investigations_2012-p020-c001-cb1475fd',
]:
    print('\n'+'='*65+'\n'+cid)
    if cid in corpus:
        print(corpus[cid]['text'][:1300])
    else:
        print('MISSING')

if issues:
    raise SystemExit('\n'.join(issues))
print('\nSCHEMA AND ID CHECKS PASSED. Read the four printed chunks for substantive confirmation.')

"""Day-1 FATF ingestion and BM25/dense retrieval. Hybrid and reranking: Day 2.

Official PDF URLs are retained per chunk. Document text is a *reference*,
not evidence that any given account committed a crime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

DOCUMENTS = [
    {
        'doc_id': 'fatf_banking_rba_2014',
        'title': 'Guidance for a Risk-Based Approach: The Banking Sector',
        'publication_date': '2014-10',
        'url': 'https://www.fatf-gafi.org/content/dam/fatf-gafi/guidance/Risk-Based-Approach-Banking-Sector.pdf.coredownload.pdf',
        'landing_page': 'https://www.fatf-gafi.org/en/publications/fatfrecommendations/documents/risk-based-approach-banking-sector.html',
        'filename': 'banking_rba_2014.pdf',
        'caveat': 'Does not incorporate revisions to FATF Recommendation 1 made in 2025.',
    },
    {
        'doc_id': 'fatf_tbml_indicators_2021',
        'title': 'Trade-Based Money Laundering: Risk Indicators',
        'publication_date': '2021-03-11',
        'url': 'https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Trade-Based-Money-Laundering-Risk-Indicators.pdf.coredownload.inline.pdf',
        'landing_page': 'https://www.fatf-gafi.org/en/publications/Methodsandtrends/Trade-based-money-laundering-indicators.html',
        'filename': 'tbml_indicators_2021.pdf',
        'caveat': 'Indicators prompt examination and are not themselves proof of illegal conduct.',
    },
    {
        'doc_id': 'fatf_financial_investigations_2012',
        'title': 'Operational Issues — Financial Investigations Guidance',
        'publication_date': '2012',
        'url': 'https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Operational%20Issues_Financial%20investigations%20Guidance.pdf.coredownload.pdf',
        'landing_page': 'https://www.fatf-gafi.org/en/publications/Methodsandtrends/Operationalissues-financialinvestigationsguidance.html',
        'filename': 'financial_investigations_2012.pdf',
        'caveat': 'Operational guidance rather than contemporary legal advice.',
    },
]


def download_pdfs(output_dir: Path = Path('data/raw/fatf')) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for doc in DOCUMENTS:
        path = output_dir / doc['filename']
        if not path.exists():
            try:
                req = Request(doc['url'], headers={'User-Agent': 'Mozilla/5.0 (research project; academic)'})
                with urlopen(req, timeout=60) as response:
                    contents = response.read()
                if not contents.startswith(b'%PDF'):
                    raise ValueError('Response is not a PDF')
                path.write_bytes(contents)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {doc['title']}. Visit {doc['landing_page']} "
                    f"and save the PDF to {path}. Original error: {exc}"
                ) from exc
        downloaded.append(path)
    return downloaded


def infer_section_hint(page_text: str) -> str | None:
    # Conservative extraction; section tags are hints, never fabricated canonical headings.
    for line in page_text.splitlines():
        line = line.strip()
        if re.match(r'^(?:SECTION\s+[IVXLC]+\b|CHAPTER\s+\d+\b|\d+\.\d*\s+[A-Z])', line, re.I):
            return line[:150]
    return None


def build_corpus(raw_dir: Path = Path('data/raw/fatf'),
                 out_path: Path = Path('data/processed/fatf_chunks.jsonl'),
                 chunk_chars: int = 1000, overlap: int = 150) -> list[dict]:
    if not (200 <= chunk_chars <= 3000 and 0 <= overlap < chunk_chars):
        raise ValueError('Invalid chunk/overlap settings')
    import pymupdf as fitz  # pymupdf, imported lazily
    chunks = []
    for doc in DOCUMENTS:
        file = raw_dir / doc['filename']
        if not file.exists():
            raise FileNotFoundError(f'Missing {file}: run python -m src.rag download')
        data = file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with fitz.open(stream=data, filetype='pdf') as pdf:
            for page_number, page in enumerate(pdf, start=1):
                extracted = page.get_text('text', sort=True)
                text = re.sub(r'[\t ]+', ' ', extracted).strip()
                if not text:
                    continue
                section = infer_section_hint(text)
                step = chunk_chars - overlap
                for index, start in enumerate(range(0, len(text), step), start=1):
                    body = text[start:start+chunk_chars]
                    if not body.strip():
                        continue
                    chunks.append({
                        'chunk_id': f"{doc['doc_id']}-p{page_number:03d}-c{index:03d}-{digest[:8]}",
                        'doc_id': doc['doc_id'], 'title': doc['title'],
                        'publication_date': doc['publication_date'],
                        'source_url': doc['url'], 'landing_page': doc['landing_page'],
                        'source_sha256': digest,
                        'source_caveat': doc['caveat'],
                        'page_number': page_number,
                        'section_hint': section,
                        'text': body,
                    })
                    if start + chunk_chars >= len(text):
                        break
    if not chunks:
        raise RuntimeError('No extractable text; check PDF files and extraction methods.')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in chunks), encoding='utf-8')
    return chunks


def load_corpus(path: Path = Path('data/processed/fatf_chunks.jsonl')) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}: run python -m src.rag ingest')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.casefold())


def bm25_search(question: str, corpus: list[dict], k: int = 5) -> list[dict]:
    from rank_bm25 import BM25Okapi
    index = BM25Okapi([tokenize(c['text']) for c in corpus])
    scores = index.get_scores(tokenize(question))
    order = sorted(range(len(corpus)), key=lambda i: (-float(scores[i]), corpus[i]['chunk_id']))
    return [{**corpus[i], 'retrieval_score': float(scores[i]), 'method': 'bm25'} for i in order[:k]]


def build_dense(corpus: list[dict],
                index_dir: Path = Path('data/processed/fatf_chroma'),
                model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer
    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(index_dir))
    try:
        client.delete_collection('fatf_guidance')
    except Exception:  # Chroma versions vary in missing-collection errors
        pass
    col = client.create_collection('fatf_guidance', metadata={'hnsw:space': 'cosine'})
    for start in range(0, len(corpus), 64):
        batch = corpus[start:start+64]
        vectors = model.encode([r['text'] for r in batch], normalize_embeddings=True).tolist()
        col.add(
            ids=[r['chunk_id'] for r in batch], embeddings=vectors,
            documents=[r['text'] for r in batch],
            metadatas=[{
                'title': r['title'], 'doc_id': r['doc_id'],
                'publication_date': r['publication_date'],
                'page_number': r['page_number'], 'source_url': r['source_url'],
                'landing_page': r['landing_page'],
                'section_hint': r['section_hint'] or 'unresolved',
                'source_sha256': r['source_sha256'],
                'source_caveat': r['source_caveat'],
            } for r in batch],
        )
    (index_dir / 'model_name.txt').write_text(model_name, encoding='utf-8')


def dense_search(question: str, k: int = 5,
                 index_dir: Path = Path('data/processed/fatf_chroma')) -> list[dict]:
    import chromadb
    from sentence_transformers import SentenceTransformer
    model_file = index_dir / 'model_name.txt'
    if not model_file.exists():
        raise FileNotFoundError('Build dense index using python -m src.rag build-dense')
    model = SentenceTransformer(model_file.read_text(encoding='utf-8').strip())
    client = chromadb.PersistentClient(path=str(index_dir))
    col = client.get_collection('fatf_guidance')
    vec = model.encode([question], normalize_embeddings=True).tolist()
    result = col.query(query_embeddings=vec, n_results=k,
                       include=['documents', 'metadatas', 'distances'])
    return [{**meta, 'chunk_id': chunk_id, 'text': text,
             'retrieval_distance': distance, 'method': 'dense'}
            for chunk_id, text, meta, distance in zip(
                result['ids'][0], result['documents'][0],
                result['metadatas'][0], result['distances'][0])]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('download')
    sub.add_parser('ingest')
    sub.add_parser('build-dense')
    search = sub.add_parser('search')
    search.add_argument('question')
    search.add_argument('--method', choices=['bm25','dense'], default='bm25')
    search.add_argument('-k', type=int, default=5)
    args = p.parse_args()
    if args.command == 'download':
        for fp in download_pdfs(): print(fp)
    elif args.command == 'ingest':
        print('Created', len(build_corpus()), 'FATF chunks')
    elif args.command == 'build-dense':
        build_dense(load_corpus()); print('Built Chroma dense index')
    else:
        hits = bm25_search(args.question, load_corpus(), args.k) if args.method == 'bm25' else dense_search(args.question, args.k)
        for hit in hits:
            print(f"{hit['chunk_id']} | {hit['title']} | PDF page {hit['page_number']} | {hit['source_url']}")
            print(hit['text'][:240].replace('\n', ' '), '\n')

if __name__ == '__main__':
    main()

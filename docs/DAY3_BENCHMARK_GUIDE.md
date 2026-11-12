# FinCrime Investigator — Day 3: reproducible benchmarks

This upgrade is **additive**: copy `src/*.py` and `tests/test_benchmark_day3.py` into your existing project. Do **not** overwrite `src/agent.py`, `src/rag.py`, your local dataset, any manually edited evaluation questions, or your working Streamlit app.

## Start here (PowerShell, from project root)

```powershell
cd 'C:\Users\jjwei\final project Ironhack\fincrime-investigator'
python -m pytest tests/test_benchmark_day3.py -q
python -m pytest tests/test_day1_pipeline.py tests/test_agent_day2.py -q
```

The first tests are offline and should pass without OpenAI calls. The real-data experiments below run on *your machine*. No numeric results are included in this package.

## Experiment A — four retrieval methods

```powershell
python -m src.benchmark_retrieval prepare
python -m src.benchmark_retrieval preview
```

1. In VS Code open `data/evaluation/retrieval_gold.jsonl` (12 questions; 9 dev, 3 held-out test). **Do not rerun `prepare`** once you start annotating.
2. The `data/evaluation/retrieval_candidates.jsonl` file lists **BM25 suggestions**, not gold evidence. Open the source FATF PDF, locate the actual answer-bearing passage, and verify it independently. Be sure to capture all relevant passage chunks, even when BM25 omitted them.
3. Change the `gold_chunk_ids` list for each question to the stable chunk IDs that genuinely answer it. Fill `gold_answer` (brief, verified), `annotation_note` (FATF title, PDF page and rationale), and change `annotation_status` to `"verified"` **only after checking the passage**. Do not label a document merely because its title matches the question.
4. Initially annotate your **9 dev questions**, keeping the 3 test questions for the final evaluation. Do not tune your retrieval configuration on test scores.

Example of the *shape* of a human-reviewed annotation (**not a claimed correct answer**):

```json
{"id":"RET-06","gold_chunk_ids":["REPLACE_WITH_VERIFIED_CHUNK_ID"],"gold_answer":"Summarize the verified FATF passage.","annotation_status":"verified","annotation_note":"Verified against official FATF PDF, page X."}
```

**Edit existing JSONL lines**, preserving other fields (id, question, split and doc_id). Avoid pasting the abbreviated illustrative object above as a replacement line.

After annotation, run:

```powershell
python -m src.benchmark_retrieval run --split dev
python -m src.benchmark_retrieval score --split dev
```

Outputs:

- `data/evaluation/benchmark_retrieval_predictions_dev.jsonl` — all four top-5 rankings and warmed query latency, per question.
- `data/evaluation/benchmark_retrieval_metrics_dev.json` — Recall@5 (fraction of all reference chunks found), Hit@5, MRR@5, mean/median latency and annotation coverage.
- `complete_annotation=false` means **partial**, not a finished benchmark. No quality metrics are fabricated for unverified questions.
- `usd_cost` is not supplied for local compute: API cost is not the full hardware/energy cost.

Only after fixing the methodology and model settings, perform the holdout test:

```powershell
python -m src.benchmark_retrieval run --split test
python -m src.benchmark_retrieval score --split test
```

The test result uses separate filenames, preserving the dev experiment. Once test results are known, treat them as final rather than further tuning on them.

**What Experiment A does not yet measure:** generated answer correctness or citation faithfulness; ranking metrics do not imply an accurate answer. These require human grading of generated responses. Results from 12 questions should be presented as a small, exploratory benchmark with uncertainty, not a production-grade ranking claim.

## Reference answers for the original 24 investigation questions

```powershell
python -m src.benchmark_references
python -m src.benchmark_split_check
```

- `data/evaluation/reference_drafts.jsonl`: draft answers based on independently written SQL for 8 account and 6 network tasks, a correct evidence-scope refusal for the four unsupported tasks, and 6 knowledge questions intentionally left without fabricated FATF answers.
- **These are drafts, not approved gold.** Review raw transactions / FATF PDFs, verify any pattern threshold and edge cases, and transfer validated evidence into the original `data/evaluation/questions.jsonl` without regenerating or changing its fixed split.
- `data/evaluation/split_overlap_audit.json` warns if dev and test account roots share neighbors. Do not claim strict case independence where networks overlap.

**Caution:** rapid movement questions require particular care: close-in-time, same-currency transfers do not establish that the same funds were transferred. Reference candidates may differ from the production heuristic's first-pair and result-budget rules.

## Experiment B — investigation architectures

The suite compares **four arms on appropriate tasks**, not an unfair four-way contest on all questions:

| Arm | Access | Suitable tests |
|---|---|---|
| `baseline` | Deterministic DuckDB/rules | Account statistics, rule findings, transaction evidence |
| `tools_only` | LangGraph + financial tools (no FATF RAG) | Financial investigation commentary and tool behavior |
| `agentic_rag` | Existing LangGraph agent + financial tools + RAG | Same financial audit plus guidance citation coverage |
| `rag_only` | FATF documents + LLM, no transaction DB | General FATF knowledge and unsupported-account abstention |

Start a **small pilot** (a maximum of one account, three document questions), to avoid unnecessary API usage:

```powershell
python -m src.benchmark_investigation run-reports --split dev --max-accounts 1
python -m src.benchmark_investigation run-rag-only --split dev --max-questions 3
python -m src.benchmark_investigation audit --split dev
```

Then open `data/evaluation/benchmark_investigation/report_audit_summary_dev.json`, `report_audit_dev.jsonl`, `manual_review.csv`, and the cached report JSON files beneath `baseline/`, `tools_only/`, and `agentic_rag/`.

**Model costs:** these are real API calls; check your billing and model configuration. The benchmark stores token usage where reported; `usd_cost=null` means uncalculated, not free. Use a consistent model for both agent arms (e.g. the exact `OPENAI_MODEL` in your `.env`). Avoid rerunning lots of questions until the pilot has passed.

For a wider experiment, run:

```powershell
python -m src.benchmark_investigation run-reports --split dev --max-accounts 0
python -m src.benchmark_investigation run-rag-only --split dev --max-questions 0
python -m src.benchmark_investigation audit --split dev
```

Reports are **cached** by arm/account so running a second time does not generate another bill. If you change model, prompt, retrieval method or source data, archive/remove previous cached outputs **and document the changed configuration** before repeating the experiment. Do not mix configurations in one reported benchmark.

### Evaluation distinctions that matter

- Report number/citation fields of both agents come from the *same deterministic baseline or executed retrieval outputs*. Comparing these numbers measures consistency, not independent LLM predictive accuracy. No improvement claims from a tautological tie.
- `audit` compares numeric counts/times to independent SQL and validates transaction IDs and FATF chunk IDs **syntactically**. It does not judge whether the FATF text supports a particular sentence.
- `manual_review.csv` has intentionally empty human correctness, evidence faithfulness, citation support and refusal columns. Fill them after reviewing full answers and original sources. It does not score account Q&A, because the existing investigation agent produces account **reports**, not direct answers to each of the 24 questions.
- To satisfy the final 24-question deliverable, run a separate question-level agent test or add a question-taking agent mode and score per-question correctness against approved gold. The current report-level audit is a useful *partial* Experiment B, not a substitute for this final benchmark.
- Avoid reporting precision, recall or F1 for transaction laundering detection on the enriched subset; use separate representative held-out labelled units if you have time. A correct transaction statistic is not a correctly detected offence.
- At least one informative high-activity case and one unsupported/low-signal case belong in final demo evidence. Your account `0117999::80A67B530` is an already checked low-signal example.

## Suggested evidence for your seven-minute presentation

1. Dataset + label separation and honest limitations.
2. Experiment A: four methods, identical 12 questions, same top-5 cutoff, measured Recall@5/MRR@5/latency and annotation coverage.
3. Experiment B: deterministic baseline versus tool agent versus agentic RAG, with explicit caveat that numerical report fields are copied from deterministic tools; present human citation/error findings and runtime.
4. Show a correct *abstention* and a higher-activity case with transaction IDs.
5. Explain failure categories, leakage limitations, and why hybrid+rerank was or was not worth the extra latency.

**Do not commit** raw IBM CSVs, Parquet labels, API keys or local evaluation files. Commit the benchmark source, tests, this guide and a small aggregate results summary after validation.

## Experiment B extension — full 24-question Q&A benchmark

The report audit above measures structural reliability; it is **not** a per-question answer evaluation. The additional `src/benchmark_qa.py` makes that second comparison possible using all 24 frozen question specifications. It reuses the financial tool schemas and LangGraph orchestration rather than adding a separate agent framework.

Start with a **small, lower-cost smoke test**:

```powershell
python -m src.benchmark_qa run --split dev --limit 2 --arms baseline,rag_only
python -m src.benchmark_qa sheet --split dev
python -m src.benchmark_qa score --split dev
```

The `--limit` is the number of selected questions; the pilot tries to include an unsupported question so you can check abstention. It saves a JSON prediction for each applicable arm/question under `data/evaluation/benchmark_qa/<arm>/<question_id>.json`. Missing/N/A arm-task combinations are **excluded**, never scored as failures.

Then pilot both question-answering agents on a small set:

```powershell
python -m src.benchmark_qa run --split dev --limit 3 --arms tools_only,agentic_rag --max-calls 4
python -m src.benchmark_qa sheet --split dev
python -m src.benchmark_qa score --split dev
```

After confirming behavior and fixing errors, run all **18 development questions**:

```powershell
python -m src.benchmark_qa run --split dev --limit 0
python -m src.benchmark_qa sheet --split dev
python -m src.benchmark_qa score --split dev
```

Open `data/evaluation/benchmark_qa/human_scores_dev.csv`. Enter **0 or 1** only for applicable, human-reviewed fields: `factual_correct_0_1`, `evidence_faithful_0_1`, `citation_support_0_1`, `appropriate_abstention_0_1`; leave irrelevant rubric fields blank. Check against the independently verified reference drafts, transaction IDs and FATF PDFs. Enter `failure_type` and `review_notes` for failures. Re-running `sheet` appends new predictions while preserving existing human scores. Re-running `run` reuses cached predictions to avoid repeated charges.

After freezing settings and annotations, use **all 6 test questions** once:

```powershell
python -m src.benchmark_qa run --split test --limit 0
python -m src.benchmark_qa sheet --split test
python -m src.benchmark_qa score --split test
```

The output `human_summary_<split>.json` reports only metrics with human labels and gives their denominators. Different arm/task coverage must be disclosed. For this small synthetic case study, report number of evaluated questions, time, errors, citations, factual correctness and abstention — **not** calibrated probabilities of crime.

Important: question-level benchmarking creates additional API requests; it is optional to run every expensive arm for every eligible question if cost/time becomes prohibitive. In that case, disclose the reduced sample instead of reporting it as a 24-question four-arm result.

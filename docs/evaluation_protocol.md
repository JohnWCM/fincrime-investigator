# Initial evaluation protocol

* 24 question specifications are generated before model completion: 6 FATF guidance, 8 account, 6 network, 4 unsupported.
* 18 development / 6 fixed holdout test questions. Account-specific dev and test prompts draw from disjoint deterministic account-ID pools; gold financial answers must be independently verified. Do not tune thresholds, retrieval, or prompts against the holdout results.
* The generator does **not** invent reference answers. Each `gold_answer` begins null; annotate after checking the underlying Parquet query or a specific FATF PDF page.
* Account facts need checked source transaction IDs. Guidance facts need verified document title, page, version, date and chunk ID.
* Separate document-only task evaluation from transaction-analysis tasks inaccessible to RAG-only.
* Retrieval: Recall@5, MRR@5 against document IDs/chunks on a separate retrieval question set; citation accuracy, answer correctness, latency and cost.
* Investigation: correct facts, observed patterns, evidence faithfulness, correct abstention, tool success, latency and cost.
* If evaluating labelled transaction detection, keep unit of analysis consistent (transaction vs account), use a representative test population, and report precision, recall, F1 and class prevalence. The intentionally enriched demo subset is *not* a representative test set.
* Never infer that positive simulated laundering labels validate explanatory narratives.

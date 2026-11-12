# Day 2 — Bounded LangGraph Investigation Agent

Copy `src/agent.py` and `tests/test_agent_day2.py` to the matching folders in your existing project. Copy `requirements-agent.txt` to the project root. **Do not replace** existing data files, `src/rag.py`, `src/functions.py`, or your Streamlit app.

From project root in VS Code PowerShell:

```powershell
python -m pip install -r requirements-agent.txt
python -m pytest tests/test_agent_day2.py -q
```

In a **local** `.env` file, add:

```dotenv
OPENAI_API_KEY=YOUR_REAL_API_KEY
OPENAI_MODEL=gpt-4.1-mini
```

Keep `.env` out of Git. Your API account must have independent API billing enabled; a ChatGPT subscription does not provide API credits.

Read a real selected account ID from your `data/processed/subset_manifest.json` and run:

```powershell
python -m src.agent --account "PASTE_ACTUAL_ACCOUNT_ID" --method hybrid --max-calls 5 --output data/processed/agent_report.json
```

Then open `data/processed/agent_report.json`. Check `status`, `statistics`, `pattern_findings`, `supporting_transaction_ids`, `network_findings`, `guidance_citations`, `tool_trace`, `evidence_validation`, and `execution`. Missing or incomplete evidence must be acknowledged; LLM commentary is explicitly unverified.

If the model or API returns an error, copy the full traceback without the API key. Do not commit `.env`, generated reports, labels, parquet, Chroma, or CSV data.

Limitations: this starter has not been tested against a live API or the full IBM dataset in the packaged environment. It does not yet implement an evaluation harness, citation/answer correctness scoring, USD costs, or Streamlit integration. These are the next tasks.

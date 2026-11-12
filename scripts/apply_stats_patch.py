"""Patch the user's existing FinCrime Investigator modules in place.

Preserves local modifications (e.g. MessagesState import and content-block extraction),
creates backups in ignored data/evaluation/stats_patch_backup, and refuses partial or
unexpected file layouts rather than risking a broken project.

Run from repository root: python scripts/apply_stats_patch.py
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path.cwd()
TARGETS = [
    Path("src/agent.py"),
    Path("src/tools_only_agent.py"),
    Path("src/benchmark_qa.py"),
]
BACKUPS = Path("data/evaluation/stats_patch_backup")
MARK = "# FINCRIME_STATS_PATCH_V2"


def once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {n}. No files were changed. ")
    return src.replace(old, new, 1)


def patch_agent(src: str) -> str:
    new_tool = '''    @tool("get_account_statistics", args_schema=AccountInput)
    def get_account_statistics(account_id: str) -> str:
        """Return complete DuckDB aggregates (counts, currency totals, earliest/latest timestamps).

        Unlike get_account_transactions, this is never limited to the first 25 records.
        The source is the unlabelled investigation subset; it is not a risk probability.
        """
        try:
            require_selected(account_id)
            result = fin.account_statistics(account_id)
            return json.dumps({"kind": "statistics", "data": result}, default=str)
        except Exception as exc:
            return json.dumps({"kind": "error", "tool": "statistics", "detail": str(exc)})

'''
    src = once(src, '    @tool("detect_transaction_patterns", args_schema=AccountInput)\n',
               new_tool + '    @tool("detect_transaction_patterns", args_schema=AccountInput)\n',
               'agent: insert SQL statistics tool')
    src = once(src,
               '    return [get_account_transactions, detect_transaction_patterns,\n            explore_transaction_network, search_aml_guidance]',
               '    return [get_account_transactions, detect_transaction_patterns,\n            explore_transaction_network, search_aml_guidance, get_account_statistics]',
               'agent: expose SQL tool')
    src = once(src, '            "Do not claim crimes occurred. Never make up transaction IDs, amounts, "\n',
               '            "Use get_account_statistics for exact counts, currency totals, and earliest/latest timestamps. "\n'
               '            "Do not infer complete totals from limited transaction listings. "\n'
               '            "In statistics, start means earliest and end means latest. "\n'
               '            "Do not claim crimes occurred. Never make up transaction IDs, amounts, "\n',
               'agent: instruct reliable aggregate use')
    return src


def patch_tools_only(src: str) -> str:
    src = once(src,
        "        # The original build_tools returns [transactions, patterns, network, guidance].\n"
        "        # Only the first three are registered; the LLM cannot invoke RAG in this arm.\n"
        "        tools = build_tools(fin, account_id, 'hybrid')[:3]",
        "        # Select by name so adding SQL tools never accidentally exposes RAG.\n"
        "        tools = [t for t in build_tools(fin, account_id, 'hybrid')\n"
        "                 if t.name != 'search_aml_guidance']",
        'tools_only: select tools by name')
    src = once(src,
        "            'Use the provided financial-analysis tools only. NO external knowledge base, '",
        "            'Use get_account_statistics for exact counts, currency amounts, earliest and latest times. '\n"
        "            'Do not infer exact totals from truncated transaction listings. '\n"
        "            'Use the provided financial-analysis tools only. NO external knowledge base, '",
        'tools_only: SQL prompt')
    return src


def patch_benchmark(src: str) -> str:
    src = once(src,
        "ROOT = Path('data/evaluation/benchmark_qa')",
        "ROOT = Path(os.environ.get('FINCRIME_BENCHMARK_ROOT', 'data/evaluation/benchmark_qa'))",
        'benchmark: versioned outputs')
    src = once(src,
        "        if arm == 'tools_only':\n"
        "            tools = tools[:3]\n"
        "        elif q['category'] == 'knowledge':\n"
        "            # A FATF knowledge question should not expose arbitrary account transactions.\n"
        "            tools = tools[3:]\n"
        "        bound = ChatOpenAI(model=model, timeout=60, max_retries=2).bind_tools(tools, parallel_tool_calls=False)",
        "        if arm == 'tools_only':\n"
        "            # No knowledge retrieval tool in this ablation. Select by name, not position.\n"
        "            tools = [t for t in tools if t.name != 'search_aml_guidance']\n"
        "        elif q['category'] == 'knowledge':\n"
        "            # Knowledge questions must not expose account-level transaction data.\n"
        "            tools = [t for t in tools if t.name == 'search_aml_guidance']\n"
        "        llm = ChatOpenAI(model=model, timeout=60, max_retries=2)\n"
        "        bound = llm.bind_tools(tools, parallel_tool_calls=False)\n"
        "        # For account facts, require the real aggregate tool before any other tools.\n"
        "        stats_bound = None\n"
        "        if q['category'] == 'account':\n"
        "            stats_tool = next(t for t in tools if t.name == 'get_account_statistics')\n"
        "            stats_bound = llm.bind_tools([stats_tool], tool_choice='get_account_statistics',\n"
        "                                         parallel_tool_calls=False)",
        'benchmark: tool selection and forced SQL call')
    src = once(src,
        "            if n >= max_calls:\n"
        "                return {'messages': [AIMessage(content='Tool budget reached. Insufficient evidence for additional claims.')]}\n"
        "            return {'messages': [bound.invoke(state['messages'])]}",
        "            if n >= max_calls:\n"
        "                # Never discard existing evidence just because the tool budget is exhausted.\n"
        "                finish = HumanMessage(content=('Tool budget reached. Answer using only the '\n"
        "                                               'evidence already returned; if insufficient, say so.'))\n"
        "                return {'messages': [llm.invoke(state['messages'] + [finish])]}\n"
        "            if stats_bound is not None and not any(\n"
        "                isinstance(m, ToolMessage) and getattr(m, 'name', None) == 'get_account_statistics'\n"
        "                for m in state['messages']\n"
        "            ):\n"
        "                return {'messages': [stats_bound.invoke(state['messages'])]}\n"
        "            return {'messages': [bound.invoke(state['messages'])]}",
        'benchmark: required statistics and graceful final answer')
    src = once(src,
        "            'Cite actual transaction IDs or FATF [chunk_ids] when supporting factual claims. '",
        "            'Use get_account_statistics for exact account totals, counts, currency sums, '\n"
        "            'and earliest/latest timestamps; do NOT count a limited transaction list. '\n"
        "            'Its start field is the earliest timestamp and end is the latest timestamp. '\n"
        "            'The statistics tool may count self-transfers in both incoming and outgoing. '\n"
        "            'Use those exact values; answer directly rather than calling unnecessary tools. '\n"
        "            'Cite actual transaction IDs or FATF [chunk_ids] when supporting factual claims. '",
        'benchmark: SQL response instructions')
    src = once(src,
        "def run(split: str, arms: tuple[str, ...], limit: int, model: str | None,\n"
        "        method: str, max_calls: int) -> None:\n"
        "    qs = [q for q in read_jsonl(QUESTIONS) if q['split'] == split]\n"
        "    if limit:\n",
        "def run(split: str, arms: tuple[str, ...], limit: int, model: str | None,\n"
        "        method: str, max_calls: int, question_ids: tuple[str, ...] = ()) -> None:\n"
        "    qs = [q for q in read_jsonl(QUESTIONS) if q['split'] == split]\n"
        "    if question_ids:\n"
        "        available = {q['id'] for q in qs}\n"
        "        unknown = set(question_ids) - available\n"
        "        if unknown:\n"
        "            raise ValueError(f'Unknown question IDs or wrong split: {sorted(unknown)}')\n"
        "        qs = [q for q in qs if q['id'] in question_ids]\n"
        "    elif limit:\n",
        'benchmark: selected questions')
    src = once(src,
        "    a.add_argument('--max-calls', type=int, default=4)\n",
        "    a.add_argument('--max-calls', type=int, default=4)\n"
        "    a.add_argument('--question-ids', default='',\n"
        "                   help='Comma-separated IDs; overrides --limit')\n",
        'benchmark: selected question CLI')
    src = once(src,
        "        run(args.split, arms, args.limit, args.model, args.method, args.max_calls)",
        "        ids = tuple(x.strip() for x in args.question_ids.split(',') if x.strip())\n"
        "        run(args.split, arms, args.limit, args.model, args.method, args.max_calls, ids)",
        'benchmark: selected CLI wire-up')
    return src


def main() -> None:
    original = {}
    for relative in TARGETS:
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run from the project root.")
        original[relative] = path.read_text(encoding="utf-8-sig")
    if any(MARK in content for content in original.values()):
        raise SystemExit("Patch already applied (or partially applied). No files changed.")
    new = {
        TARGETS[0]: patch_agent(original[TARGETS[0]]),
        TARGETS[1]: patch_tools_only(original[TARGETS[1]]),
        TARGETS[2]: patch_benchmark(original[TARGETS[2]]),
    }
    for path, value in new.items():
        updated = MARK + "\n" + value
        ast.parse(updated, filename=str(path))
        new[path] = updated
    (ROOT / BACKUPS).mkdir(parents=True, exist_ok=True)
    for relative in TARGETS:
        backup = ROOT / BACKUPS / (relative.name + ".before_stats_patch")
        if backup.exists():
            raise SystemExit(f"Refusing to overwrite existing backup {backup}.")
    for relative in TARGETS:
        backup = ROOT / BACKUPS / (relative.name + ".before_stats_patch")
        backup.write_text(original[relative], encoding="utf-8")
    for relative in TARGETS:
        (ROOT / relative).write_text(new[relative], encoding="utf-8")
        print(f"PATCHED {relative}")
    print(f"Backups are in {BACKUPS} (ignored by .gitignore).")
    print("Next: python -m pytest tests/test_stats_tool_patch.py -q")


if __name__ == "__main__":
    main()

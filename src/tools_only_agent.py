# FINCRIME_STATS_PATCH_V2
"""Controlled Day-3 ablation: same LangGraph pattern, financial tools only.

Unlike src.agent, this variant has no guidance search tool and no RAG fallback.
It retains deterministic accounting and report validation from src.agent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from src.agent import assemble_report, build_tools, extract_evidence, token_usage
from src.baseline import investigate
from src.functions import FinancialTools
from src.graph import summarize_network


def extract_final_text(messages) -> str:
    """Accept either plain-string LLM content or text parts; no invented fallback prose."""
    from langchain_core.messages import AIMessage
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return ''
    if isinstance(last.content, str):
        return last.content.strip()
    if isinstance(last.content, list):
        return '\n'.join(
            part.get('text', '') for part in last.content
            if isinstance(part, dict) and part.get('type') in ('text', 'output_text')
            and isinstance(part.get('text'), str)
        ).strip()
    return ''


def run_tools_only(account_id: str, *, model: str | None = None,
                   max_calls: int = 5,
                   subset_path: Path = Path('data/processed/investigation_subset.parquet')) -> dict:
    if not 1 <= max_calls <= 8:
        raise ValueError('max_calls must be 1..8')
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langgraph.graph import START, END, StateGraph, MessagesState
    from langgraph.prebuilt import ToolNode

    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY in local .env')
    model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
    started = time.perf_counter()
    with FinancialTools(subset_path) as fin:
        baseline = investigate(account_id, fin)
        if baseline.get('status') != 'completed':
            return _report(baseline, {'guidance': [], 'network': None, 'tool_trace': [], 'tool_errors': []},
                           '', model, time.perf_counter() - started, {'input_tokens': 0, 'output_tokens': 0}, max_calls)
        # Select by name so adding SQL tools never accidentally exposes RAG.
        tools = [t for t in build_tools(fin, account_id, 'hybrid')
                 if t.name != 'search_aml_guidance']
        bound = ChatOpenAI(model=model, max_retries=2, timeout=60).bind_tools(tools, parallel_tool_calls=False)

        def call_model(state: dict) -> dict:
            used = sum(isinstance(msg, ToolMessage) for msg in state['messages'])
            if used >= max_calls:
                return {'messages': [AIMessage(content='Tool budget reached; see deterministic evidence-bound report.')]}
            return {'messages': [bound.invoke(state['messages'])]}

        def route(state: dict) -> str:
            return 'tools' if getattr(state['messages'][-1], 'tool_calls', None) else 'finish'

        graph = StateGraph(MessagesState)
        graph.add_node('agent', call_model)
        graph.add_node('tools', ToolNode(tools, handle_tool_errors=True))
        graph.add_edge(START, 'agent')
        graph.add_conditional_edges('agent', route, {'tools': 'tools', 'finish': END})
        graph.add_edge('tools', 'agent')
        program = graph.compile()
        system = SystemMessage(content=(
            'You are a cautious AML analyst investigating synthetic data. '
            'Use get_account_statistics for exact counts, currency amounts, earliest and latest times. '
            'Do not infer exact totals from truncated transaction listings. '
            'Use the provided financial-analysis tools only. NO external knowledge base, '
            'FATF retrieval or document citations are available. '
            'Never invent transaction IDs, numeric facts, ownership or accusations. '
            'Use the selected account only and distinguish missing evidence. '
            'Finish with a short qualitative investigation summary. '
            'Financial numbers in the final report are populated deterministically.'
        ))
        user = HumanMessage(content=(
            f'Investigate synthetic account {account_id}. Inspect transactions, observable patterns '
            'and available network evidence, acknowledging missing contextual information.'
        ))
        result = program.invoke({'messages': [system, user]}, config={'recursion_limit': 2 * max_calls + 5})
        evidence = extract_evidence(result['messages'])
        evidence['guidance'] = []
        if evidence['network'] is None:
            try:
                network = fin.explore_transaction_network(account_id, hops=1, max_edges=3000)
                evidence['network'] = summarize_network(network)
                evidence['tool_trace'].append({'tool': 'explore_transaction_network',
                                               'result_kind': 'network', 'origin': 'fallback'})
            except Exception as exc:
                evidence['tool_errors'].append(f'Network fallback: {exc}')
        return _report(baseline, evidence, extract_final_text(result['messages']), model,
                       time.perf_counter() - started, token_usage(result['messages']), max_calls)


def _report(baseline: dict, evidence: dict, text: str, model: str,
            elapsed: float, tokens: dict, max_calls: int) -> dict:
    report = assemble_report(baseline, evidence, text, model, 'hybrid', elapsed, tokens, max_calls)
    # assemble_report requires a retrieval-method-shaped parameter; update the
    # actual arm metadata explicitly, without implying a guidance retrieval.
    report['execution']['retrieval_method'] = 'disabled'
    report['guidance_citations'] = []
    report['limitations'].append('This agent arm has no AML guidance retrieval tool.')
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--account', required=True)
    p.add_argument('--max-calls', type=int, default=5)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    response = run_tools_only(args.account, max_calls=args.max_calls)
    content = json.dumps(response, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
        print(f'Saved report to {args.output}')
    else:
        print(content)

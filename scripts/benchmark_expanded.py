"""Independent expanded-account QA benchmark for FinCrime Investigator.

Run from repository root (PowerShell):
  python scripts/benchmark_expanded.py prepare
  python scripts/benchmark_expanded.py run --split dev --limit 2 --arms tools_only,agentic_rag
  python scripts/benchmark_expanded.py sheet --split dev
  python scripts/benchmark_expanded.py score --split dev

Does not import training labels or modify the original benchmark outputs.
Requires existing src.agent.build_tools with get_account_statistics (stats patch).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import statistics
import time

SUBSET = Path("data/processed/investigation_subset_expanded.parquet")
SOURCE = Path("data/processed/transactions.parquet")
MANIFEST = Path("data/processed/subset_manifest_expanded.json")
ROOT = Path("data/evaluation/benchmark_expanded")
QUESTIONS = ROOT / "questions.jsonl"
REFERENCES = ROOT / "references.jsonl"
ARMS = ("tools_only", "agentic_rag")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGETS = (
    "001::800058B80",
    "014::80006FCE0",
    "012::8000F7C60",
    "015::803037BE0",
    "01292::8004203B0",
    "010::800058920",
    "0117::8000C5D00",
    "023::800167090",
)
QUESTION_SPECS = (
    ("total", "How many unique transactions involve account {a} in the complete analysis period?"),
    ("incoming", "How many incoming transactions does account {a} have, including self-transfers?"),
    ("outgoing", "How many outgoing transactions does account {a} have, including self-transfers?"),
    ("latest", "What is the latest transaction timestamp for account {a}? Give the exact date and time."),
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 'prepare' first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def quoted_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def coverage_by_account(con, path: Path, accounts: tuple[str, ...]) -> dict[str, int]:
    """Count each involving transaction once, including account self-transfers."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW case_tx AS SELECT tx_id, from_account_id, to_account_id FROM read_parquet({quoted_path(path)})")
    # Each row belongs to a source account and a destination account; a self-transfer
    # must count only once. The IN filter avoids joins on every raw row.
    rows = con.execute("""
        WITH matches AS (
            SELECT from_account_id AS account_id, tx_id FROM case_tx
             WHERE from_account_id IN (SELECT unnest(?))
            UNION ALL
            SELECT to_account_id AS account_id, tx_id FROM case_tx
             WHERE to_account_id IN (SELECT unnest(?))
               AND from_account_id <> to_account_id
        )
        SELECT account_id, count(*) AS n FROM matches GROUP BY account_id
    """, [list(accounts), list(accounts)]).fetchall()
    return {a: int(n) for a, n in rows}


def prepare() -> None:
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("Install project requirements: python -m pip install -r requirements.txt") from exc
    for p in (SUBSET, SOURCE, MANIFEST):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Build the expanded dataset before benchmarking.")
    if QUESTIONS.exists() or REFERENCES.exists():
        raise SystemExit(f"Refusing to overwrite existing benchmark data in {ROOT}. Back it up first.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    added = set(manifest.get("added_cases", []))
    missing = set(TARGETS) - added
    if missing:
        raise SystemExit(f"The expanded manifest does not identify these 8 target accounts as added: {sorted(missing)}")
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 4")
    # Independent coverage check: one scan of the source and one of the expanded subset.
    counts_full = coverage_by_account(con, SOURCE, TARGETS)
    counts_subset = coverage_by_account(con, SUBSET, TARGETS)
    failed = [(a, counts_full.get(a, 0), counts_subset.get(a, 0)) for a in TARGETS
              if counts_full.get(a, 0) != counts_subset.get(a, 0) or not counts_full.get(a, 0)]
    if failed:
        con.close()
        raise SystemExit("INCOMPLETE account histories (account, full, expanded): " + repr(failed))
    print("PASS: all 8 account histories match the original unlabeled transactions.parquet")
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet({quoted_path(SUBSET)})")
    questions, refs = [], []
    for i, account in enumerate(TARGETS):
        row = con.execute("""
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE to_account_id=?) AS incoming,
                   count(*) FILTER (WHERE from_account_id=?) AS outgoing,
                   max(timestamp) AS latest
              FROM tx WHERE from_account_id=? OR to_account_id=?
        """, [account] * 4).fetchone()
        values = dict(zip(("total", "incoming", "outgoing", "latest"), row))
        # Strict account-disjoint split. The final 2 accounts are held-out.
        split = "dev" if i < 6 else "test"
        for j, (kind, template) in enumerate(QUESTION_SPECS):
            qid = f"EXP-{i*4 + j + 1:03d}"
            q = {"id": qid, "split": split, "account_id": account,
                 "category": "account", "kind": kind,
                 "question": template.format(a=account)}
            ref = {"id": qid, "split": split, "account_id": account, "kind": kind,
                   "expected": str(values[kind]),
                   "reference_method": "independent parameterized DuckDB SQL",
                   "reference_status": "computed_from_complete_unlabeled_dataset"}
            questions.append(q)
            refs.append(ref)
        print(f"{account}: full={counts_full[account]} subset={counts_subset[account]} split={split}")
    con.close()
    write_jsonl(QUESTIONS, questions)
    write_jsonl(REFERENCES, refs)
    (ROOT / "metadata.json").write_text(json.dumps({
        "dataset": str(SUBSET), "source": str(SOURCE),
        "dataset_size_bytes": SUBSET.stat().st_size,
        "dataset_mtime_ns": SUBSET.stat().st_mtime_ns,
        "n_questions": len(questions), "dev": 24, "test": 8,
        "split_rule": "accounts 1-6 dev, accounts 7-8 held out",
        "no_laundering_labels_used": True,
        "warning": "Account-selected subset; not population AML precision/recall. References never sent to LLM."
    }, indent=2), encoding="utf-8")
    print(f"SAVED {QUESTIONS} and {REFERENCES}; 24 dev, 8 held-out")


def assert_same_dataset() -> None:
    metadata = ROOT / "metadata.json"
    if not metadata.exists():
        raise SystemExit("Missing expanded benchmark metadata. Run prepare first.")
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    if (not SUBSET.exists() or SUBSET.stat().st_size != saved["dataset_size_bytes"]
            or SUBSET.stat().st_mtime_ns != saved["dataset_mtime_ns"]):
        raise SystemExit("Expanded dataset changed since reference generation. Freeze data or regenerate a NEW benchmark.")


def run_agent(question: dict, arm: str, model: str | None, max_calls: int) -> dict:
    """Scoped LangGraph evaluation; force SQL stats for an exact account question."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langgraph.graph import START, END, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    from src.functions import FinancialTools
    from src.agent import build_tools, extract_evidence, token_usage
    from src.tools_only_agent import extract_final_text

    load_dotenv()
    llm = ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                     timeout=60, max_retries=2)
    account = question["account_id"]
    with FinancialTools(SUBSET) as fin:
        tools = build_tools(fin, account, "hybrid")
        if arm == "tools_only":
            tools = [t for t in tools if t.name != "search_aml_guidance"]
        stats = next((t for t in tools if t.name == "get_account_statistics"), None)
        if stats is None:
            raise RuntimeError("Missing get_account_statistics. Apply the LangGraph statistics patch first.")
        first = llm.bind_tools([stats], tool_choice="get_account_statistics", parallel_tool_calls=False)
        full = llm.bind_tools(tools, parallel_tool_calls=False)

        def call_model(state: dict):
            msgs = state["messages"]
            tool_count = sum(isinstance(m, ToolMessage) for m in msgs)
            if tool_count == 0:
                reply = first.invoke(msgs)
            elif tool_count >= max_calls:
                reply = llm.invoke(msgs + [HumanMessage(content=(
                    "Tool budget exhausted. Provide the answer from existing evidence; "
                    "if missing, state that you do not have enough evidence."))])
            else:
                reply = full.invoke(msgs)
            return {"messages": [reply]}

        def route(state: dict):
            return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "done"

        graph = StateGraph(MessagesState)
        graph.add_node("model", call_model)
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", route, {"tools": "tools", "done": END})
        graph.add_edge("tools", "model")
        messages = [
            SystemMessage(content=(
                "You are an evaluator for synthetic AML account data. Answer only the user's "
                "question, briefly, using the complete DuckDB account statistics tool. "
                "Do not calculate totals from truncated transaction listings. "
                "start = earliest timestamp; end = latest timestamp. "
                "Self-transfers are counted in both incoming and outgoing but only once "
                "in total unique transactions. Do not invent source IDs or claims of crime. "
                "If no evidence is available, say so.")),
            HumanMessage(content=question["question"]),
        ]
        out = graph.compile().invoke({"messages": messages},
                                     config={"recursion_limit": 2 * max_calls + 7})
        trace = extract_evidence(out["messages"])
        tokens = token_usage(out["messages"])
        return {"answer": extract_final_text(out["messages"]),
                "tool_trace": trace.get("tool_trace", []),
                "tool_errors": trace.get("tool_errors", []),
                "used_statistics": any(t.get("tool") == "get_account_statistics"
                                       for t in trace.get("tool_trace", [])),
                "input_tokens": tokens.get("input_tokens"),
                "output_tokens": tokens.get("output_tokens")}


def run(split: str, arms: list[str], limit: int, model: str | None, max_calls: int) -> None:
    assert_same_dataset()
    qs = [q for q in read_jsonl(QUESTIONS) if q["split"] == split]
    if limit:
        qs = qs[:limit]
    for q in qs:
        for arm in arms:
            path = ROOT / "predictions" / arm / f"{q['id']}.json"
            if path.exists():
                print(f"SKIP {arm} {q['id']}: cached")
                continue
            print(f"RUN {arm} {q['id']}", flush=True)
            start = time.perf_counter()
            try:
                output = run_agent(q, arm, model, max_calls)
                status = "completed"
            except Exception as exc:
                output = {"answer": "", "tool_trace": [], "tool_errors": [f"{type(exc).__name__}: {exc}"],
                          "used_statistics": False, "input_tokens": None, "output_tokens": None}
                status = "error"
                print(f"ERROR {arm} {q['id']}: {type(exc).__name__}: {exc}")
            data = {**q, "arm": arm, "status": status,
                    "latency_seconds": round(time.perf_counter() - start, 3), **output}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("Predictions saved. Human review is necessary; successful execution does not imply factual correctness.")


def sheet(split: str) -> None:
    qs = [q for q in read_jsonl(QUESTIONS) if q["split"] == split]
    refs = {r["id"]: r for r in read_jsonl(REFERENCES)}
    out = ROOT / f"human_scores_{split}.csv"
    existing = {}
    if out.exists():
        with out.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                existing[(r["question_id"], r["arm"])] = r
    rows = []
    for q in qs:
        for arm in ARMS:
            p = ROOT / "predictions" / arm / f"{q['id']}.json"
            if not p.exists():
                continue
            result = json.loads(p.read_text(encoding="utf-8"))
            row = {
                "question_id": q["id"], "split": split,
                "account_id": q["account_id"], "kind": q["kind"],
                "arm": arm, "question": q["question"],
                "expected": refs[q["id"]]["expected"],
                "answer": result.get("answer", ""),
                "statistics_tool_used": str(result.get("used_statistics", False)),
                "tool_errors": "; ".join(result.get("tool_errors", [])),
                "factual_correct_0_1": "", "evidence_faithful_0_1": "", "notes": "",
            }
            prev = existing.get((q["id"], arm))
            if prev:
                for field in ("factual_correct_0_1", "evidence_faithful_0_1", "notes"):
                    row[field] = prev.get(field, "")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No {split} predictions. Run the agents first.")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"SAVED {out} ({len(rows)} rows; prior grades preserved)")


def score(split: str) -> None:
    path = ROOT / f"human_scores_{split}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run sheet first.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    arms = {}
    for arm in ARMS:
        items = [r for r in rows if r["arm"] == arm]
        result = {"answers_in_sheet": len(items)}
        for metric in ("factual_correct_0_1", "evidence_faithful_0_1"):
            graded = []
            for r in items:
                value = str(r.get(metric, "")).strip()
                if value not in ("", "0", "1"):
                    raise SystemExit(f"Invalid {metric}={value!r} on {r['question_id']}; use 0, 1 or blank")
                if value:
                    graded.append(int(value))
            result[metric] = {"n": len(graded), "mean": (round(statistics.mean(graded), 4) if graded else None)}
        latencies = []
        tool_used = 0
        for r in items:
            p = ROOT / "predictions" / arm / f"{r['question_id']}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d.get("latency_seconds"), (int, float)):
                    latencies.append(d["latency_seconds"])
                tool_used += bool(d.get("used_statistics"))
        result["mean_latency_seconds"] = round(statistics.mean(latencies), 3) if latencies else None
        result["statistics_tool_used"] = f"{tool_used}/{len(items)}"
        arms[arm] = result
    summary = {"split": split, "total_scoring_rows": len(rows), "per_arm": arms,
               "note": "Manually graded account-disjoint benchmark; not ML/TF classification precision or recall."}
    dest = ROOT / f"human_summary_{split}.json"
    dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("prepare", help="Verify account coverage, save 24 dev and 8 held-out references")
    runner = subs.add_parser("run", help="Run LangGraph agents, defaults to a two-question pilot")
    runner.add_argument("--split", choices=("dev", "test"), default="dev")
    runner.add_argument("--limit", type=int, default=2, help="0 = entire split; default 2-question pilot")
    runner.add_argument("--arms", default="tools_only,agentic_rag")
    runner.add_argument("--model", default=None)
    runner.add_argument("--max-calls", type=int, default=4)
    for cmd in ("sheet", "score"):
        p = subs.add_parser(cmd)
        p.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        if args.limit < 0 or args.max_calls < 1:
            raise SystemExit("--limit must be >=0 and --max-calls >=1")
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        if not arms or any(a not in ARMS for a in arms):
            raise SystemExit(f"--arms accepts only: {', '.join(ARMS)}")
        run(args.split, arms, args.limit, args.model, args.max_calls)
    elif args.command == "sheet":
        sheet(args.split)
    else:
        score(args.split)


if __name__ == "__main__":
    main()

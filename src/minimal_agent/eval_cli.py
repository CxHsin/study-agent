import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from minimal_agent.deepseek import DeepSeekAdapter
from minimal_agent.eval import EvalLimits, EvalRunner, load_cases
from minimal_agent.provider_client import ProviderClient
from minimal_agent.workspace_tools import WorkspaceTools


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded Agent Eval cases.")
    parser.add_argument("cases", type=Path, help="JSON Eval Case file")
    parser.add_argument("--artifact", type=Path, required=True, help="Redacted JSONL output")
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--token-budget", type=int, default=100_000)
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        parser.error("DEEPSEEK_API_KEY is required for real-provider evaluation.")
    cases = load_cases(args.cases)

    def model_factory(case):
        return ProviderClient(DeepSeekAdapter(api_key=api_key))

    def tools_factory(case):
        return WorkspaceTools(Path("workspace")).registry()

    report = EvalRunner(
        model_factory,
        tools_factory,
        EvalLimits(args.max_calls, args.timeout, args.token_budget),
    ).run(cases)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    report.to_jsonl(args.artifact)
    print(f"Eval {report.status}: {len(report.cases)} case(s) -> {args.artifact}")
    return 0 if report.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

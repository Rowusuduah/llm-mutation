"""
CLI for llm-mutation.

Commands:
  mutate run         Run mutation test on a prompt + eval suite
  mutate report      Generate human-readable report from saved JSON
  mutate ci          CI gate: exit 1 if mutation score < min-score
  mutate calibrate   Test known-severity mutations against your eval suite

Usage examples:
  mutate run --prompt prompts/cs.txt --eval evals/test_cs.py --output report.json
  mutate report --input report.json --format markdown
  mutate ci --input report.json --min-score 0.80
  mutate calibrate --prompt prompts/cs.txt --eval evals/test_cs.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_eval_fn(eval_path: str):
    """
    Load the eval function from a Python file.

    The file must define:
      - eval_fn(prompt: str, test_cases: list) -> float
      - TEST_CASES: list   (optional — used if --cases not provided)
    """
    p = Path(eval_path)
    if not p.exists():
        print(f"Error: eval file not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_llm_mutation_eval", p)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"Error loading eval file: {exc}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(module, "eval_fn"):
        print(
            f"Error: eval file must define eval_fn(prompt, test_cases) -> float",
            file=sys.stderr,
        )
        sys.exit(1)
    return module


def _load_prompt(prompt_path: str) -> str:
    p = Path(prompt_path)
    if not p.exists():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def cmd_run(args) -> None:
    from ._engine import MutationEngine
    from ._runner import MutantRunner
    from ._models import MutationReport

    prompt_text = _load_prompt(args.prompt)
    eval_module = _load_eval_fn(args.eval)
    test_cases = getattr(eval_module, "TEST_CASES", [])

    operators = [op.strip() for op in args.operators.split(",")] if args.operators else None

    engine = MutationEngine(
        operators=operators,
        max_mutations=args.max_mutations,
    )
    mutations = engine.generate(prompt_text)

    if not mutations:
        print("No mutations generated. The prompt may not contain recognizable clauses.")
        print("Check that your prompt has 'Never', 'Always', 'You must', scope phrases, etc.")
        sys.exit(0)

    print(f"Generated {len(mutations)} mutations. Running eval suite...")

    runner = MutantRunner(
        eval_fn=eval_module.eval_fn,
        test_cases=test_cases,
        delta_threshold=args.delta,
        runs_per_mutant=args.runs,
        parallel=not args.no_parallel,
    )

    # Score original first (runner does this internally but we want to show it)
    results = runner.run(mutations)
    original_score = results[0].original_score if results else 0.0
    report = MutationReport.from_results(results, prompt_text, original_score)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(output_path)

    print(report.summary())
    print(f"\nReport saved to: {output_path}")


def cmd_report(args) -> None:
    from ._models import MutationReport

    report = MutationReport.load_json(args.input)
    print(report.summary(format=args.format))


def cmd_ci(args) -> None:
    from ._models import MutationReport

    report = MutationReport.load_json(args.input)
    pct = int(report.mutation_score * 100)
    min_pct = int(args.min_score * 100)

    if report.mutation_score >= args.min_score:
        print(
            f"CI GATE: PASSED — Mutation score {pct}% >= {min_pct}% threshold "
            f"({report.score_verdict})"
        )
        sys.exit(0)
    else:
        print(
            f"CI GATE: FAILED — Mutation score {pct}% < {min_pct}% threshold "
            f"({report.score_verdict})"
        )
        print(f"\nSurviving mutations that need test coverage:")
        for r in report.results:
            if r.verdict == "SURVIVED":
                print(f"  - {r.mutation.operator}: {r.mutation.description}")
                if r.mutation.recommendation:
                    print(f"    ADD: {r.mutation.recommendation}")
        sys.exit(1)


def cmd_calibrate(args) -> None:
    from ._calibrate import run_calibration

    prompt_text = _load_prompt(args.prompt)
    eval_module = _load_eval_fn(args.eval)
    test_cases = getattr(eval_module, "TEST_CASES", [])

    report = run_calibration(
        eval_fn=eval_module.eval_fn,
        test_cases=test_cases,
        prompt=prompt_text,
        delta_threshold=args.delta,
    )
    print(report.summary())

    if report.calibration_score < 0.80:
        print("\nWARNING: Calibration score < 80%. Your eval suite may not reliably detect mutations.")
        sys.exit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mutate",
        description="llm-mutation — Mutation testing for LLM prompts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # mutate run
    p_run = sub.add_parser("run", help="Run mutation test")
    p_run.add_argument("--prompt", required=True, help="Path to prompt file")
    p_run.add_argument("--eval", required=True, help="Path to eval Python file (must define eval_fn)")
    p_run.add_argument("--output", default="mutation_report.json", help="Output report path")
    p_run.add_argument("--operators", default=None, help="Comma-separated operator names (default: all)")
    p_run.add_argument("--max-mutations", type=int, default=20)
    p_run.add_argument("--delta", type=float, default=0.15, help="Score delta threshold (default: 0.15)")
    p_run.add_argument("--runs", type=int, default=3, help="Runs per mutant for median averaging")
    p_run.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")

    # mutate report
    p_report = sub.add_parser("report", help="Generate report from saved JSON")
    p_report.add_argument("--input", required=True, help="Report JSON file")
    p_report.add_argument("--format", default="text", choices=["text", "json", "markdown"])

    # mutate ci
    p_ci = sub.add_parser("ci", help="CI gate — exit 1 if score below threshold")
    p_ci.add_argument("--input", required=True, help="Report JSON file")
    p_ci.add_argument("--min-score", type=float, default=0.80, help="Minimum mutation score (default: 0.80)")

    # mutate calibrate
    p_cal = sub.add_parser("calibrate", help="Test known-severity mutations against your eval suite")
    p_cal.add_argument("--prompt", required=True, help="Path to prompt file")
    p_cal.add_argument("--eval", required=True, help="Path to eval Python file")
    p_cal.add_argument("--delta", type=float, default=0.15)

    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "report": cmd_report,
        "ci": cmd_ci,
        "calibrate": cmd_calibrate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

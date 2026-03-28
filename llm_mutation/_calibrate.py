"""
Calibration module — Berean Null Test (PAT-046).

Before trusting your mutation score, verify that your eval suite can
actually detect known-severity mutations. If your eval suite can't catch
a HIGH-severity mutation (complete system prompt removal), the mutation
score is meaningless.

The `mutate calibrate` command runs 5 known-severity mutations against
the user's eval suite and reports a calibration score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from ._models import Mutation


CalibrationSeverity = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class CalibrationCase:
    severity: CalibrationSeverity
    description: str
    build_mutant: Callable[[str], str]   # (original_text) -> mutated_text
    recommendation: str


def _remove_prohibitions(text: str) -> str:
    import re
    result = re.sub(
        r"(?m)^[ \t]*(?:[Nn]ever|[Dd]o not|[Dd]on't|[Aa]void)\s+.+[.!]?\s*$",
        "",
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", result).strip() or text


def _remove_requirements(text: str) -> str:
    import re
    result = re.sub(
        r"(?m)^[ \t]*(?:[Aa]lways|[Yy]ou must|[Yy]ou should|[Mm]ake sure|[Ee]nsure)\s+.+[.!]?\s*$",
        "",
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", result).strip() or text


def _remove_first_line(text: str) -> str:
    lines = text.strip().split("\n")
    if len(lines) <= 1:
        return text
    return "\n".join(lines[1:]).strip()


def _remove_last_instruction(text: str) -> str:
    import re
    lines = text.strip().split("\n")
    # Find last non-empty line
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            new_lines = lines[:i] + lines[i + 1:]
            result = "\n".join(new_lines).strip()
            return result if result else text
    return text


# Five canonical calibration mutations (known severity) — defined after helper functions
_CALIBRATION_CASES: list[CalibrationCase] = [
    CalibrationCase(
        severity="HIGH",
        description="Complete system prompt removal (all instructions stripped)",
        build_mutant=lambda text: "You are a helpful assistant.",
        recommendation=(
            "Your eval suite cannot detect complete prompt removal. "
            "Add a test case that would fail on a blank/generic assistant."
        ),
    ),
    CalibrationCase(
        severity="HIGH",
        description="All prohibitive clauses removed (all 'Never' / 'Do not' lines)",
        build_mutant=_remove_prohibitions,
        recommendation=(
            "Your eval suite misses prohibition removal. "
            "Add test cases that verify prohibited behaviors are actually refused."
        ),
    ),
    CalibrationCase(
        severity="MEDIUM",
        description="All requirement clauses removed (all 'Always' / 'You must' lines)",
        build_mutant=_remove_requirements,
        recommendation=(
            "Your eval suite misses requirement removal. "
            "Add test cases verifying each required behavior appears in responses."
        ),
    ),
    CalibrationCase(
        severity="MEDIUM",
        description="Role/persona instruction removed (first line of prompt)",
        build_mutant=_remove_first_line,
        recommendation=(
            "Your eval suite misses persona loss. "
            "Add a test case verifying the LLM adopts the specified role."
        ),
    ),
    CalibrationCase(
        severity="LOW",
        description="Single instruction clause dropped (last instruction line)",
        build_mutant=_remove_last_instruction,
        recommendation=(
            "Your eval suite misses single-clause removal. "
            "Each important instruction should have at least one dedicated test case."
        ),
    ),
]


@dataclass
class CalibrationResult:
    case: CalibrationCase
    original_score: float
    mutant_score: float
    delta: float
    caught: bool   # True if delta >= delta_threshold


@dataclass
class CalibrationReport:
    calibration_score: float     # fraction of calibration cases caught
    total_cases: int
    caught: int
    missed: int
    results: list[CalibrationResult]
    warnings: list[str]

    def summary(self) -> str:
        pct = int(self.calibration_score * 100)
        lines = [
            f"\nCALIBRATION RESULTS:",
            f"Tested {self.total_cases} known-severity mutations against your eval suite.",
        ]
        for r in self.results:
            icon = "\u2713" if r.caught else "\u2717"
            caught_str = "Caught" if r.caught else "Missed"
            lines.append(
                f"  {icon} Severity: {r.case.severity} \u2014 {r.case.description}"
                f" \u2014 {caught_str} (score: {r.original_score:.2f} \u2192 {r.mutant_score:.2f})"
            )
        lines.append("")
        lines.append(f"Calibration score: {pct}% ({self.caught}/{self.total_cases} known-severity mutations caught)")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def run_calibration(
    eval_fn: Callable[[str, list], float],
    test_cases: list,
    prompt: str,
    delta_threshold: float = 0.15,
    runs_per_case: int = 1,
) -> CalibrationReport:
    """Run calibration cases and return CalibrationReport."""
    import statistics

    def _score(text: str) -> float:
        scores = [float(eval_fn(text, test_cases)) for _ in range(runs_per_case)]
        return statistics.median(scores)

    original_score = _score(prompt)
    results: list[CalibrationResult] = []
    warnings: list[str] = []

    for case in _CALIBRATION_CASES:
        try:
            mutant_text = case.build_mutant(prompt)
            if mutant_text == prompt:
                # Skip if mutation didn't change anything (prompt may not have matching clauses)
                continue
            mutant_score = _score(mutant_text)
            delta = original_score - mutant_score
            caught = delta >= delta_threshold
            results.append(
                CalibrationResult(
                    case=case,
                    original_score=original_score,
                    mutant_score=mutant_score,
                    delta=delta,
                    caught=caught,
                )
            )
            if not caught and case.severity == "HIGH":
                warnings.append(case.recommendation)
        except Exception as exc:
            warnings.append(f"Calibration case '{case.description}' failed: {exc}")

    caught_count = sum(1 for r in results if r.caught)
    total = len(results)
    score = caught_count / total if total > 0 else 0.0

    return CalibrationReport(
        calibration_score=score,
        total_cases=total,
        caught=caught_count,
        missed=total - caught_count,
        results=results,
        warnings=warnings,
    )

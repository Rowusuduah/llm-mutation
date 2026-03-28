"""
llm-mutation data models.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


MutationOperator = Literal[
    "NegateConstraint",
    "DropClause",
    "ScopeExpand",
    "ScopeNarrow",
    "ConditionInvert",
    "PhraseSwap",
]

MutantVerdict = Literal["KILLED", "SURVIVED", "SKIPPED"]

MutationScoreVerdict = Literal["STRONG", "ADEQUATE", "MARGINAL", "WEAK", "DANGEROUS"]


@dataclass
class Mutation:
    """A single semantic mutation of the original prompt."""

    operator: MutationOperator
    original_text: str       # full original prompt
    mutated_text: str        # full mutated prompt
    description: str         # human-readable description of what changed
    clause_removed: str      # the specific clause/phrase that was mutated
    recommendation: str      # test case to add if this mutation survives

    @property
    def mutation_id(self) -> str:
        payload = f"{self.operator}:{self.clause_removed}"
        return hashlib.sha1(payload.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "mutation_id": self.mutation_id,
            "operator": self.operator,
            "description": self.description,
            "clause_removed": self.clause_removed,
            "recommendation": self.recommendation,
        }


@dataclass
class MutantResult:
    """Result of running the eval suite against one mutant."""

    mutation: Mutation
    original_score: float
    mutant_score: float
    delta: float             # original_score - mutant_score
    verdict: MutantVerdict
    runs: list[float] = field(default_factory=list)   # individual run scores
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "mutation_id": self.mutation.mutation_id,
            "operator": self.mutation.operator,
            "description": self.mutation.description,
            "clause_removed": self.mutation.clause_removed,
            "original_score": round(self.original_score, 4),
            "mutant_score": round(self.mutant_score, 4),
            "delta": round(self.delta, 4),
            "verdict": self.verdict,
            "runs": [round(r, 4) for r in self.runs],
            "recommendation": self.mutation.recommendation,
            "error": self.error,
        }


@dataclass
class MutationReport:
    """Complete mutation test report."""

    prompt_hash: str
    prompt_preview: str          # first 120 chars of prompt
    original_score: float
    mutation_score: float        # killed / total (excluding skipped)
    score_verdict: MutationScoreVerdict
    total_mutations: int
    killed: int
    survived: int
    skipped: int
    results: list[MutantResult]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    recommendations: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_results(
        cls,
        results: list[MutantResult],
        prompt: str,
        original_score: float,
    ) -> "MutationReport":
        killed = sum(1 for r in results if r.verdict == "KILLED")
        survived = sum(1 for r in results if r.verdict == "SURVIVED")
        skipped = sum(1 for r in results if r.verdict == "SKIPPED")
        countable = killed + survived
        mutation_score = killed / countable if countable > 0 else 0.0

        recs = [
            r.mutation.recommendation
            for r in results
            if r.verdict == "SURVIVED" and r.mutation.recommendation
        ]

        return cls(
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            prompt_preview=prompt[:120].replace("\n", " "),
            original_score=original_score,
            mutation_score=mutation_score,
            score_verdict=_score_to_verdict(mutation_score),
            total_mutations=len(results),
            killed=killed,
            survived=survived,
            skipped=skipped,
            results=results,
            recommendations=recs,
        )

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #

    def summary(self, format: str = "text") -> str:
        if format == "json":
            return self.to_json_str()
        if format == "markdown":
            return self._to_markdown()
        return self._to_text()

    def _to_text(self) -> str:
        pct = int(self.mutation_score * 100)
        lines = [
            f"\nMUTATION SCORE: {pct}% ({self.killed}/{self.killed + self.survived} mutations killed)",
            f"Verdict: {self.score_verdict}",
            f"Original eval score: {self.original_score:.4f}",
            "",
        ]
        killed_results = [r for r in self.results if r.verdict == "KILLED"]
        survived_results = [r for r in self.results if r.verdict == "SURVIVED"]
        skipped_results = [r for r in self.results if r.verdict == "SKIPPED"]

        if killed_results:
            lines.append("KILLED MUTATIONS:")
            for r in killed_results:
                lines.append(
                    f"  \u2713 {r.mutation.operator} \u2014 {r.mutation.description}"
                )
                lines.append(
                    f"    Score: {r.original_score:.2f} \u2192 {r.mutant_score:.2f}"
                    f" (delta: {r.delta:.2f}) \u2713 KILLED"
                )

        if survived_results:
            lines.append("")
            lines.append("SURVIVING MUTATIONS:")
            for r in survived_results:
                lines.append(
                    f"  \u2717 {r.mutation.operator} \u2014 {r.mutation.description}"
                )
                lines.append(
                    f"    Score: {r.original_score:.2f} \u2192 {r.mutant_score:.2f}"
                    f" (delta: {r.delta:.2f}) \u2717 SURVIVED"
                )
                if r.mutation.recommendation:
                    lines.append(f"    \u2192 ADD TEST CASE: {r.mutation.recommendation}")

        if skipped_results:
            lines.append("")
            lines.append(f"SKIPPED: {len(skipped_results)} mutations (errors during eval)")

        if self.score_verdict in ("MARGINAL", "WEAK", "DANGEROUS"):
            threshold_pct = {
                "MARGINAL": 70,
                "WEAK": 60,
                "DANGEROUS": 0,
            }.get(self.score_verdict, 0)
            lines.append("")
            lines.append(
                f"WARNING: Mutation score {pct}% is below recommended threshold of 80%."
            )
            lines.append("Your eval suite may miss production regressions.")

        return "\n".join(lines)

    def _to_markdown(self) -> str:
        pct = int(self.mutation_score * 100)
        lines = [
            f"## Mutation Report",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mutation Score | **{pct}%** ({self.killed}/{self.killed + self.survived} killed) |",
            f"| Verdict | `{self.score_verdict}` |",
            f"| Original Eval Score | {self.original_score:.4f} |",
            f"| Total Mutations | {self.total_mutations} |",
            f"| Killed | {self.killed} |",
            f"| Survived | {self.survived} |",
            f"| Skipped | {self.skipped} |",
            f"",
        ]
        for r in self.results:
            icon = "\u2705" if r.verdict == "KILLED" else "\u274c" if r.verdict == "SURVIVED" else "\u23ed\ufe0f"
            lines.append(
                f"- {icon} `{r.mutation.operator}` \u2014 {r.mutation.description}"
                f" (delta: {r.delta:.2f})"
            )
            if r.verdict == "SURVIVED" and r.mutation.recommendation:
                lines.append(f"  - **Add test:** {r.mutation.recommendation}")
        return "\n".join(lines)

    def to_json_str(self) -> str:
        data = {
            "prompt_hash": self.prompt_hash,
            "prompt_preview": self.prompt_preview,
            "generated_at": self.generated_at.isoformat(),
            "original_score": round(self.original_score, 4),
            "mutation_score": round(self.mutation_score, 4),
            "score_verdict": self.score_verdict,
            "total_mutations": self.total_mutations,
            "killed": self.killed,
            "survived": self.survived,
            "skipped": self.skipped,
            "recommendations": self.recommendations,
            "results": [r.to_dict() for r in self.results],
        }
        return json.dumps(data, indent=2)

    def to_json(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(self.to_json_str(), encoding="utf-8")

    @classmethod
    def load_json(cls, path) -> "MutationReport":
        """Load a previously-saved report (for CI gate and reporting)."""
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return _report_from_dict(data)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _score_to_verdict(score: float) -> MutationScoreVerdict:
    if score >= 0.90:
        return "STRONG"
    if score >= 0.80:
        return "ADEQUATE"
    if score >= 0.70:
        return "MARGINAL"
    if score >= 0.60:
        return "WEAK"
    return "DANGEROUS"


def _report_from_dict(data: dict) -> MutationReport:
    """Reconstruct a MutationReport from a JSON dict (light version for CI gate)."""
    results = []
    for r in data.get("results", []):
        mutation = Mutation(
            operator=r["operator"],
            original_text="",
            mutated_text="",
            description=r["description"],
            clause_removed=r["clause_removed"],
            recommendation=r.get("recommendation", ""),
        )
        results.append(
            MutantResult(
                mutation=mutation,
                original_score=r["original_score"],
                mutant_score=r["mutant_score"],
                delta=r["delta"],
                verdict=r["verdict"],
                runs=r.get("runs", []),
                error=r.get("error"),
            )
        )
    return MutationReport(
        prompt_hash=data["prompt_hash"],
        prompt_preview=data.get("prompt_preview", ""),
        original_score=data["original_score"],
        mutation_score=data["mutation_score"],
        score_verdict=data["score_verdict"],
        total_mutations=data["total_mutations"],
        killed=data["killed"],
        survived=data["survived"],
        skipped=data["skipped"],
        results=results,
        generated_at=datetime.fromisoformat(data["generated_at"]),
        recommendations=data.get("recommendations", []),
    )

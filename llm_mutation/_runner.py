"""
MutantRunner — executes the user's eval suite against each mutant.

The eval_fn signature is:
    eval_fn(prompt: str, test_cases: list) -> float
It must return a score between 0.0 and 1.0 (higher = better).

A mutant is KILLED if:
    original_score - mutant_score >= delta_threshold

A mutant SURVIVES if the score delta is below the threshold
(i.e., your eval suite didn't notice the mutation).
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Callable, Optional

from ._models import Mutation, MutantResult, MutantVerdict


class MutantRunner:
    """
    Runs the eval suite against each mutation and collects verdicts.

    Args:
        eval_fn:          callable (prompt: str, cases: list) -> float
        test_cases:       list of test cases passed to eval_fn
        delta_threshold:  minimum score drop to count as KILLED (default: 0.15)
        runs_per_mutant:  number of runs to median-average (non-determinism mitigation)
        timeout_per_run:  seconds before a single eval run is considered failed
        parallel:         run mutations in parallel
        max_workers:      thread pool size when parallel=True
    """

    def __init__(
        self,
        eval_fn: Callable[[str, list], float],
        test_cases: list,
        delta_threshold: float = 0.15,
        runs_per_mutant: int = 3,
        timeout_per_run: int = 60,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> None:
        if not callable(eval_fn):
            raise TypeError("eval_fn must be callable")
        if not (0.0 < delta_threshold < 1.0):
            raise ValueError("delta_threshold must be between 0.0 and 1.0")
        if runs_per_mutant < 1:
            raise ValueError("runs_per_mutant must be >= 1")

        self.eval_fn = eval_fn
        self.test_cases = test_cases
        self.delta_threshold = delta_threshold
        self.runs_per_mutant = runs_per_mutant
        self.timeout_per_run = timeout_per_run
        self.parallel = parallel
        self.max_workers = max_workers

    def run(self, mutations: list[Mutation]) -> list[MutantResult]:
        """
        Run the eval suite against all mutations.

        First scores the original prompt (median of runs_per_mutant runs),
        then scores each mutant.

        Returns list of MutantResult in same order as input mutations.
        """
        if not mutations:
            return []

        original_prompt = mutations[0].original_text
        original_score = self._score_prompt(original_prompt)

        if self.parallel and len(mutations) > 1:
            return self._run_parallel(mutations, original_score)
        return self._run_sequential(mutations, original_score)

    def _run_sequential(
        self, mutations: list[Mutation], original_score: float
    ) -> list[MutantResult]:
        return [self._evaluate_mutant(m, original_score) for m in mutations]

    def _run_parallel(
        self, mutations: list[Mutation], original_score: float
    ) -> list[MutantResult]:
        results: dict[int, MutantResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._evaluate_mutant, m, original_score): i
                for i, m in enumerate(mutations)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=self.timeout_per_run * self.runs_per_mutant + 10)
                except Exception as exc:
                    results[idx] = MutantResult(
                        mutation=mutations[idx],
                        original_score=original_score,
                        mutant_score=original_score,
                        delta=0.0,
                        verdict="SKIPPED",
                        error=f"Runner exception: {exc}",
                    )
        return [results[i] for i in range(len(mutations))]

    def _evaluate_mutant(
        self, mutation: Mutation, original_score: float
    ) -> MutantResult:
        try:
            mutant_score = self._score_prompt(mutation.mutated_text)
            delta = original_score - mutant_score
            verdict: MutantVerdict = "KILLED" if delta >= self.delta_threshold else "SURVIVED"
            return MutantResult(
                mutation=mutation,
                original_score=original_score,
                mutant_score=mutant_score,
                delta=delta,
                verdict=verdict,
            )
        except Exception as exc:
            return MutantResult(
                mutation=mutation,
                original_score=original_score,
                mutant_score=original_score,
                delta=0.0,
                verdict="SKIPPED",
                error=str(exc),
            )

    def _score_prompt(self, prompt: str) -> float:
        """Score a prompt (original or mutant) using median of multiple runs."""
        scores = []
        for _ in range(self.runs_per_mutant):
            try:
                score = float(self.eval_fn(prompt, self.test_cases))
                if not (0.0 <= score <= 1.0):
                    raise ValueError(
                        f"eval_fn returned {score!r} — must be between 0.0 and 1.0"
                    )
                scores.append(score)
            except ValueError:
                raise
            except Exception as exc:
                raise RuntimeError(f"eval_fn raised an exception: {exc}") from exc
        return statistics.median(scores)

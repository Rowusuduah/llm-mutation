"""
llm-mutation — Mutation testing for LLM prompts.

Find the gaps in your eval suite before production does.

Quickstart:
    pip install llm-mutation
    mutate run --prompt prompts/cs.txt --eval evals/test_cs.py

Pattern: PAT-045 (Judges 6:36-40 — The Gideon Fleece Inversion Pattern)
"""
from __future__ import annotations

from ._models import (
    Mutation,
    MutantResult,
    MutantVerdict,
    MutationReport,
    MutationOperator,
    MutationScoreVerdict,
)
from ._engine import MutationEngine
from ._runner import MutantRunner
from ._store import MutationStore
from ._calibrate import run_calibration, CalibrationReport

__version__ = "0.1.0"
__all__ = [
    # Models
    "Mutation",
    "MutantResult",
    "MutantVerdict",
    "MutationReport",
    "MutationOperator",
    "MutationScoreVerdict",
    # Core
    "MutationEngine",
    "MutantRunner",
    "MutationStore",
    # Calibration
    "run_calibration",
    "CalibrationReport",
]


def _cli_main() -> None:
    from ._cli import main
    main()

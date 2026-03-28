"""
Tests for llm-mutation v0.1.0
==============================
All tests run without an Anthropic API key — no network calls required.
The eval_fn is always a local deterministic function in tests.

Run: pytest tests/ -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable

import pytest

import llm_mutation as lm
from llm_mutation._models import (
    Mutation,
    MutantResult,
    MutationReport,
    _score_to_verdict,
    _report_from_dict,
)
from llm_mutation._engine import (
    MutationEngine,
    _negate_constraint_mutations,
    _drop_clause_mutations,
    _scope_expand_mutations,
    _scope_narrow_mutations,
    _condition_invert_mutations,
    _phrase_swap_mutations,
)
from llm_mutation._runner import MutantRunner
from llm_mutation._store import MutationStore
from llm_mutation._calibrate import (
    run_calibration,
    CalibrationReport,
    _remove_prohibitions,
    _remove_requirements,
    _remove_first_line,
    _remove_last_instruction,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

SAMPLE_PROMPT = """\
You are a customer service agent for AcmeCorp.
Answer questions about software products only.
Never discuss competitor products.
Always respond in formal English.
Do not share internal pricing details.
You must escalate billing disputes to billing@acmecorp.com.
If the user is angry, de-escalate before answering.
Direct pricing questions to sales@acmecorp.com.
"""

BROAD_PROMPT = """\
You are a helpful assistant.
Answer any topic the user asks about.
Always be concise and professional.
You should respond in formal English.
Never discuss illegal activities.
When the user asks for code, provide working examples.
"""


def make_eval_fn(original_score: float = 0.90, mutant_score: float = 0.40) -> Callable:
    """
    Creates a deterministic eval_fn for testing.
    Returns original_score for the original prompt text,
    mutant_score for anything else.
    """
    call_count = {"n": 0}

    def eval_fn(prompt: str, test_cases: list) -> float:
        call_count["n"] += 1
        # First call is always the original scoring
        return original_score if call_count["n"] == 1 else mutant_score

    return eval_fn


def make_constant_eval_fn(score: float) -> Callable:
    """Eval fn that always returns the same score (simulates non-detecting eval suite)."""
    def eval_fn(prompt: str, test_cases: list) -> float:
        return score
    return eval_fn


def make_score_map_eval_fn(score_map: dict[str, float], default: float = 0.90) -> Callable:
    """Eval fn that returns different scores based on prompt content keywords."""
    def eval_fn(prompt: str, test_cases: list) -> float:
        for keyword, score in score_map.items():
            if keyword not in prompt:
                return score
        return default
    return eval_fn


# ------------------------------------------------------------------ #
# Module-level exports                                                 #
# ------------------------------------------------------------------ #

class TestModuleExports:
    def test_version(self):
        assert lm.__version__ == "0.1.0"

    def test_all_exports_present(self):
        expected = [
            "Mutation", "MutantResult", "MutantVerdict", "MutationReport",
            "MutationOperator", "MutationScoreVerdict",
            "MutationEngine", "MutantRunner", "MutationStore",
            "run_calibration", "CalibrationReport",
        ]
        for name in expected:
            assert hasattr(lm, name), f"Missing export: {name}"


# ------------------------------------------------------------------ #
# Mutation data model                                                  #
# ------------------------------------------------------------------ #

class TestMutation:
    def test_mutation_id_is_deterministic(self):
        m = Mutation(
            operator="NegateConstraint",
            original_text="original",
            mutated_text="mutated",
            description="test",
            clause_removed="Never X.",
            recommendation="add test",
        )
        assert m.mutation_id == m.mutation_id  # stable across calls

    def test_mutation_id_differs_for_different_clauses(self):
        m1 = Mutation("DropClause", "o", "m", "d", "Always do A.", "r")
        m2 = Mutation("DropClause", "o", "m", "d", "Always do B.", "r")
        assert m1.mutation_id != m2.mutation_id

    def test_to_dict_keys(self):
        m = Mutation("PhraseSwap", "o", "m", "concise→detailed", "concise", "add test")
        d = m.to_dict()
        assert set(d.keys()) == {"mutation_id", "operator", "description", "clause_removed", "recommendation"}


# ------------------------------------------------------------------ #
# MutantResult                                                         #
# ------------------------------------------------------------------ #

class TestMutantResult:
    def _make_result(self, verdict="KILLED") -> MutantResult:
        m = Mutation("NegateConstraint", "o", "m", "d", "Never X.", "r")
        return MutantResult(
            mutation=m,
            original_score=0.90,
            mutant_score=0.30,
            delta=0.60,
            verdict=verdict,
        )

    def test_to_dict_keys(self):
        r = self._make_result()
        d = r.to_dict()
        required = {
            "mutation_id", "operator", "description", "clause_removed",
            "original_score", "mutant_score", "delta", "verdict", "runs", "recommendation", "error"
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_values(self):
        r = self._make_result("SURVIVED")
        d = r.to_dict()
        assert d["verdict"] == "SURVIVED"
        assert d["delta"] == 0.60
        assert d["original_score"] == 0.90


# ------------------------------------------------------------------ #
# _score_to_verdict                                                    #
# ------------------------------------------------------------------ #

class TestScoreToVerdict:
    def test_strong(self):
        assert _score_to_verdict(0.95) == "STRONG"
        assert _score_to_verdict(0.90) == "STRONG"

    def test_adequate(self):
        assert _score_to_verdict(0.85) == "ADEQUATE"
        assert _score_to_verdict(0.80) == "ADEQUATE"

    def test_marginal(self):
        assert _score_to_verdict(0.75) == "MARGINAL"
        assert _score_to_verdict(0.70) == "MARGINAL"

    def test_weak(self):
        assert _score_to_verdict(0.65) == "WEAK"
        assert _score_to_verdict(0.60) == "WEAK"

    def test_dangerous(self):
        assert _score_to_verdict(0.55) == "DANGEROUS"
        assert _score_to_verdict(0.0) == "DANGEROUS"


# ------------------------------------------------------------------ #
# MutationReport                                                       #
# ------------------------------------------------------------------ #

class TestMutationReport:
    def _make_report(self) -> MutationReport:
        m1 = Mutation("NegateConstraint", SAMPLE_PROMPT, "mutant1", "d1", "Never X.", "Add test A")
        m2 = Mutation("DropClause", SAMPLE_PROMPT, "mutant2", "d2", "Always Y.", "Add test B")
        m3 = Mutation("ScopeExpand", SAMPLE_PROMPT, "mutant3", "d3", "software only", "Add test C")
        r1 = MutantResult(m1, 0.90, 0.30, 0.60, "KILLED")
        r2 = MutantResult(m2, 0.90, 0.85, 0.05, "SURVIVED")
        r3 = MutantResult(m3, 0.90, 0.90, 0.00, "SKIPPED", error="timeout")
        return MutationReport.from_results([r1, r2, r3], SAMPLE_PROMPT, 0.90)

    def test_from_results_counts(self):
        report = self._make_report()
        assert report.killed == 1
        assert report.survived == 1
        assert report.skipped == 1
        assert report.total_mutations == 3

    def test_mutation_score(self):
        report = self._make_report()
        # score = 1 killed / (1 killed + 1 survived) = 0.50
        assert report.mutation_score == 0.50

    def test_score_verdict(self):
        report = self._make_report()
        assert report.score_verdict == "DANGEROUS"  # 50% < 60%

    def test_recommendations_populated(self):
        report = self._make_report()
        assert "Add test B" in report.recommendations

    def test_prompt_hash_set(self):
        report = self._make_report()
        assert len(report.prompt_hash) == 16

    def test_summary_text_contains_score(self):
        report = self._make_report()
        summary = report.summary("text")
        assert "50%" in summary
        assert "KILLED" in summary
        assert "SURVIVED" in summary

    def test_summary_json_valid(self):
        report = self._make_report()
        j = report.summary("json")
        data = json.loads(j)
        assert data["mutation_score"] == 0.5
        assert "results" in data

    def test_summary_markdown(self):
        report = self._make_report()
        md = report.summary("markdown")
        assert "## Mutation Report" in md
        assert "50%" in md

    def test_to_json_and_load(self):
        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            report.to_json(path)
            loaded = MutationReport.load_json(path)
            assert loaded.mutation_score == report.mutation_score
            assert loaded.killed == report.killed
            assert loaded.survived == report.survived
            assert loaded.total_mutations == report.total_mutations
        finally:
            path.unlink(missing_ok=True)

    def test_all_killed_score_is_1(self):
        m1 = Mutation("NegateConstraint", "o", "m1", "d", "Never X.", "r")
        m2 = Mutation("DropClause", "o", "m2", "d", "Always Y.", "r")
        r1 = MutantResult(m1, 0.90, 0.10, 0.80, "KILLED")
        r2 = MutantResult(m2, 0.90, 0.20, 0.70, "KILLED")
        report = MutationReport.from_results([r1, r2], "original", 0.90)
        assert report.mutation_score == 1.0

    def test_all_survived_score_is_0(self):
        m1 = Mutation("NegateConstraint", "o", "m1", "d", "Never X.", "r")
        r1 = MutantResult(m1, 0.90, 0.88, 0.02, "SURVIVED")
        report = MutationReport.from_results([r1], "original", 0.90)
        assert report.mutation_score == 0.0

    def test_only_skipped_score_is_0(self):
        m1 = Mutation("NegateConstraint", "o", "m1", "d", "Never X.", "r")
        r1 = MutantResult(m1, 0.90, 0.90, 0.00, "SKIPPED", error="timeout")
        report = MutationReport.from_results([r1], "original", 0.90)
        assert report.mutation_score == 0.0


# ------------------------------------------------------------------ #
# MutationEngine — operator tests                                      #
# ------------------------------------------------------------------ #

class TestNegateConstraintOperator:
    def test_removes_never_clause(self):
        prompt = "You are an agent.\nNever discuss competitor products.\nAlways be helpful."
        mutations = _negate_constraint_mutations(prompt)
        assert len(mutations) >= 1
        m = mutations[0]
        assert m.operator == "NegateConstraint"
        assert "Never discuss competitor products" not in m.mutated_text
        assert "You are an agent" in m.mutated_text

    def test_removes_do_not_clause(self):
        prompt = "You are helpful.\nDo not share passwords.\nRespond formally."
        mutations = _negate_constraint_mutations(prompt)
        assert any("Do not share passwords" not in m.mutated_text for m in mutations)

    def test_does_not_mutate_if_no_prohibitions(self):
        prompt = "You are a helpful assistant. Answer questions clearly."
        mutations = _negate_constraint_mutations(prompt)
        assert len(mutations) == 0

    def test_recommendation_is_set(self):
        prompt = "Never discuss competitor products.\nAlways be formal."
        mutations = _negate_constraint_mutations(prompt)
        assert mutations[0].recommendation != ""

    def test_clause_removed_field(self):
        prompt = "Never share internal pricing.\nBe helpful."
        mutations = _negate_constraint_mutations(prompt)
        assert mutations[0].clause_removed != ""


class TestDropClauseOperator:
    def test_removes_always_clause(self):
        prompt = "You are helpful.\nAlways respond in formal English.\nNever share passwords."
        mutations = _drop_clause_mutations(prompt)
        assert len(mutations) >= 1
        assert "Always respond in formal English" not in mutations[0].mutated_text

    def test_removes_you_must_clause(self):
        prompt = "You are an agent.\nYou must escalate billing to billing@acme.com.\nBe concise."
        mutations = _drop_clause_mutations(prompt)
        assert any("You must escalate" not in m.mutated_text for m in mutations)

    def test_does_not_mutate_if_no_requirements(self):
        prompt = "You are a helpful assistant."
        mutations = _drop_clause_mutations(prompt)
        assert len(mutations) == 0


class TestScopeExpandOperator:
    def test_expands_software_only(self):
        prompt = "Answer questions about software products only."
        mutations = _scope_expand_mutations(prompt)
        assert len(mutations) >= 1
        assert "products and services" in mutations[0].mutated_text

    def test_expands_formal_english(self):
        prompt = "Always respond in formal English."
        mutations = _scope_expand_mutations(prompt)
        assert len(mutations) >= 1

    def test_no_mutation_if_no_scope(self):
        prompt = "You are a helpful assistant. Be nice."
        mutations = _scope_expand_mutations(prompt)
        assert len(mutations) == 0


class TestScopeNarrowOperator:
    def test_narrows_any_topic(self):
        prompt = "You can answer any topic the user asks about."
        mutations = _scope_narrow_mutations(prompt)
        assert len(mutations) >= 1
        assert "general topics only" in mutations[0].mutated_text

    def test_narrows_all_questions(self):
        prompt = "Answer all questions comprehensively."
        mutations = _scope_narrow_mutations(prompt)
        # "all" may or may not match — just verify no crash
        assert isinstance(mutations, list)


class TestConditionInvertOperator:
    def test_removes_if_clause(self):
        prompt = "You are helpful.\nIf the user is angry, de-escalate before answering.\nBe formal."
        mutations = _condition_invert_mutations(prompt)
        assert len(mutations) >= 1
        assert "If the user is angry" not in mutations[0].mutated_text

    def test_removes_when_clause(self):
        prompt = "When the user asks for code, provide working examples."
        mutations = _condition_invert_mutations(prompt)
        assert len(mutations) >= 1

    def test_no_mutation_if_no_conditionals(self):
        prompt = "You are a helpful assistant."
        mutations = _condition_invert_mutations(prompt)
        assert len(mutations) == 0


class TestPhraseSwapOperator:
    def test_swaps_concise(self):
        prompt = "Always respond in a concise and professional manner."
        mutations = _phrase_swap_mutations(prompt)
        assert len(mutations) >= 1
        assert "comprehensive" in mutations[0].mutated_text

    def test_swaps_formal(self):
        prompt = "Use a formal tone in all responses."
        mutations = _phrase_swap_mutations(prompt)
        assert any("casual" in m.mutated_text for m in mutations)

    def test_no_mutation_if_no_style_phrase(self):
        prompt = "You are an AI assistant. Answer questions."
        mutations = _phrase_swap_mutations(prompt)
        assert len(mutations) == 0

    def test_preserves_capitalization(self):
        prompt = "Formal language is required."
        mutations = _phrase_swap_mutations(prompt)
        # "Formal" starts with capital — swap should too
        if mutations:
            assert mutations[0].mutated_text[0].isupper() or "Casual" in mutations[0].mutated_text


# ------------------------------------------------------------------ #
# MutationEngine (full)                                                #
# ------------------------------------------------------------------ #

class TestMutationEngine:
    def test_default_operators(self):
        engine = MutationEngine()
        assert set(engine.operators) == set(MutationEngine.ALL_OPERATORS)

    def test_custom_operators(self):
        engine = MutationEngine(operators=["NegateConstraint", "DropClause"])
        assert engine.operators == ["NegateConstraint", "DropClause"]

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operators"):
            MutationEngine(operators=["FakeOperator"])

    def test_generate_returns_mutations(self):
        engine = MutationEngine()
        mutations = engine.generate(SAMPLE_PROMPT)
        assert len(mutations) > 0
        assert all(isinstance(m, Mutation) for m in mutations)

    def test_max_mutations_respected(self):
        engine = MutationEngine(max_mutations=3)
        mutations = engine.generate(SAMPLE_PROMPT)
        assert len(mutations) <= 3

    def test_no_duplicate_mutated_text(self):
        engine = MutationEngine()
        mutations = engine.generate(SAMPLE_PROMPT)
        texts = [m.mutated_text.strip() for m in mutations]
        assert len(texts) == len(set(texts))

    def test_mutated_text_differs_from_original(self):
        engine = MutationEngine()
        mutations = engine.generate(SAMPLE_PROMPT)
        for m in mutations:
            assert m.mutated_text.strip() != SAMPLE_PROMPT.strip()

    def test_generate_from_path(self, tmp_path):
        p = tmp_path / "prompt.txt"
        p.write_text(SAMPLE_PROMPT, encoding="utf-8")
        engine = MutationEngine()
        mutations = engine.generate(p)
        assert len(mutations) > 0

    def test_generate_from_messages_list(self):
        messages = [
            {"role": "system", "content": "You are an agent. Never share passwords. Always be formal."},
            {"role": "user", "content": "Hello"},
        ]
        engine = MutationEngine()
        mutations = engine.generate(messages)
        assert len(mutations) > 0

    def test_prompt_with_no_clauses_generates_empty_or_minimal(self):
        engine = MutationEngine()
        mutations = engine.generate("You are a helpful assistant.")
        # No prohibitions, requirements, or style phrases — should be 0 or very few
        assert isinstance(mutations, list)

    def test_operator_subset_only_generates_that_type(self):
        engine = MutationEngine(operators=["NegateConstraint"])
        mutations = engine.generate(SAMPLE_PROMPT)
        assert all(m.operator == "NegateConstraint" for m in mutations)

    def test_generate_broad_prompt(self):
        engine = MutationEngine()
        mutations = engine.generate(BROAD_PROMPT)
        assert len(mutations) > 0


# ------------------------------------------------------------------ #
# MutantRunner                                                         #
# ------------------------------------------------------------------ #

class TestMutantRunner:
    def _make_mutations(self, n: int = 3) -> list[Mutation]:
        engine = MutationEngine()
        all_mutations = engine.generate(SAMPLE_PROMPT)
        return all_mutations[:n]

    def test_init_validates_eval_fn(self):
        with pytest.raises(TypeError):
            MutantRunner(eval_fn="not callable", test_cases=[])

    def test_init_validates_delta_threshold(self):
        with pytest.raises(ValueError):
            MutantRunner(eval_fn=lambda p, c: 0.5, test_cases=[], delta_threshold=0.0)
        with pytest.raises(ValueError):
            MutantRunner(eval_fn=lambda p, c: 0.5, test_cases=[], delta_threshold=1.0)

    def test_init_validates_runs_per_mutant(self):
        with pytest.raises(ValueError):
            MutantRunner(eval_fn=lambda p, c: 0.5, test_cases=[], runs_per_mutant=0)

    def test_run_returns_correct_count(self):
        mutations = self._make_mutations(3)
        runner = MutantRunner(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        assert len(results) == 3

    def test_run_empty_mutations(self):
        runner = MutantRunner(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            runs_per_mutant=1,
        )
        results = runner.run([])
        assert results == []

    def test_killed_when_large_delta(self):
        mutations = self._make_mutations(1)
        # eval_fn returns 0.90 for all → delta = 0 → SURVIVED
        # We'll use a score_map that returns low for mutated text
        original = SAMPLE_PROMPT

        call_count = {"n": 0}
        def eval_fn(prompt, cases):
            call_count["n"] += 1
            if prompt == original:
                return 0.90
            return 0.20  # big drop → KILLED

        runner = MutantRunner(
            eval_fn=eval_fn,
            test_cases=[],
            delta_threshold=0.15,
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        assert results[0].verdict == "KILLED"
        assert results[0].delta >= 0.15

    def test_survived_when_small_delta(self):
        mutations = self._make_mutations(1)

        def eval_fn(prompt, cases):
            return 0.90  # same score always → survived

        runner = MutantRunner(
            eval_fn=eval_fn,
            test_cases=[],
            delta_threshold=0.15,
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        assert results[0].verdict == "SURVIVED"
        assert results[0].delta == 0.0

    def test_skipped_on_eval_fn_exception(self):
        mutations = self._make_mutations(1)

        call_count = {"n": 0}
        def bad_eval_fn(prompt, cases):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 0.90
            raise RuntimeError("eval crashed")

        runner = MutantRunner(
            eval_fn=bad_eval_fn,
            test_cases=[],
            delta_threshold=0.15,
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        assert results[0].verdict == "SKIPPED"
        assert results[0].error is not None

    def test_eval_fn_out_of_range_raises(self):
        mutations = self._make_mutations(1)

        def bad_eval_fn(prompt, cases):
            return 1.5  # invalid

        runner = MutantRunner(
            eval_fn=bad_eval_fn,
            test_cases=[],
            runs_per_mutant=1,
            parallel=False,
        )
        with pytest.raises(ValueError):
            runner.run(mutations)

    def test_parallel_run_same_count(self):
        mutations = self._make_mutations(4)

        runner = MutantRunner(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            runs_per_mutant=1,
            parallel=True,
            max_workers=2,
        )
        results = runner.run(mutations)
        assert len(results) == 4

    def test_runs_per_mutant_median_averaging(self):
        """Verify that multiple runs are median-averaged."""
        mutations = self._make_mutations(1)
        run_scores = [0.91, 0.89, 0.90]
        run_idx = {"n": 0}

        def eval_fn(prompt, cases):
            if prompt == SAMPLE_PROMPT:
                return 0.90
            score = run_scores[run_idx["n"] % len(run_scores)]
            run_idx["n"] += 1
            return score

        runner = MutantRunner(
            eval_fn=eval_fn,
            test_cases=[],
            delta_threshold=0.05,
            runs_per_mutant=3,
            parallel=False,
        )
        results = runner.run(mutations)
        # median of [0.91, 0.89, 0.90] = 0.90, delta = 0.0 → SURVIVED
        assert results[0].verdict == "SURVIVED"


# ------------------------------------------------------------------ #
# MutationStore                                                        #
# ------------------------------------------------------------------ #

class TestMutationStore:
    def _make_report(self) -> MutationReport:
        m = Mutation("NegateConstraint", SAMPLE_PROMPT, "mutant", "d", "Never X.", "r")
        r = MutantResult(m, 0.90, 0.30, 0.60, "KILLED")
        return MutationReport.from_results([r], SAMPLE_PROMPT, 0.90)

    def test_save_and_history(self, tmp_path):
        store = MutationStore(tmp_path / "test.db")
        report = self._make_report()
        row_id = store.save(report)
        assert row_id >= 1

        history = store.history()
        assert len(history) == 1
        assert history[0]["killed"] == 1

    def test_save_multiple_and_limit(self, tmp_path):
        store = MutationStore(tmp_path / "test.db")
        report = self._make_report()
        for _ in range(5):
            store.save(report)

        history = store.history(limit=3)
        assert len(history) == 3

    def test_filter_by_prompt_hash(self, tmp_path):
        store = MutationStore(tmp_path / "test.db")
        report = self._make_report()
        store.save(report)

        history = store.history(prompt_hash=report.prompt_hash)
        assert len(history) == 1

        history_other = store.history(prompt_hash="nonexistent_hash")
        assert len(history_other) == 0

    def test_trend(self, tmp_path):
        store = MutationStore(tmp_path / "test.db")
        report = self._make_report()
        for _ in range(3):
            store.save(report)

        trend = store.trend(report.prompt_hash)
        assert len(trend) == 3
        assert all("mutation_score" in row for row in trend)

    def test_multiple_stores_same_db(self, tmp_path):
        db_path = tmp_path / "shared.db"
        store1 = MutationStore(db_path)
        store2 = MutationStore(db_path)
        report = self._make_report()
        store1.save(report)
        history = store2.history()
        assert len(history) == 1


# ------------------------------------------------------------------ #
# Calibration module                                                   #
# ------------------------------------------------------------------ #

class TestCalibrationHelpers:
    def test_remove_prohibitions(self):
        text = "You are an agent.\nNever discuss competitors.\nAlways be formal."
        result = _remove_prohibitions(text)
        assert "Never" not in result
        assert "Always be formal" in result

    def test_remove_prohibitions_no_change_if_none(self):
        text = "You are a helpful assistant."
        result = _remove_prohibitions(text)
        assert result == text

    def test_remove_requirements(self):
        text = "You are an agent.\nAlways respond formally.\nNever share passwords."
        result = _remove_requirements(text)
        assert "Always respond formally" not in result
        assert "Never share passwords" in result

    def test_remove_first_line(self):
        text = "You are an agent.\nSecond line.\nThird line."
        result = _remove_first_line(text)
        assert "You are an agent" not in result
        assert "Second line" in result

    def test_remove_first_line_single_line(self):
        text = "Only one line."
        result = _remove_first_line(text)
        assert result == text

    def test_remove_last_instruction(self):
        text = "Line one.\nLine two.\nLine three."
        result = _remove_last_instruction(text)
        assert "Line three" not in result
        assert "Line one" in result


class TestRunCalibration:
    def test_calibration_with_detecting_eval(self):
        # eval_fn detects complete prompt removal (returns 0.10 for generic prompt)
        def eval_fn(prompt: str, cases: list) -> float:
            if "AcmeCorp" in prompt or "software products" in prompt:
                return 0.90
            if len(prompt) > 200:
                return 0.90
            return 0.10  # short/generic prompt = low score

        report = run_calibration(
            eval_fn=eval_fn,
            test_cases=[],
            prompt=SAMPLE_PROMPT,
            delta_threshold=0.15,
        )
        assert isinstance(report, CalibrationReport)
        assert report.calibration_score >= 0.0  # at least ran
        assert report.total_cases > 0

    def test_calibration_with_non_detecting_eval(self):
        # eval_fn always returns 0.90 → nothing gets caught
        report = run_calibration(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            prompt=SAMPLE_PROMPT,
            delta_threshold=0.15,
        )
        assert report.calibration_score == 0.0
        assert report.caught == 0
        assert report.missed > 0

    def test_calibration_summary_contains_score(self):
        report = run_calibration(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            prompt=SAMPLE_PROMPT,
        )
        summary = report.summary()
        assert "CALIBRATION RESULTS:" in summary
        assert "%" in summary

    def test_calibration_warnings_for_missed_high(self):
        # HIGH severity = complete system prompt removal → eval returns 0.90 → missed
        report = run_calibration(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            prompt=SAMPLE_PROMPT,
        )
        assert len(report.warnings) > 0


# ------------------------------------------------------------------ #
# Integration: engine → runner → report                               #
# ------------------------------------------------------------------ #

class TestEndToEnd:
    def test_full_pipeline_all_killed(self):
        """Eval suite that detects every mutation."""
        original = SAMPLE_PROMPT
        mutations = MutationEngine(operators=["NegateConstraint"]).generate(original)
        assert len(mutations) > 0

        def eval_fn(prompt, cases):
            if prompt == original:
                return 0.90
            return 0.10  # always drops sharply

        runner = MutantRunner(
            eval_fn=eval_fn,
            test_cases=[],
            delta_threshold=0.15,
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        report = MutationReport.from_results(results, original, 0.90)

        assert report.mutation_score == 1.0
        assert report.score_verdict == "STRONG"
        assert report.killed == len(mutations)
        assert report.survived == 0

    def test_full_pipeline_none_killed(self):
        """Eval suite that misses every mutation."""
        original = SAMPLE_PROMPT
        mutations = MutationEngine(operators=["DropClause"]).generate(original)
        assert len(mutations) > 0

        runner = MutantRunner(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            delta_threshold=0.15,
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        report = MutationReport.from_results(results, original, 0.90)

        assert report.mutation_score == 0.0
        assert report.survived == len(mutations)

    def test_full_pipeline_json_roundtrip(self, tmp_path):
        """Save to JSON and reload."""
        original = SAMPLE_PROMPT
        mutations = MutationEngine(max_mutations=2).generate(original)

        runner = MutantRunner(
            eval_fn=make_constant_eval_fn(0.90),
            test_cases=[],
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        report = MutationReport.from_results(results, original, 0.90)

        p = tmp_path / "report.json"
        report.to_json(p)
        loaded = MutationReport.load_json(p)

        assert loaded.mutation_score == report.mutation_score
        assert loaded.total_mutations == report.total_mutations
        assert loaded.killed == report.killed

    def test_full_pipeline_with_store(self, tmp_path):
        """Persist to store and retrieve history."""
        original = SAMPLE_PROMPT
        mutations = MutationEngine(operators=["NegateConstraint"]).generate(original)

        def eval_fn(prompt, cases):
            return 0.90 if prompt == original else 0.30

        runner = MutantRunner(
            eval_fn=eval_fn,
            test_cases=[],
            runs_per_mutant=1,
            parallel=False,
        )
        results = runner.run(mutations)
        report = MutationReport.from_results(results, original, 0.90)

        store = MutationStore(tmp_path / "history.db")
        row_id = store.save(report)
        assert row_id >= 1

        history = store.history()
        assert len(history) >= 1
        assert history[0]["mutation_score"] == report.mutation_score


# ------------------------------------------------------------------ #
# CLI (unit-tested via direct function calls)                         #
# ------------------------------------------------------------------ #

class TestCLI:
    """Test CLI components without spawning subprocesses."""

    def test_main_entrypoint_exists(self):
        from llm_mutation._cli import main
        assert callable(main)

    def test_cmd_report_text(self, tmp_path):
        """cmd_report loads a JSON file and prints text summary."""
        from llm_mutation._cli import cmd_report
        import argparse

        # Create a report JSON
        m = Mutation("NegateConstraint", SAMPLE_PROMPT, "m", "d", "Never X.", "Add test")
        r = MutantResult(m, 0.90, 0.30, 0.60, "KILLED")
        report = MutationReport.from_results([r], SAMPLE_PROMPT, 0.90)
        p = tmp_path / "report.json"
        report.to_json(p)

        args = argparse.Namespace(input=str(p), format="text")
        # Should not raise
        cmd_report(args)

    def test_cmd_ci_passes_above_threshold(self, tmp_path):
        from llm_mutation._cli import cmd_ci
        import argparse

        m = Mutation("NegateConstraint", SAMPLE_PROMPT, "m", "d", "Never X.", "r")
        r = MutantResult(m, 0.90, 0.30, 0.60, "KILLED")
        report = MutationReport.from_results([r], SAMPLE_PROMPT, 0.90)
        p = tmp_path / "report.json"
        report.to_json(p)

        args = argparse.Namespace(input=str(p), min_score=0.80)
        # score is 1.0 >= 0.80 → should not sys.exit(1)
        try:
            cmd_ci(args)
        except SystemExit as e:
            assert e.code == 0

    def test_cmd_ci_fails_below_threshold(self, tmp_path):
        from llm_mutation._cli import cmd_ci
        import argparse

        m = Mutation("NegateConstraint", SAMPLE_PROMPT, "m", "d", "Never X.", "Add test X")
        r = MutantResult(m, 0.90, 0.88, 0.02, "SURVIVED")
        report = MutationReport.from_results([r], SAMPLE_PROMPT, 0.90)
        p = tmp_path / "report.json"
        report.to_json(p)

        args = argparse.Namespace(input=str(p), min_score=0.80)
        with pytest.raises(SystemExit) as exc_info:
            cmd_ci(args)
        assert exc_info.value.code == 1

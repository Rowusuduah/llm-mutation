# llm-mutation

**Mutation testing for LLM prompts. Find the gaps in your eval suite before production does.**

```bash
pip install llm-mutation
mutate run --prompt prompts/customer_service.txt --eval evals/test_cs.py
```

## The Problem

You have an eval suite. It passes. You ship. Production breaks.

Your eval suite tested 50 specific cases you wrote. It was never tested itself. **llm-mutation tests whether your eval suite would notice if a key constraint was removed, a clause was dropped, or a scope was expanded.**

## Quickstart

```python
from llm_mutation import MutationEngine, MutantRunner, MutationReport

# 1. Generate semantic mutations of your prompt
engine = MutationEngine()
mutations = engine.generate("prompts/customer_service.txt")

# 2. Run your eval suite against each mutant
def my_eval_fn(prompt: str, test_cases: list) -> float:
    # your existing eval logic — returns 0.0-1.0
    ...

runner = MutantRunner(eval_fn=my_eval_fn, test_cases=my_test_cases)
results = runner.run(mutations)

# 3. See your gaps
report = MutationReport.from_results(results, prompt, original_score=0.91)
print(report.summary())
# MUTATION SCORE: 71% (5/7 mutations killed)
# SURVIVING MUTATIONS:
#   ✗ DropClause — "Direct pricing questions to sales@acmecorp.com." removed
#     → ADD TEST CASE: "User asks 'What does the enterprise plan cost?'"
```

## Six Deterministic Mutation Operators

| Operator | What it does |
|----------|--------------|
| `NegateConstraint` | Removes a prohibitive clause ("Never X") |
| `DropClause` | Removes a requirement ("Always X", "You must X") |
| `ScopeExpand` | Widens a scope restriction ("software only" → "products and services") |
| `ScopeNarrow` | Narrows a permission ("any topic" → "general topics only") |
| `ConditionInvert` | Removes a conditional behavior ("if A, then B") |
| `PhraseSwap` | Swaps a style phrase ("concise" ↔ "comprehensive") |

No LLM required for mutation generation — all operators are deterministic text transforms.

## Mutation Score

| Score | Verdict | Meaning |
|-------|---------|---------|
| >= 90% | STRONG | Eval suite is comprehensive |
| 80-89% | ADEQUATE | Good for CI gate |
| 70-79% | MARGINAL | Meaningful gaps |
| 60-69% | WEAK | Significant gaps |
| < 60% | DANGEROUS | Not fit for purpose |

**Recommended minimum for production CI gate: 80%**

## CLI

```bash
# Run mutation test
mutate run --prompt prompts/cs.txt --eval evals/test_cs.py --output report.json

# Generate report
mutate report --input report.json --format markdown

# CI gate (exit 1 if score < 80%)
mutate ci --input report.json --min-score 0.80

# Calibrate your eval suite
mutate calibrate --prompt prompts/cs.txt --eval evals/test_cs.py
```

## GitHub Action

```yaml
- run: pip install llm-mutation
- name: Run mutation tests
  run: |
    mutate run --prompt prompts/cs.txt --eval evals/test_cs.py --output report.json
    mutate ci --input report.json --min-score 0.80
```

## Pattern Foundation

Built on **PAT-045 — Judges 6:36-40 (The Gideon Fleece Inversion Pattern)**.

Gideon designed a two-condition invertible test: fleece wet/ground dry, then fleece dry/ground wet. He wasn't testing God's power — he was testing whether his testing mechanism could discriminate signal from coincidence.

**llm-mutation is the bowlful of water. Your mutation score is your measurement.**

Supporting: PAT-046 (Acts 17:11 — Berean Null Test) → `mutate calibrate`
Supporting: PAT-047 (Numbers 13:25-33 — Twelve Spies Divergence) → `mutate verify-judge`

## License

MIT

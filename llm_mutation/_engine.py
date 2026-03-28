"""
MutationEngine — deterministic semantic mutation operators.

All six operators work purely on text: no LLM required for mutation generation.
The mutations are semantically meaningful (not random character noise) because
they target the logical structure of prompt clauses.

Operators:
  NegateConstraint  — removes/negates a prohibitive clause ("Never X", "Do not X")
  DropClause        — removes a requirement clause entirely ("Always X", "You must X")
  ScopeExpand       — widens a scope restriction ("software only" → "products and services")
  ScopeNarrow       — narrows a permission ("any topic" → "general topics only")
  ConditionInvert   — inverts a conditional behavior ("if A then B" → "if A then not B")
  PhraseSwap        — substitutes key style/behavior phrase ("concise" ↔ "comprehensive")
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from ._models import Mutation, MutationOperator


# ------------------------------------------------------------------ #
# Operator patterns                                                    #
# ------------------------------------------------------------------ #

# Prohibitive clause patterns (NegateConstraint)
_PROHIBIT_PATTERNS = [
    r"(?m)^[ \t]*[Nn]ever\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Dd]o not\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Dd]on't\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Aa]void\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Dd]o not\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Yy]ou must not\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Yy]ou should not\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Rr]efuse\s+.+[.!]?\s*$",
]

# Requirement clause patterns (DropClause)
_REQUIRE_PATTERNS = [
    r"(?m)^[ \t]*[Aa]lways\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Yy]ou must\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Yy]ou should\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Mm]ake sure\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Ee]nsure\s+.+[.!]?\s*$",
    r"(?m)^[ \t]*[Rr]espond\s+.+[.!]?\s*$",
]

# Scope restriction patterns (ScopeExpand)
_SCOPE_PATTERNS = [
    (r"\b(software\s+(?:products?\s+)?only)\b", "products and services"),
    (r"\b(in English only)\b", "in any language"),
    (r"\b(formal English)\b", "casual language"),
    (r"\b(English)\b(?=\s+only|\s+language)", "the user's preferred language"),
    (r"\b(topics? related to \w+)\b", "any relevant topic"),
    (r"\b(company products?)\b(?=\s+only)", "all available products"),
    (r"\b(technical questions?)\b(?=\s+only|\s+and)", "all questions"),
]

# Permission expansion patterns (ScopeNarrow)
_PERMISSION_PATTERNS = [
    (r"\b(any (?:topic|question|subject|issue)s?)\b", "general topics only"),
    (r"\b(all (?:topic|question|subject)s?)\b", "common topics only"),
    (r"\b(anything)\b(?=\s+(?:the user|they|you))", "straightforward requests"),
    (r"\b(broadly)\b", "narrowly"),
]

# Conditional behavior patterns (ConditionInvert)
_CONDITIONAL_PATTERNS = [
    r"(?m)^[ \t]*[Ii]f\s+.+,\s*.+[.!]?\s*$",
    r"(?m)^[ \t]*[Ww]hen\s+.+,\s*.+[.!]?\s*$",
]

# Style/behavior phrase swaps (PhraseSwap)
_PHRASE_SWAPS = [
    ("concise", "comprehensive"),
    ("comprehensive", "concise"),
    ("formal", "casual"),
    ("casual", "formal"),
    ("brief", "detailed"),
    ("detailed", "brief"),
    ("professional", "informal"),
    ("polite", "direct"),
    ("friendly", "clinical"),
    ("structured", "free-form"),
]


# ------------------------------------------------------------------ #
# MutationEngine                                                       #
# ------------------------------------------------------------------ #

class MutationEngine:
    """
    Generates deterministic semantic mutations of an LLM prompt.
    No LLM required — all operators work on text structure.

    Args:
        operators:       subset of the 6 operators to apply (default: all 6)
        max_mutations:   cap on total mutations to prevent combinatorial explosion
        prompt_format:   "string" | "messages" | "auto"
    """

    ALL_OPERATORS: list[MutationOperator] = [
        "NegateConstraint",
        "DropClause",
        "ScopeExpand",
        "ScopeNarrow",
        "ConditionInvert",
        "PhraseSwap",
    ]

    def __init__(
        self,
        operators: list[MutationOperator] | None = None,
        max_mutations: int = 20,
        prompt_format: str = "auto",
    ) -> None:
        self.operators = operators or list(self.ALL_OPERATORS)
        self.max_mutations = max_mutations
        self.prompt_format = prompt_format

        unknown = set(self.operators) - set(self.ALL_OPERATORS)
        if unknown:
            raise ValueError(f"Unknown operators: {unknown}")

    def generate(
        self,
        prompt: Union[str, list, Path],
    ) -> list[Mutation]:
        """
        Generate semantic mutations of the given prompt.

        Args:
            prompt: prompt string, list of message dicts, or Path to a .txt file

        Returns:
            List of Mutation objects (deduplicated, capped at max_mutations)
        """
        text = self._resolve_prompt(prompt)
        mutations: list[Mutation] = []

        if "NegateConstraint" in self.operators:
            mutations.extend(_negate_constraint_mutations(text))
        if "DropClause" in self.operators:
            mutations.extend(_drop_clause_mutations(text))
        if "ScopeExpand" in self.operators:
            mutations.extend(_scope_expand_mutations(text))
        if "ScopeNarrow" in self.operators:
            mutations.extend(_scope_narrow_mutations(text))
        if "ConditionInvert" in self.operators:
            mutations.extend(_condition_invert_mutations(text))
        if "PhraseSwap" in self.operators:
            mutations.extend(_phrase_swap_mutations(text))

        # Deduplicate by mutated text
        seen: set[str] = set()
        unique: list[Mutation] = []
        for m in mutations:
            key = m.mutated_text.strip()
            if key != text.strip() and key not in seen:
                seen.add(key)
                unique.append(m)

        return unique[: self.max_mutations]

    def _resolve_prompt(self, prompt: Union[str, list, Path]) -> str:
        if isinstance(prompt, Path):
            return prompt.read_text(encoding="utf-8")
        if isinstance(prompt, list):
            # OpenAI-style messages list
            parts = []
            for msg in prompt:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(f"[{role.upper()}] {content}")
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            parts.append(f"[{role.upper()}] {block['text']}")
            return "\n\n".join(parts)
        return str(prompt)


# ------------------------------------------------------------------ #
# Per-operator generators                                             #
# ------------------------------------------------------------------ #

def _negate_constraint_mutations(text: str) -> list[Mutation]:
    mutations = []
    for pattern in _PROHIBIT_PATTERNS:
        for match in re.finditer(pattern, text):
            clause = match.group(0)
            # Remove the clause entirely
            mutated = text[: match.start()] + text[match.end() :]
            # Clean up double blank lines
            mutated = re.sub(r"\n{3,}", "\n\n", mutated).strip()
            if mutated != text.strip():
                mutations.append(
                    Mutation(
                        operator="NegateConstraint",
                        original_text=text,
                        mutated_text=mutated,
                        description=f'Prohibitive clause removed: "{clause.strip()}"',
                        clause_removed=clause.strip(),
                        recommendation=_recommend_for_negation(clause.strip()),
                    )
                )
    return mutations


def _drop_clause_mutations(text: str) -> list[Mutation]:
    mutations = []
    for pattern in _REQUIRE_PATTERNS:
        for match in re.finditer(pattern, text):
            clause = match.group(0)
            mutated = text[: match.start()] + text[match.end() :]
            mutated = re.sub(r"\n{3,}", "\n\n", mutated).strip()
            if mutated != text.strip():
                mutations.append(
                    Mutation(
                        operator="DropClause",
                        original_text=text,
                        mutated_text=mutated,
                        description=f'Requirement clause dropped: "{clause.strip()}"',
                        clause_removed=clause.strip(),
                        recommendation=_recommend_for_drop(clause.strip()),
                    )
                )
    return mutations


def _scope_expand_mutations(text: str) -> list[Mutation]:
    mutations = []
    for pattern, replacement in _SCOPE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            original_phrase = match.group(0)
            mutated = text[: match.start()] + replacement + text[match.end() :]
            mutations.append(
                Mutation(
                    operator="ScopeExpand",
                    original_text=text,
                    mutated_text=mutated,
                    description=f'Scope widened: "{original_phrase}" \u2192 "{replacement}"',
                    clause_removed=original_phrase,
                    recommendation=(
                        f'Test a request that would be rejected by the original scope '
                        f'restriction "{original_phrase}" — verify the LLM still enforces scope.'
                    ),
                )
            )
    return mutations


def _scope_narrow_mutations(text: str) -> list[Mutation]:
    mutations = []
    for pattern, replacement in _PERMISSION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            original_phrase = match.group(0)
            mutated = text[: match.start()] + replacement + text[match.end() :]
            mutations.append(
                Mutation(
                    operator="ScopeNarrow",
                    original_text=text,
                    mutated_text=mutated,
                    description=f'Scope narrowed: "{original_phrase}" \u2192 "{replacement}"',
                    clause_removed=original_phrase,
                    recommendation=(
                        f'Test a broad question that the original "{original_phrase}" '
                        f"permission would have allowed — verify no over-refusal."
                    ),
                )
            )
    return mutations


def _condition_invert_mutations(text: str) -> list[Mutation]:
    mutations = []
    for pattern in _CONDITIONAL_PATTERNS:
        for match in re.finditer(pattern, text):
            clause = match.group(0).strip()
            # Simple inversion: drop the conditional instruction
            mutated = text[: match.start()] + text[match.end() :]
            mutated = re.sub(r"\n{3,}", "\n\n", mutated).strip()
            if mutated != text.strip():
                mutations.append(
                    Mutation(
                        operator="ConditionInvert",
                        original_text=text,
                        mutated_text=mutated,
                        description=f'Conditional behavior removed: "{clause}"',
                        clause_removed=clause,
                        recommendation=(
                            f'Test the condition trigger described in "{clause}" '
                            f"— verify the conditional behavior is present."
                        ),
                    )
                )
    return mutations


def _phrase_swap_mutations(text: str) -> list[Mutation]:
    mutations = []
    for original_phrase, swap_phrase in _PHRASE_SWAPS:
        # Case-insensitive word boundary match
        pattern = r"\b" + re.escape(original_phrase) + r"\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found = match.group(0)
            # Preserve capitalization of first letter
            if found[0].isupper():
                replacement = swap_phrase[0].upper() + swap_phrase[1:]
            else:
                replacement = swap_phrase
            mutated = text[: match.start()] + replacement + text[match.end() :]
            mutations.append(
                Mutation(
                    operator="PhraseSwap",
                    original_text=text,
                    mutated_text=mutated,
                    description=f'Style phrase swapped: "{found}" \u2192 "{replacement}"',
                    clause_removed=found,
                    recommendation=(
                        f'Test that the LLM output style matches "{found}" '
                        f"(not the mutated \"{replacement}\") — verify style constraint is enforced."
                    ),
                )
            )
    return mutations


# ------------------------------------------------------------------ #
# Recommendation generators                                           #
# ------------------------------------------------------------------ #

def _recommend_for_negation(clause: str) -> str:
    # Extract the core action from the clause for recommendation text
    clean = re.sub(r"^(never|do not|don't|avoid|you must not|you should not|refuse)\s+",
                   "", clause, flags=re.IGNORECASE).rstrip(".!").strip()
    return (
        f'Add test case where user attempts to: {clean}. '
        f"Expected: LLM refuses/redirects."
    )


def _recommend_for_drop(clause: str) -> str:
    clean = re.sub(r"^(always|you must|you should|make sure|ensure|respond)\s+",
                   "", clause, flags=re.IGNORECASE).rstrip(".!").strip()
    return (
        f'Add test case verifying the LLM does: {clean}. '
        f"Expected: behavior present in every response."
    )

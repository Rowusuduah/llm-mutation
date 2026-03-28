"""
SQLite result store for llm-mutation.

Persists mutation reports for trend tracking and CI history.
Zero external dependencies — uses Python's built-in sqlite3.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from ._models import MutationReport


_SCHEMA = """
CREATE TABLE IF NOT EXISTS mutation_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    prompt_preview  TEXT,
    mutation_score  REAL NOT NULL,
    score_verdict   TEXT NOT NULL,
    killed          INTEGER NOT NULL,
    survived        INTEGER NOT NULL,
    skipped         INTEGER NOT NULL,
    total_mutations INTEGER NOT NULL,
    original_score  REAL NOT NULL,
    report_json     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_hash ON mutation_runs (prompt_hash);
CREATE INDEX IF NOT EXISTS idx_run_at      ON mutation_runs (run_at);
"""


class MutationStore:
    """
    Append-only SQLite store for mutation reports.

    Args:
        db_path: path to SQLite database file. Defaults to ./mutation_history.db
    """

    def __init__(self, db_path: Path | str = "mutation_history.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, report: MutationReport) -> int:
        """Persist a report. Returns the new row ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO mutation_runs
                  (run_at, prompt_hash, prompt_preview, mutation_score, score_verdict,
                   killed, survived, skipped, total_mutations, original_score, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.generated_at.isoformat(),
                    report.prompt_hash,
                    report.prompt_preview,
                    report.mutation_score,
                    report.score_verdict,
                    report.killed,
                    report.survived,
                    report.skipped,
                    report.total_mutations,
                    report.original_score,
                    report.to_json_str(),
                ),
            )
            return cursor.lastrowid

    def history(
        self,
        prompt_hash: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Return recent runs, optionally filtered by prompt hash."""
        with self._connect() as conn:
            if prompt_hash:
                rows = conn.execute(
                    "SELECT * FROM mutation_runs WHERE prompt_hash = ? "
                    "ORDER BY run_at DESC LIMIT ?",
                    (prompt_hash, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM mutation_runs ORDER BY run_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def trend(self, prompt_hash: str, limit: int = 10) -> list[dict]:
        """Return mutation_score trend for a specific prompt."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_at, mutation_score, score_verdict, killed, survived "
                "FROM mutation_runs WHERE prompt_hash = ? "
                "ORDER BY run_at ASC LIMIT ?",
                (prompt_hash, limit),
            ).fetchall()
        return [dict(r) for r in rows]

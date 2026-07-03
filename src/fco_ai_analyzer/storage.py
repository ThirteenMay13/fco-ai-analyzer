from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import ParsedLog


class FCOStorage:
    """SQLite-backed storage for parsed FC Online logs."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_index)
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    line_number INTEGER NOT NULL,
                    bait_sequence TEXT NOT NULL,
                    outcome INTEGER NOT NULL,
                    label TEXT,
                    raw_text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    UNIQUE(session_id, attempt_index)
                );

                CREATE TABLE IF NOT EXISTS bait_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id INTEGER NOT NULL,
                    position_in_attempt INTEGER NOT NULL,
                    value INTEGER NOT NULL,
                    line_number INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    FOREIGN KEY(attempt_id) REFERENCES attempts(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL UNIQUE,
                    source_name TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_attempts_session_id ON attempts(session_id);
                CREATE INDEX IF NOT EXISTS idx_attempts_outcome ON attempts(outcome);
                CREATE INDEX IF NOT EXISTS idx_bait_events_attempt_id ON bait_events(attempt_id);
                """
            )

    def max_session_index(self) -> int:
        self.initialize()
        with self.connect() as connection:
            row = connection.execute("SELECT COALESCE(MAX(session_index), -1) AS max_idx FROM sessions").fetchone()
            assert row is not None
            return int(row["max_idx"])

    def register_ingestion(self, content_hash: str, source_name: str | None = None) -> bool:
        self.initialize()
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO ingestions(content_hash, source_name) VALUES (?, ?)",
                (content_hash, source_name),
            )
            return bool(cursor.rowcount)

    def reset_all(self) -> None:
        self.initialize()
        with self.connect() as connection:
            connection.execute("DELETE FROM bait_events")
            connection.execute("DELETE FROM attempts")
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM ingestions")

    def upsert_parsed_log(self, parsed_log: ParsedLog) -> None:
        self.initialize()
        with self.connect() as connection:
            for session in parsed_log.sessions:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO sessions(session_index) VALUES (?)",
                    (session.session_index,),
                )
                if cursor.lastrowid:
                    session_id = cursor.lastrowid
                else:
                    row = connection.execute(
                        "SELECT id FROM sessions WHERE session_index = ?",
                        (session.session_index,),
                    ).fetchone()
                    assert row is not None
                    session_id = int(row["id"])

                for attempt in session.attempts:
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO attempts(
                            session_id, attempt_index, line_number, bait_sequence, outcome, label, raw_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            attempt.attempt_index,
                            attempt.line_number,
                            ",".join(str(value) for value in attempt.bait_sequence),
                            attempt.outcome,
                            attempt.label,
                            attempt.raw_text,
                        ),
                    )
                    if not cursor.lastrowid:
                        row = connection.execute(
                            """
                            SELECT id FROM attempts
                            WHERE session_id = ? AND attempt_index = ?
                            """,
                            (session_id, attempt.attempt_index),
                        ).fetchone()
                        assert row is not None
                        attempt_id = int(row["id"])
                    else:
                        attempt_id = cursor.lastrowid

                    connection.execute(
                        "DELETE FROM bait_events WHERE attempt_id = ?",
                        (attempt_id,),
                    )
                    for bait_event in [
                        event for event in session.bait_events if event.attempt_index == attempt.attempt_index
                    ]:
                        connection.execute(
                            """
                            INSERT INTO bait_events(
                                attempt_id, position_in_attempt, value, line_number, raw_text
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                attempt_id,
                                bait_event.position_in_attempt,
                                bait_event.value,
                                bait_event.line_number,
                                bait_event.raw_text,
                            ),
                        )

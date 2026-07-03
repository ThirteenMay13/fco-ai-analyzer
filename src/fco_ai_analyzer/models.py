from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BaitEvent:
    session_index: int
    attempt_index: int
    position_in_attempt: int
    value: int
    line_number: int
    raw_text: str


@dataclass(frozen=True, slots=True)
class MainAttempt:
    session_index: int
    attempt_index: int
    bait_sequence: tuple[int, ...]
    outcome: int
    line_number: int
    raw_text: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.outcome == 4


@dataclass(slots=True)
class LogSession:
    session_index: int
    raw_lines: list[str] = field(default_factory=list)
    bait_events: list[BaitEvent] = field(default_factory=list)
    attempts: list[MainAttempt] = field(default_factory=list)


@dataclass(slots=True)
class ParsedLog:
    sessions: list[LogSession]

    @property
    def attempts(self) -> list[MainAttempt]:
        return [attempt for session in self.sessions for attempt in session.attempts]

    @property
    def bait_events(self) -> list[BaitEvent]:
        return [bait for session in self.sessions for bait in session.bait_events]

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BaitEvent, LogSession, MainAttempt, ParsedLog

TOKEN_PATTERN = re.compile(r"[124=]")
SEPARATOR_PATTERN = re.compile(r"^\s*[=\-]{3,}\s*$")


@dataclass(slots=True)
class ParserState:
    session_index: int = 0
    attempt_index: int = 0


class LogParser:
    """Parse FC Online enhancement logs into sessions and attempts."""

    def parse(self, raw_text: str) -> ParsedLog:
        sessions: list[LogSession] = []
        state = ParserState()
        current_session = LogSession(session_index=state.session_index)

        for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            if SEPARATOR_PATTERN.match(line):
                if current_session.raw_lines or current_session.attempts or current_session.bait_events:
                    sessions.append(current_session)
                state.session_index += 1
                current_session = LogSession(session_index=state.session_index)
                state.attempt_index = 0
                continue

            tokens = TOKEN_PATTERN.findall(line)
            if not tokens:
                current_session.raw_lines.append(raw_line)
                continue

            current_session.raw_lines.append(raw_line)
            current_session, state = self._parse_tokens(
                tokens=tokens,
                raw_line=raw_line,
                line_number=line_number,
                session=current_session,
                state=state,
            )

        if current_session.raw_lines or current_session.attempts or current_session.bait_events:
            sessions.append(current_session)

        return ParsedLog(sessions=sessions)

    def _parse_tokens(
        self,
        tokens: list[str],
        raw_line: str,
        line_number: int,
        session: LogSession,
        state: ParserState,
    ) -> tuple[LogSession, ParserState]:
        bait_buffer: list[int] = []
        awaiting_outcome = False
        bait_snapshot: list[int] = []
        pending_label = self._extract_label(raw_line)

        for token in tokens:
            if token == "=":
                bait_snapshot = bait_buffer.copy()
                bait_buffer.clear()
                awaiting_outcome = True
                continue

            value = int(token)
            if awaiting_outcome:
                attempt = MainAttempt(
                    session_index=session.session_index,
                    attempt_index=state.attempt_index,
                    bait_sequence=tuple(bait_snapshot),
                    outcome=value,
                    line_number=line_number,
                    raw_text=raw_line,
                    label=pending_label,
                )
                session.attempts.append(attempt)
                for position, bait_value in enumerate(bait_snapshot, start=1):
                    session.bait_events.append(
                        BaitEvent(
                            session_index=session.session_index,
                            attempt_index=state.attempt_index,
                            position_in_attempt=position,
                            value=bait_value,
                            line_number=line_number,
                            raw_text=raw_line,
                        )
                    )
                state.attempt_index += 1
                awaiting_outcome = False
                bait_buffer = []
                bait_snapshot = []
            else:
                bait_buffer.append(value)

        return session, state

    @staticmethod
    def _extract_label(raw_line: str) -> str | None:
        match = re.search(r"\(([^)]*)\)", raw_line)
        if match:
            label = match.group(1).strip()
            return label or None
        return None

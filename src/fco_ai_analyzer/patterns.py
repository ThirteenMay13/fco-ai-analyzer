from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2

from .models import MainAttempt, ParsedLog


@dataclass(slots=True)
class PatternStat:
    pattern: tuple[int, ...]
    sample_size: int
    wins: int
    losses: int
    win_rate: float
    confidence: float
    lift: float
    support: float


class PatternMiner:
    """Mine exact and contiguous bait patterns from 1 to 7 tokens."""

    def mine(self, parsed_log: ParsedLog, max_length: int = 7, min_support: int = 1) -> list[PatternStat]:
        attempt_list = parsed_log.attempts
        if not attempt_list:
            return []

        overall_win_rate = sum(1 for attempt in attempt_list if attempt.is_win) / len(attempt_list)
        counts: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0])

        for attempt in attempt_list:
            patterns = self._extract_patterns(attempt.bait_sequence, max_length=max_length)
            for pattern in patterns:
                bucket = counts[pattern]
                bucket[0] += 1
                bucket[1] += 1 if attempt.is_win else 0

        stats: list[PatternStat] = []
        for pattern, (sample_size, wins) in counts.items():
            if sample_size < min_support:
                continue
            losses = sample_size - wins
            win_rate = wins / sample_size if sample_size else 0.0
            confidence = self._confidence_score(sample_size, wins)
            lift = (win_rate / overall_win_rate) if overall_win_rate else 0.0
            support = sample_size / len(attempt_list)
            stats.append(
                PatternStat(
                    pattern=pattern,
                    sample_size=sample_size,
                    wins=wins,
                    losses=losses,
                    win_rate=win_rate,
                    confidence=confidence,
                    lift=lift,
                    support=support,
                )
            )

        stats.sort(key=lambda item: (item.confidence, item.win_rate, item.sample_size), reverse=True)
        return stats

    @staticmethod
    def _extract_patterns(sequence: tuple[int, ...], max_length: int) -> set[tuple[int, ...]]:
        patterns: set[tuple[int, ...]] = set()
        bounded_sequence = sequence[:max_length]
        for length in range(1, min(max_length, len(bounded_sequence)) + 1):
            for start in range(0, len(bounded_sequence) - length + 1):
                patterns.add(tuple(bounded_sequence[start : start + length]))
        if len(sequence) <= max_length:
            patterns.add(sequence)
        return patterns

    @staticmethod
    def _confidence_score(sample_size: int, wins: int) -> float:
        if sample_size == 0:
            return 0.0
        p = wins / sample_size
        entropy = 0.0
        if 0.0 < p < 1.0:
            entropy = -(p * log2(p) + (1 - p) * log2(1 - p))
        return sample_size * (1.0 - entropy)

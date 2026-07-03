from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .models import MainAttempt, ParsedLog


@dataclass(slots=True)
class MarkovPrediction:
    state: tuple[int, ...]
    probability_win: float
    probability_lose: float
    sample_size: int


class MarkovChainAnalyzer:
    """Simple context-to-outcome Markov analyzer for enhancement sequences."""

    def __init__(self, order: int = 3) -> None:
        self.order = order

    def build(self, parsed_log: ParsedLog) -> dict[tuple[int, ...], Counter[int]]:
        transitions: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for attempt in parsed_log.attempts:
            context = tuple(attempt.bait_sequence[-self.order :])
            transitions[context][attempt.outcome] += 1
        return transitions

    def predict(self, parsed_log: ParsedLog, state: tuple[int, ...]) -> MarkovPrediction:
        transitions = self.build(parsed_log)
        counts = transitions.get(state[-self.order :], Counter())
        sample_size = sum(counts.values())
        if sample_size == 0:
            return MarkovPrediction(state=state, probability_win=0.0, probability_lose=0.0, sample_size=0)
        probability_win = counts.get(4, 0) / sample_size
        probability_lose = 1.0 - probability_win
        return MarkovPrediction(
            state=state,
            probability_win=probability_win,
            probability_lose=probability_lose,
            sample_size=sample_size,
        )

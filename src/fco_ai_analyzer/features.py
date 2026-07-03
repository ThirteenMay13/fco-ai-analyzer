from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2
from statistics import mean, pstdev
from typing import Any

from .models import MainAttempt, ParsedLog


@dataclass(slots=True)
class FeatureRow:
    session_index: int
    attempt_index: int
    label: int
    features: dict[str, float]


class FeatureEngineer:
    """Build rich per-attempt feature vectors for supervised learning and scoring."""

    def __init__(self, history_windows: tuple[int, ...] = (1, 2, 3, 5, 10)) -> None:
        self.history_windows = history_windows

    def transform(self, parsed_log: ParsedLog) -> list[FeatureRow]:
        rows: list[FeatureRow] = []
        for session in parsed_log.sessions:
            prior_baits: list[int] = []
            prior_outcomes: list[int] = []
            for attempt in session.attempts:
                features = self.build_context_features(
                    bait_sequence=attempt.bait_sequence,
                    prior_baits=prior_baits,
                    prior_outcomes=prior_outcomes,
                    session_attempts=len(session.attempts),
                    attempt_index=attempt.attempt_index,
                )
                rows.append(
                    FeatureRow(
                        session_index=attempt.session_index,
                        attempt_index=attempt.attempt_index,
                        label=int(attempt.is_win),
                        features=features,
                    )
                )
                prior_baits.extend(attempt.bait_sequence)
                prior_outcomes.append(attempt.outcome)
        return rows

    def build_context_features(
        self,
        bait_sequence: tuple[int, ...],
        prior_baits: list[int],
        prior_outcomes: list[int],
        session_attempts: int,
        attempt_index: int,
    ) -> dict[str, float]:
        preview_attempt = MainAttempt(
            session_index=0,
            attempt_index=attempt_index,
            bait_sequence=bait_sequence,
            outcome=4,
            line_number=0,
            raw_text="",
        )
        return self._build_features(preview_attempt, prior_baits, prior_outcomes, session_attempts)

    def _build_features(
        self,
        attempt: MainAttempt,
        prior_baits: list[int],
        prior_outcomes: list[int],
        session_attempts: int,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        bait_sequence = list(attempt.bait_sequence)
        recent_baits = prior_baits[-10:]
        recent_outcomes = prior_outcomes[-10:]

        features["attempt_index"] = float(attempt.attempt_index)
        features["session_attempts"] = float(session_attempts)
        features["session_prior_attempts"] = float(len(prior_outcomes))
        features["session_prior_wins"] = float(sum(1 for outcome in prior_outcomes if outcome == 4))
        features["session_prior_losses"] = float(sum(1 for outcome in prior_outcomes if outcome != 4))
        features["session_prior_win_rate"] = float(features["session_prior_wins"] / features["session_prior_attempts"] if prior_outcomes else 0.0)
        features["session_remaining_attempts"] = float(max(session_attempts - attempt.attempt_index - 1, 0))
        features["bait_length"] = float(len(bait_sequence))
        features["bait_sum"] = float(sum(bait_sequence)) if bait_sequence else 0.0
        features["bait_mean"] = float(mean(bait_sequence)) if bait_sequence else 0.0
        features["bait_std"] = float(pstdev(bait_sequence)) if len(bait_sequence) > 1 else 0.0
        features["bait_unique_count"] = float(len(set(bait_sequence)))
        features["bait_entropy"] = float(self._entropy(bait_sequence))
        features["bait_repetition_rate"] = float(self._repetition_rate(bait_sequence))
        features["last_outcome_change"] = float(prior_outcomes[-1] - prior_outcomes[-2]) if len(prior_outcomes) > 1 else 0.0

        for value in (1, 2, 4):
            features[f"bait_count_{value}"] = float(bait_sequence.count(value))
            features[f"prior_bait_count_{value}"] = float(prior_baits.count(value))
            features[f"recent_bait_count_{value}"] = float(recent_baits.count(value))
            features[f"prior_outcome_count_{value}"] = float(prior_outcomes.count(value))
            features[f"recent_outcome_count_{value}"] = float(recent_outcomes.count(value))

        for window in self.history_windows:
            window_baits = prior_baits[-window:]
            window_outcomes = prior_outcomes[-window:]
            window_size = len(window_outcomes)
            features[f"window_{window}_bait_count"] = float(len(window_baits))
            features[f"window_{window}_bait_entropy"] = float(self._entropy(window_baits))
            features[f"window_{window}_bait_mean"] = float(mean(window_baits)) if window_baits else 0.0
            features[f"window_{window}_bait_std"] = float(pstdev(window_baits)) if len(window_baits) > 1 else 0.0
            features[f"window_{window}_win_rate"] = float(sum(1 for outcome in window_outcomes if outcome == 4) / window_size) if window_size else 0.0
            features[f"window_{window}_lose_rate"] = float(sum(1 for outcome in window_outcomes if outcome != 4) / window_size) if window_size else 0.0
            features[f"window_{window}_unique_outcomes"] = float(len(set(window_outcomes)))
            features[f"window_{window}_unique_baits"] = float(len(set(window_baits)))

        for index in range(1, 11):
            features[f"prev_bait_{index}"] = float(recent_baits[-index]) if len(recent_baits) >= index else -1.0
            features[f"prev_outcome_{index}"] = float(recent_outcomes[-index]) if len(recent_outcomes) >= index else -1.0

        for index in range(1, 11):
            prefix = bait_sequence[:index]
            features[f"bait_prefix_len_{index}"] = float(len(prefix))
            features[f"bait_prefix_sum_{index}"] = float(sum(prefix)) if prefix else 0.0
            features[f"bait_prefix_mean_{index}"] = float(mean(prefix)) if prefix else 0.0
            features[f"bait_prefix_entropy_{index}"] = float(self._entropy(list(prefix)))

        for value in (1, 2, 4):
            features[f"recent_transition_to_{value}"] = float(self._transition_rate(recent_outcomes, value))
            features[f"prior_transition_to_{value}"] = float(self._transition_rate(prior_outcomes, value))

        features["distance_since_last_4"] = float(self._distance_since(prior_outcomes, 4))
        features["distance_since_last_2"] = float(self._distance_since(prior_outcomes, 2))
        features["distance_since_last_1"] = float(self._distance_since(prior_outcomes, 1))
        features["last_bait_streak"] = float(self._streak_length(prior_baits))
        features["last_outcome_streak"] = float(self._streak_length(prior_outcomes))
        features["rolling_win_rate_3"] = float(self._rolling_rate(prior_outcomes, 3))
        features["rolling_win_rate_5"] = float(self._rolling_rate(prior_outcomes, 5))
        features["rolling_win_rate_10"] = float(self._rolling_rate(prior_outcomes, 10))
        features["rolling_bait_mean_5"] = float(self._rolling_mean(prior_baits, 5))
        features["rolling_bait_std_5"] = float(self._rolling_std(prior_baits, 5))
        features["transition_to_win_rate"] = float(self._transition_rate(prior_outcomes, 4))
        features["transition_to_loss_rate"] = float(self._transition_rate(prior_outcomes, 2))
        features["transition_to_x1_rate"] = float(self._transition_rate(prior_outcomes, 1))

        exact_pattern = tuple(bait_sequence[-7:])
        features["pattern_signature_length"] = float(len(exact_pattern))
        features["pattern_signature_hash"] = float(abs(hash(exact_pattern)) % 10_000)
        features["pattern_token_count"] = float(len(exact_pattern))
        features["pattern_token_sum"] = float(sum(exact_pattern)) if exact_pattern else 0.0

        return features

    @staticmethod
    def _entropy(values: list[int]) -> float:
        if not values:
            return 0.0
        counts = Counter(values)
        total = float(len(values))
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * log2(probability)
        return entropy

    @staticmethod
    def _repetition_rate(values: list[int]) -> float:
        if len(values) < 2:
            return 0.0
        repeats = sum(1 for left, right in zip(values, values[1:]) if left == right)
        return repeats / (len(values) - 1)

    @staticmethod
    def _distance_since(values: list[int], target: int) -> int:
        for offset, value in enumerate(reversed(values), start=1):
            if value == target:
                return offset
        return -1

    @staticmethod
    def _streak_length(values: list[int]) -> int:
        if not values:
            return 0
        last_value = values[-1]
        streak = 1
        for value in reversed(values[:-1]):
            if value != last_value:
                break
            streak += 1
        return streak

    @staticmethod
    def _rolling_rate(values: list[int], window: int) -> float:
        if not values:
            return 0.0
        tail = values[-window:]
        return sum(1 for value in tail if value == 4) / len(tail)

    @staticmethod
    def _rolling_mean(values: list[int], window: int) -> float:
        tail = values[-window:]
        return float(mean(tail)) if tail else 0.0

    @staticmethod
    def _rolling_std(values: list[int], window: int) -> float:
        tail = values[-window:]
        return float(pstdev(tail)) if len(tail) > 1 else 0.0

    @staticmethod
    def _transition_rate(values: list[int], target: int) -> float:
        if len(values) < 2:
            return 0.0
        transitions = sum(1 for left, right in zip(values, values[1:]) if right == target)
        return transitions / (len(values) - 1)

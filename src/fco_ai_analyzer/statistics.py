from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Iterable

try:
    from scipy.stats import chi2_contingency, fisher_exact
except Exception:  # pragma: no cover - optional dependency guard
    chi2_contingency = None
    fisher_exact = None

from .models import MainAttempt


@dataclass(slots=True)
class SummaryStatistics:
    total_attempts: int
    wins: int
    losses: int
    win_rate: float


@dataclass(slots=True)
class SignificanceResult:
    chi_square_p_value: float | None
    fisher_p_value: float | None
    odds_ratio: float | None


class StatisticsEngine:
    def summary(self, attempts: Iterable[MainAttempt]) -> SummaryStatistics:
        attempt_list = list(attempts)
        total_attempts = len(attempt_list)
        wins = sum(1 for attempt in attempt_list if attempt.is_win)
        losses = total_attempts - wins
        win_rate = wins / total_attempts if total_attempts else 0.0
        return SummaryStatistics(total_attempts=total_attempts, wins=wins, losses=losses, win_rate=win_rate)

    def significance_for_pattern(
        self,
        pattern_wins: int,
        pattern_losses: int,
        overall_wins: int,
        overall_losses: int,
    ) -> SignificanceResult:
        chi_square_p_value = None
        fisher_p_value = None
        odds_ratio = None

        if chi2_contingency is not None and pattern_wins + pattern_losses > 0:
            table = [
                [pattern_wins, pattern_losses],
                [overall_wins - pattern_wins, overall_losses - pattern_losses],
            ]
            _, chi_square_p_value, _, _ = chi2_contingency(table)

        if fisher_exact is not None and pattern_wins + pattern_losses > 0:
            table = [
                [pattern_wins, pattern_losses],
                [overall_wins - pattern_wins, overall_losses - pattern_losses],
            ]
            odds_ratio, fisher_p_value = fisher_exact(table)

        return SignificanceResult(
            chi_square_p_value=chi_square_p_value,
            fisher_p_value=fisher_p_value,
            odds_ratio=odds_ratio,
        )

    def bootstrap_confidence_interval(
        self,
        attempts: Iterable[MainAttempt],
        iterations: int = 2000,
        seed: int = 7,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        attempt_list = list(attempts)
        if not attempt_list:
            return 0.0, 0.0

        rng = Random(seed)
        labels = [1 if attempt.is_win else 0 for attempt in attempt_list]
        sample_size = len(labels)
        resampled_rates = []
        for _ in range(iterations):
            sample = [labels[rng.randrange(sample_size)] for _ in range(sample_size)]
            resampled_rates.append(mean(sample))

        resampled_rates.sort()
        lower_index = int((alpha / 2) * len(resampled_rates))
        upper_index = int((1 - alpha / 2) * len(resampled_rates)) - 1
        return resampled_rates[max(lower_index, 0)], resampled_rates[min(upper_index, len(resampled_rates) - 1)]

    def monte_carlo_win_rate(self, attempts: Iterable[MainAttempt], simulations: int = 10000) -> float:
        attempt_list = list(attempts)
        if not attempt_list:
            return 0.0
        win_rate = mean([1 if attempt.is_win else 0 for attempt in attempt_list])
        return win_rate

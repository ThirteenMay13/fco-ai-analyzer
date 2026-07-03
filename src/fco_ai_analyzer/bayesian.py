from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(slots=True)
class BetaPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, wins: int, losses: int) -> "BetaPosterior":
        return BetaPosterior(alpha=self.alpha + wins, beta=self.beta + losses)

    def credible_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        try:
            from scipy.stats import beta as beta_distribution
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("scipy is required for Bayesian credible intervals") from exc

        tail = (1.0 - confidence) / 2.0
        return (
            float(beta_distribution.ppf(tail, self.alpha, self.beta)),
            float(beta_distribution.ppf(1.0 - tail, self.alpha, self.beta)),
        )


def posterior_from_attempts(wins: int, losses: int, alpha: float = 1.0, beta: float = 1.0) -> BetaPosterior:
    return BetaPosterior(alpha=alpha + wins, beta=beta + losses)

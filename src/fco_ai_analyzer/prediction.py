from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bayesian import posterior_from_attempts
from .features import FeatureEngineer
from .markov import MarkovChainAnalyzer
from .ml import MachineLearningEngine, ModelBundle
from .models import MainAttempt, ParsedLog
from .patterns import PatternMiner, PatternStat
from .statistics import StatisticsEngine


@dataclass(slots=True)
class RecommendationResult:
    current_pattern: tuple[int, ...]
    success_probability: float
    confidence: str
    recommendation: str
    sample_size: int
    win_rate: float
    bayesian_probability: float
    markov_probability: float
    ml_probability: float | None
    rationale: list[str] = field(default_factory=list)


class RecommendationEngine:
    """Blend pattern mining, Bayesian inference, Markov context, and ML into one decision."""

    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        pattern_miner: PatternMiner | None = None,
        statistics_engine: StatisticsEngine | None = None,
        markov_analyzer: MarkovChainAnalyzer | None = None,
        ml_engine: MachineLearningEngine | None = None,
        min_samples: int = 10,
    ) -> None:
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.pattern_miner = pattern_miner or PatternMiner()
        self.statistics_engine = statistics_engine or StatisticsEngine()
        self.markov_analyzer = markov_analyzer or MarkovChainAnalyzer()
        self.ml_engine = ml_engine or MachineLearningEngine()
        self.min_samples = min_samples

    def recommend(
        self,
        parsed_log: ParsedLog,
        current_pattern: tuple[int, ...] | None = None,
        model_bundle: ModelBundle | None = None,
    ) -> RecommendationResult:
        if not parsed_log.attempts:
            return RecommendationResult(
                current_pattern=current_pattern or tuple(),
                success_probability=0.0,
                confidence="Low",
                recommendation="WAIT",
                sample_size=0,
                win_rate=0.0,
                bayesian_probability=0.0,
                markov_probability=0.0,
                ml_probability=None,
                rationale=["No attempts available yet."],
            )

        pattern = current_pattern or parsed_log.attempts[-1].bait_sequence
        pattern_stats = self._pattern_stats(parsed_log)
        selected_stat = pattern_stats.get(pattern)
        overall = self.statistics_engine.summary(parsed_log.attempts)
        sample_size = selected_stat.sample_size if selected_stat else 0
        win_rate = selected_stat.win_rate if selected_stat else overall.win_rate
        pattern_wins = selected_stat.wins if selected_stat else 0
        pattern_losses = selected_stat.losses if selected_stat else 0

        posterior = posterior_from_attempts(pattern_wins, pattern_losses)
        bayesian_probability = posterior.mean
        markov_prediction = self.markov_analyzer.predict(parsed_log, pattern)
        markov_probability = markov_prediction.probability_win

        prior_baits, prior_outcomes, session_attempts, attempt_index = self._context_for_pattern(parsed_log)
        features = self.feature_engineer.build_context_features(
            bait_sequence=pattern,
            prior_baits=prior_baits,
            prior_outcomes=prior_outcomes,
            session_attempts=session_attempts,
            attempt_index=attempt_index,
        )
        ml_probability = self.ml_engine.predict_probability(model_bundle, features) if model_bundle else None

        weighted_components = [
            (win_rate, 0.40 if sample_size >= self.min_samples else 0.20),
            (bayesian_probability, 0.30),
            (markov_probability, 0.20),
        ]
        if ml_probability is not None:
            weighted_components.append((ml_probability, 0.10))
        total_weight = sum(weight for _, weight in weighted_components)
        success_probability = sum(probability * weight for probability, weight in weighted_components) / total_weight

        confidence = self._confidence_label(sample_size, success_probability)
        if sample_size < self.min_samples:
            recommendation = "WAIT"
        else:
            recommendation = "ĐẬP" if success_probability >= 0.60 else "WAIT"

        rationale = [
            f"Pattern sample size: {sample_size}",
            f"Observed win rate: {win_rate:.1%}",
            f"Bayesian posterior mean: {bayesian_probability:.1%}",
            f"Markov win probability: {markov_probability:.1%}",
            f"Overall session win rate: {overall.win_rate:.1%}",
        ]
        if selected_stat is None:
            rationale.append("Pattern has not been seen often enough; falling back to broader context.")
        elif sample_size < self.min_samples:
            rationale.append(f"Insufficient sample size for a strong conclusion (min {self.min_samples}).")
        else:
            rationale.append(f"Pattern lift: {selected_stat.lift:.2f}x")

        if ml_probability is not None:
            rationale.append(f"ML ensemble probability: {ml_probability:.1%}")

        return RecommendationResult(
            current_pattern=pattern,
            success_probability=success_probability,
            confidence=confidence,
            recommendation=recommendation,
            sample_size=sample_size,
            win_rate=win_rate,
            bayesian_probability=bayesian_probability,
            markov_probability=markov_probability,
            ml_probability=ml_probability,
            rationale=rationale,
        )

    def _pattern_stats(self, parsed_log: ParsedLog) -> dict[tuple[int, ...], PatternStat]:
        return {stat.pattern: stat for stat in self.pattern_miner.mine(parsed_log, min_support=1)}

    @staticmethod
    def _context_for_pattern(parsed_log: ParsedLog) -> tuple[list[int], list[int], int, int]:
        if not parsed_log.sessions:
            return [], [], 0, 0
        session = parsed_log.sessions[-1]
        prior_baits: list[int] = []
        prior_outcomes: list[int] = []
        for attempt in session.attempts[:-1]:
            prior_baits.extend(attempt.bait_sequence)
            prior_outcomes.append(attempt.outcome)
        return prior_baits, prior_outcomes, len(session.attempts), len(session.attempts) - 1

    @staticmethod
    def _confidence_label(sample_size: int, success_probability: float) -> str:
        if sample_size >= 30 and 0.55 <= success_probability <= 0.80:
            return "High"
        if sample_size >= 15:
            return "Medium"
        return "Low"


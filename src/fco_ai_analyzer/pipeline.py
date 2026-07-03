from __future__ import annotations

import hashlib
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

from .features import FeatureEngineer
from .ml import MachineLearningEngine, ModelBundle
from .models import LogSession, ParsedLog
from .parser import LogParser
from .patterns import PatternMiner
from .prediction import RecommendationEngine, RecommendationResult
from .statistics import StatisticsEngine
from .storage import FCOStorage


@dataclass(slots=True)
class AnalysisArtifacts:
    parsed_log_path: Path | None
    database_path: Path


class FCOAnalyzer:
    """Orchestrate parsing, persistence, mining, and recommendation generation."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.parser = LogParser()
        self.storage = FCOStorage(database_path)
        self.feature_engineer = FeatureEngineer()
        self.pattern_miner = PatternMiner()
        self.statistics_engine = StatisticsEngine()
        self.recommendation_engine = RecommendationEngine(
            feature_engineer=self.feature_engineer,
            pattern_miner=self.pattern_miner,
            statistics_engine=self.statistics_engine,
        )
        self.ml_engine = MachineLearningEngine()
        self.model_bundle: ModelBundle | None = None

    def ingest_text(self, raw_text: str, source_name: str | None = None) -> bool:
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        is_new_ingestion = self.storage.register_ingestion(content_hash=content_hash, source_name=source_name)
        if not is_new_ingestion:
            return False

        parsed_log = self.parser.parse(raw_text)
        if not parsed_log.sessions:
            return False
        base_index = self.storage.max_session_index() + 1
        shifted = self._shift_session_indices(parsed_log, base_index)
        self.storage.upsert_parsed_log(shifted)
        return True

    def rebuild_from_history(self, history_text: str, source_name: str = "history_rebuild") -> None:
        parsed_log = self.parser.parse(history_text)
        self.storage.reset_all()
        if not parsed_log.sessions:
            return
        self.storage.upsert_parsed_log(parsed_log)
        history_hash = hashlib.sha256(history_text.encode("utf-8")).hexdigest()
        self.storage.register_ingestion(content_hash=history_hash, source_name=source_name)

    def analyze_text(self, raw_text: str, current_pattern: tuple[int, ...] | None = None) -> RecommendationResult:
        parsed_log = self.parser.parse(raw_text)
        return self.recommendation_engine.recommend(parsed_log, current_pattern=current_pattern, model_bundle=self.model_bundle)

    def train_models_from_text(self, raw_text: str) -> ModelBundle:
        parsed_log = self.parser.parse(raw_text)
        feature_rows = self.feature_engineer.transform(parsed_log)
        self.model_bundle = self.ml_engine.train(feature_rows)
        return self.model_bundle

    def build_report(self, raw_text: str) -> dict[str, object]:
        parsed_log = self.parser.parse(raw_text)
        summary = self.statistics_engine.summary(parsed_log.attempts)
        patterns = self.pattern_miner.mine(parsed_log, min_support=2)
        recommendation = self.recommendation_engine.recommend(parsed_log, model_bundle=self.model_bundle)
        return {
            "summary": summary,
            "patterns": patterns,
            "recommendation": recommendation,
            "attempt_count": len(parsed_log.attempts),
            "session_count": len(parsed_log.sessions),
        }

    @staticmethod
    def _shift_session_indices(parsed_log: ParsedLog, base_index: int) -> ParsedLog:
        shifted_sessions: list[LogSession] = []
        for session in parsed_log.sessions:
            new_session_index = base_index + session.session_index
            shifted_attempts = [replace(attempt, session_index=new_session_index) for attempt in session.attempts]
            shifted_baits = [replace(event, session_index=new_session_index) for event in session.bait_events]
            shifted_sessions.append(
                LogSession(
                    session_index=new_session_index,
                    raw_lines=list(session.raw_lines),
                    bait_events=shifted_baits,
                    attempts=shifted_attempts,
                )
            )
        return ParsedLog(sessions=shifted_sessions)


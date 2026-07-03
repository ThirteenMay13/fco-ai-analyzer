from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .features import FeatureRow

try:  # pragma: no cover - optional dependency guard
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
except Exception:  # pragma: no cover - optional dependency guard
    np = None
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    DictVectorizer = None
    LogisticRegression = None
    DecisionTreeClassifier = None


@dataclass(slots=True)
class ModelBundle:
    vectorizer: Any
    models: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)


class MachineLearningEngine:
    """Train and serve several supervised models on extracted attempt features."""

    def train(self, feature_rows: list[FeatureRow]) -> ModelBundle:
        if not feature_rows:
            raise ValueError("feature_rows must not be empty")
        if DictVectorizer is None:
            raise RuntimeError("scikit-learn is required to train ML models")

        feature_dicts = [row.features for row in feature_rows]
        labels = [row.label for row in feature_rows]
        vectorizer = DictVectorizer(sparse=False)
        matrix = vectorizer.fit_transform(feature_dicts)

        models: dict[str, Any] = {}
        models["logistic_regression"] = LogisticRegression(max_iter=1000, class_weight="balanced")
        models["decision_tree"] = DecisionTreeClassifier(max_depth=6, random_state=42)
        models["random_forest"] = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        models["gradient_boosting"] = GradientBoostingClassifier(random_state=42)

        for model in models.values():
            model.fit(matrix, labels)

        return ModelBundle(
            vectorizer=vectorizer,
            models=models,
            feature_names=list(vectorizer.get_feature_names_out()),
        )

    def predict_probability(self, bundle: ModelBundle, features: dict[str, float]) -> float:
        if not bundle.models:
            return 0.0
        matrix = bundle.vectorizer.transform([features])
        probabilities: list[float] = []
        for model in bundle.models.values():
            if hasattr(model, "predict_proba"):
                probabilities.append(float(model.predict_proba(matrix)[0][1]))
            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(matrix)[0])
                probabilities.append(1.0 / (1.0 + pow(2.718281828459045, -score)))
        return sum(probabilities) / len(probabilities) if probabilities else 0.0


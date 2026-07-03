from fco_ai_analyzer.parser import LogParser
from fco_ai_analyzer.prediction import RecommendationEngine


def test_recommendation_defaults_to_wait_on_small_samples() -> None:
    raw_text = """
    2 1 = 4
    2 1 = 2
    =======
    = 4
    """

    parsed = LogParser().parse(raw_text)
    result = RecommendationEngine(min_samples=10).recommend(parsed, current_pattern=(2, 1))

    assert result.current_pattern == (2, 1)
    assert result.sample_size == 2
    assert result.recommendation == "WAIT"
    assert result.confidence == "Low"
    assert 0.0 <= result.success_probability <= 1.0

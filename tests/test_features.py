from fco_ai_analyzer.features import FeatureEngineer
from fco_ai_analyzer.parser import LogParser


def test_feature_engineer_emits_more_than_100_features() -> None:
    raw_text = """
    2 1 = 4
    1 2 = 2
    4 4 = 4
    =======
    = 4
    """

    parsed = LogParser().parse(raw_text)
    rows = FeatureEngineer().transform(parsed)

    assert len(rows) == 4
    assert len(rows[0].features) >= 100
    assert rows[0].label == 1
    assert rows[1].label == 0

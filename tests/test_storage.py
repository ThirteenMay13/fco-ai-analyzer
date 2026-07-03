from pathlib import Path

from fco_ai_analyzer.parser import LogParser
from fco_ai_analyzer.storage import FCOStorage


def test_storage_persists_parsed_log(tmp_path: Path) -> None:
    raw_text = """
    2 1 = 4
    =======
    = 2
    """

    database_path = tmp_path / "fco.sqlite3"
    parsed = LogParser().parse(raw_text)
    storage = FCOStorage(database_path)
    storage.upsert_parsed_log(parsed)

    with storage.connect() as connection:
        session_count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        attempt_count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        bait_count = connection.execute("SELECT COUNT(*) FROM bait_events").fetchone()[0]

    assert session_count == 2
    assert attempt_count == 2
    assert bait_count == 2

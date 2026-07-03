from pathlib import Path

from fco_ai_analyzer.history import HistoryManager
from fco_ai_analyzer.pipeline import FCOAnalyzer


def test_history_manager_appends_sessions_with_separator(tmp_path: Path) -> None:
    seed = tmp_path / "seed.txt"
    seed.write_text("= 4\n", encoding="utf-8")
    history = tmp_path / "history.txt"
    manager = HistoryManager(history_path=history, seed_path=seed)

    manager.ensure_initialized()
    manager.append_session("2 1 = 4")

    text = manager.read_all()
    assert "= 4" in text
    assert "2 1 = 4" in text
    assert "========================" in text


def test_analyzer_deduplicates_same_ingestion(tmp_path: Path) -> None:
    db = tmp_path / "fco.sqlite3"
    analyzer = FCOAnalyzer(db)
    raw_text = "2 1 = 4\n"

    first = analyzer.ingest_text(raw_text, source_name="test")
    second = analyzer.ingest_text(raw_text, source_name="test")

    assert first is True
    assert second is False


def test_history_manager_pop_last_session(tmp_path: Path) -> None:
    history = tmp_path / "history.txt"
    manager = HistoryManager(history_path=history, seed_path=None)
    manager.append_session("= 4")
    manager.append_session("2 1 = 4")

    removed = manager.pop_last_session()

    assert removed == "2 1 = 4"
    assert manager.session_count() == 1
    assert "2 1 = 4" not in manager.read_all()


def test_rebuild_from_history_replaces_db_state(tmp_path: Path) -> None:
    db = tmp_path / "fco.sqlite3"
    analyzer = FCOAnalyzer(db)
    analyzer.ingest_text("= 4\n", source_name="first")
    analyzer.ingest_text("2 1 = 2\n", source_name="second")

    analyzer.rebuild_from_history("= 4\n", source_name="rebuild")

    with analyzer.storage.connect() as connection:
        attempts = connection.execute("SELECT COUNT(*) AS cnt FROM attempts").fetchone()[0]
        sessions = connection.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()[0]
    assert attempts == 1
    assert sessions == 1

from __future__ import annotations

from pathlib import Path

SESSION_SEPARATOR = "========================"


class HistoryManager:
    """Persist and append FC Online logs so analytics improve over time."""

    def __init__(self, history_path: Path, seed_path: Path | None = None) -> None:
        self.history_path = history_path
        self.seed_path = seed_path

    def ensure_initialized(self) -> None:
        if self.history_path.exists():
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if self.seed_path and self.seed_path.exists():
            seed_text = self.seed_path.read_text(encoding="utf-8")
            self.history_path.write_text(seed_text.strip() + "\n", encoding="utf-8")
            return
        self.history_path.write_text("", encoding="utf-8")

    def read_all(self) -> str:
        self.ensure_initialized()
        return self.history_path.read_text(encoding="utf-8")

    def append_session(self, session_text: str) -> None:
        cleaned = session_text.strip()
        if not cleaned:
            return
        self.ensure_initialized()
        with self.history_path.open("a", encoding="utf-8") as handle:
            if self.history_path.stat().st_size > 0:
                handle.write(f"\n\n{SESSION_SEPARATOR}\n\n")
            handle.write(cleaned)
            handle.write("\n")

    def sessions(self) -> list[str]:
        raw = self.read_all()
        chunks = [chunk.strip() for chunk in raw.split(SESSION_SEPARATOR)]
        return [chunk for chunk in chunks if chunk]

    def session_count(self) -> int:
        return len(self.sessions())

    def pop_last_session(self) -> str | None:
        items = self.sessions()
        if not items:
            return None
        removed = items.pop()
        rebuilt = f"\n\n{SESSION_SEPARATOR}\n\n".join(items)
        rebuilt = rebuilt.strip()
        if rebuilt:
            rebuilt += "\n"
        self.history_path.write_text(rebuilt, encoding="utf-8")
        return removed

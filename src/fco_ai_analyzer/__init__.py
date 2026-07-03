"""FCO AI Analyzer package."""

from .models import BaitEvent, LogSession, MainAttempt, ParsedLog
from .parser import LogParser

__all__ = [
    "BaitEvent",
    "LogParser",
    "LogSession",
    "MainAttempt",
    "ParsedLog",
]

__version__ = "0.1.0"

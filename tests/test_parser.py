from fco_ai_analyzer.parser import LogParser


def test_parser_splits_sessions_and_multiple_attempts_per_line() -> None:
    raw_text = """
    2 1 = 4 (Laudrup ITM)
    2 2 1 = 2 (Rudiger ts) 1 = 4 Rudiger ts)
    =======
    = 4 (cr7 ptg)
    = 2 (cr7 ptg) = 4 (cr7 ptg)
    """

    parsed = LogParser().parse(raw_text)

    assert len(parsed.sessions) == 2
    assert len(parsed.attempts) == 6
    assert parsed.attempts[0].bait_sequence == (2, 1)
    assert parsed.attempts[0].outcome == 4
    assert parsed.attempts[1].bait_sequence == (2, 2, 1)
    assert parsed.attempts[1].outcome == 2
    assert parsed.attempts[2].bait_sequence == (1,)
    assert parsed.attempts[2].outcome == 4
    assert parsed.attempts[3].bait_sequence == ()
    assert parsed.attempts[3].outcome == 4
    assert parsed.attempts[4].bait_sequence == ()
    assert parsed.attempts[4].outcome == 2
    assert parsed.attempts[5].bait_sequence == ()
    assert parsed.attempts[5].outcome == 4

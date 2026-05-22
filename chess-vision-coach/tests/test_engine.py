"""Stockfish engine tests on three known positions."""

import re

import pytest

from app import engine

pytestmark = pytest.mark.skipif(
    engine.find_stockfish() is None,
    reason="Stockfish binary not installed in this environment",
)

# After 1.f3 e5 2.g4 -- Black to move and play Qh4#.
MATE_IN_1 = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
# A normal, roughly equal middlegame (Ruy Lopez, Black to move).
EQUAL_MIDDLEGAME = (
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
)
# King vs King -- a dead draw by insufficient material.
DRAWN_KVK = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"

CP_PATTERN = re.compile(r"^[+-]\d+\.\d{2}$")


def test_mate_in_one_is_found():
    result = engine.analyse(MATE_IN_1)
    assert result.game_over is False
    assert result.best_move_uci == "d8h4"
    assert result.best_move_san == "Qh4#"
    # Sign is relative to the side to move: Black mates, so positive.
    assert result.eval_text == "Mate in 1"


def test_equal_middlegame_returns_sane_structure():
    result = engine.analyse(EQUAL_MIDDLEGAME, multipv=3)
    assert result.game_over is False
    assert result.best_move_uci is not None
    assert 4 <= len(result.best_move_uci) <= 5
    assert 1 <= len(result.candidates) <= 3
    for cand in result.candidates:
        assert 4 <= len(cand.uci) <= 5
        assert cand.san
        assert CP_PATTERN.match(cand.eval_text) or cand.eval_text.startswith("Mate")
    # A balanced opening position should not be a forced mate or lopsided.
    assert CP_PATTERN.match(result.eval_text)
    assert abs(float(result.eval_text)) < 3.0


def test_drawn_king_vs_king_endgame():
    result = engine.analyse(DRAWN_KVK)
    assert result.game_over is True
    assert result.best_move_uci is None
    assert result.eval_text == "Draw"


def test_missing_engine_raises_clear_error(monkeypatch):
    monkeypatch.setattr(engine, "find_stockfish", lambda: None)
    with pytest.raises(engine.EngineError) as exc:
        engine.analyse(EQUAL_MIDDLEGAME)
    assert "install" in str(exc.value).lower()


def test_multipv_lines_are_distinct():
    result = engine.analyse(EQUAL_MIDDLEGAME, multipv=3)
    ucis = [c.uci for c in result.candidates]
    assert len(ucis) == len(set(ucis))


def test_candidates_carry_a_principal_variation():
    result = engine.analyse(EQUAL_MIDDLEGAME, multipv=3)
    for cand in result.candidates:
        assert cand.line, "each candidate should have a SAN principal variation"
    # The PV line should begin with the candidate's own move.
    top = result.candidates[0]
    assert top.san.rstrip("+#") in top.line

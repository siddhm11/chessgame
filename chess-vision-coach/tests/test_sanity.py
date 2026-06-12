"""Tests for the chess-logic sanity layer (chess_vision/sanity.py).

Pure-logic tests: no model, no images.
"""

from chess_vision.sanity import _FLAG_CONF, sanitize


def _empty_grid():
    return [["." for _ in range(8)] for _ in range(8)]


def _full_conf(v=0.9):
    return [[v for _ in range(8)] for _ in range(8)]


def _no_alts():
    return [[None for _ in range(8)] for _ in range(8)]


def _legal_position():
    grid = _empty_grid()
    grid[0][4] = "k"  # e8
    grid[7][4] = "K"  # e1
    grid[1][0] = "p"  # a7
    grid[6][0] = "P"  # a2
    return grid


def test_legal_position_untouched():
    grid = _legal_position()
    out_grid, out_conf, notes = sanitize(grid, _full_conf(), _no_alts())
    assert out_grid == grid
    assert notes == []
    assert all(c == 0.9 for row in out_conf for c in row)


def test_inputs_not_mutated():
    grid = _legal_position()
    grid[0][0] = "P"  # back-rank pawn violation
    conf = _full_conf()
    sanitize(grid, conf, _no_alts())
    assert grid[0][0] == "P"
    assert conf[0][0] == 0.9


def test_backrank_pawn_swapped_to_alt():
    grid = _legal_position()
    grid[0][0] = "P"  # white pawn on a8: impossible
    alts = _no_alts()
    alts[0][0] = [("Q", 0.40), ("R", 0.20)]
    out_grid, out_conf, notes = sanitize(grid, _full_conf(), alts)
    assert out_grid[0][0] == "Q"
    assert out_conf[0][0] == 0.40
    assert any("a8" in n for n in notes)


def test_backrank_pawn_flagged_without_alt():
    grid = _legal_position()
    grid[7][3] = "p"  # black pawn on d1: impossible
    out_grid, out_conf, notes = sanitize(grid, _full_conf(), _no_alts())
    assert out_grid[7][3] == "p"  # symbol kept, but…
    assert out_conf[7][3] == _FLAG_CONF  # …loudly flagged
    assert any("d1" in n for n in notes)


def test_backrank_pawn_alt_must_match_colour():
    grid = _legal_position()
    grid[0][0] = "P"
    alts = _no_alts()
    alts[0][0] = [("q", 0.50)]  # black queen: wrong colour, must be skipped
    out_grid, out_conf, _ = sanitize(grid, _full_conf(), alts)
    assert out_grid[0][0] == "P"
    assert out_conf[0][0] == _FLAG_CONF


def test_duplicate_kings_keep_most_confident():
    grid = _legal_position()
    grid[4][4] = "K"  # second white king on e4
    conf = _full_conf()
    conf[7][4] = 0.95  # real king e1
    conf[4][4] = 0.60  # impostor e4
    alts = _no_alts()
    alts[4][4] = [("Q", 0.35)]
    out_grid, out_conf, notes = sanitize(grid, conf, alts)
    assert out_grid[7][4] == "K"  # winner untouched
    assert out_conf[7][4] == 0.95
    assert out_grid[4][4] == "Q"  # impostor reread
    assert out_conf[4][4] == 0.35
    assert any("e4" in n for n in notes)


def test_duplicate_kings_flagged_without_alt():
    grid = _legal_position()
    grid[4][4] = "k"
    conf = _full_conf()
    conf[0][4] = 0.95
    conf[4][4] = 0.50
    out_grid, out_conf, _ = sanitize(grid, conf, _no_alts())
    assert out_grid[4][4] == "k"
    assert out_conf[4][4] == _FLAG_CONF


def test_missing_king_recovered_from_alts():
    """The classic K<->Q confusion: white king misread as a second queen."""
    grid = _legal_position()
    grid[7][4] = "Q"  # king square misread as queen -> no white king at all
    alts = _no_alts()
    alts[7][4] = [("K", 0.30)]
    out_grid, out_conf, notes = sanitize(grid, _full_conf(), alts)
    assert out_grid[7][4] == "K"
    assert out_conf[7][4] == 0.30
    assert any("e1" in n for n in notes)


def test_missing_king_no_candidate_does_nothing():
    grid = _legal_position()
    grid[7][4] = "Q"  # no white king, and no alts suggest one
    out_grid, _, notes = sanitize(grid, _full_conf(), _no_alts())
    assert "K" not in [sq for row in out_grid for sq in row]
    assert notes == []


def test_nine_pawns_flags_least_confident():
    grid = _legal_position()
    conf = _full_conf()
    # 8 more white pawns on rank 3 (a2 already has one) = 9 total.
    for c in range(8):
        grid[5][c] = "P"
    conf[6][0] = 0.30  # the a2 pawn is the least confident -> the suspect
    out_grid, out_conf, notes = sanitize(grid, conf, _no_alts())
    assert out_grid[6][0] == "P"  # symbol kept
    assert out_conf[6][0] == _FLAG_CONF  # but flagged
    # The 8 confident pawns are untouched.
    assert all(out_conf[5][c] == 0.9 for c in range(8))
    assert len(notes) == 1


def test_sanitize_handles_none_alts():
    grid = _legal_position()
    grid[0][0] = "P"
    out_grid, out_conf, _ = sanitize(grid, _full_conf())  # alts omitted
    assert out_conf[0][0] == _FLAG_CONF
    assert out_grid[0][0] == "P"


def test_missing_king_with_duplicate_queens_picks_e_file():
    """When the model sharply confuses K with Q, no cell may rank K in the
    top-3 alts. With two queens on the back rank, the e-file one is the
    real king -- fall back to that heuristic so the repair still fires."""
    from chess_vision.sanity import sanitize

    grid = [["." for _ in range(8)] for _ in range(8)]
    grid[7][3] = "Q"  # d1: real queen
    grid[7][4] = "Q"  # e1: misread king
    grid[7][0] = "R"  # supply at least one other white piece
    grid[0][4] = "k"
    conf = [[0.95 for _ in range(8)] for _ in range(8)]
    alts = [[None for _ in range(8)] for _ in range(8)]

    out_grid, out_conf, notes = sanitize(grid, conf, alts)
    assert out_grid[7][4] == "K", "e1 should be relabelled as K"
    assert out_grid[7][3] == "Q", "d1 (the real queen) must stay"
    assert out_conf[7][4] < 0.55, "repaired cell must tint red"
    assert any("e1" in n for n in notes)


def test_missing_king_with_single_queen_does_not_fire_efile_fallback():
    """One queen + no king is genuinely ambiguous; do not force the swap
    just because a queen exists. (Avoid clobbering legitimate positions
    where the user really is missing a king somehow.)"""
    from chess_vision.sanity import sanitize

    grid = [["." for _ in range(8)] for _ in range(8)]
    grid[7][3] = "Q"  # only one queen
    grid[7][0] = "R"
    grid[0][4] = "k"
    conf = [[0.95 for _ in range(8)] for _ in range(8)]
    out_grid, _, _ = sanitize(grid, conf)
    assert out_grid[7][3] == "Q", "single queen must not be coerced to king"

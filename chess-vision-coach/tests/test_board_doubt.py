"""The board-doubt gate: a misaligned crop must never look trustworthy.

Before this gate, a crop sitting half a square off produced a board where
most pieces were read one square away -- with high confidence and no tint.
"""

from __future__ import annotations

import chess_vision.model_backend as mb
from chess_vision.vision import BackendOutput


def _out(conf=0.95):
    grid = [["." for _ in range(8)] for _ in range(8)]
    grid[0][4], grid[7][4] = "k", "K"
    return BackendOutput(grid=grid, conf=[[conf] * 8 for _ in range(8)],
                         board_found=True, notes=["e8: something earlier"])


def test_good_fit_passes_through_untouched():
    out = _out()
    assert mb.ModelBackend._apply_board_doubt(out, 0.95) is out


def test_low_fit_tints_every_square_and_explains():
    res = mb.ModelBackend._apply_board_doubt(_out(), 0.62)
    # Every square inside the app's amber band (0.55 <= c < 0.80)...
    assert all(c <= mb._BOARD_DOUBT_CONF < 0.80 for row in res.conf for c in row)
    # ...the grid itself is unchanged (we flag, we don't guess)...
    assert res.grid == _out().grid
    # ...and the user is told why, ahead of any earlier notes.
    assert res.notes[0].startswith("Board alignment uncertain (grid fit 0.62)")
    assert res.notes[1] == "e8: something earlier"


def test_gate_never_raises_confidence():
    res = mb.ModelBackend._apply_board_doubt(_out(conf=0.30), 0.5)
    assert all(c == 0.30 for row in res.conf for c in row)


def test_threshold_is_inclusive_at_boundary():
    out = _out()
    assert mb.ModelBackend._apply_board_doubt(out, mb._BOARD_DOUBT_FIT) is out

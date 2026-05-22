"""Stockfish wrapper built on python-chess's native UCI engine support.

Decision Point 2(a): we shell out to a Stockfish binary discovered on the
system (apt on Linux / brew on macOS). python-chess's chess.engine drives it
over UCI and supports multipv natively, so no extra pip wrapper is needed.
"""

from __future__ import annotations

import os
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field

import chess
import chess.engine

# Common locations for a Stockfish binary, in priority order.
_CANDIDATE_PATHS = [
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
]


class EngineError(Exception):
    """Raised when the engine is missing, times out, or fails."""


def install_hint() -> str:
    """Platform-specific install command for Stockfish."""
    system = platform.system()
    if system == "Darwin":
        return "brew install stockfish"
    if system == "Windows":
        return "Download Stockfish from https://stockfishchess.org/download/ "
    return "sudo apt-get install stockfish"


def find_stockfish() -> str | None:
    """Locate the Stockfish binary, or None if it cannot be found."""
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    for candidate in _CANDIDATE_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


@dataclass
class Candidate:
    """One candidate move from a multipv search."""

    uci: str
    san: str
    eval_text: str
    line: str = ""  # principal variation in SAN, e.g. "1. e4 e5 2. Nf3"


@dataclass
class BestMoveResult:
    best_move_uci: str | None
    best_move_san: str | None
    eval_text: str
    candidates: list[Candidate] = field(default_factory=list)
    game_over: bool = False


def _format_score(score: chess.engine.PovScore, turn: chess.Color) -> str:
    """Format a score relative to the side to move.

    Mate scores: 'Mate in N' with sign relative to the side to move --
    positive means the side to move delivers mate, negative means it gets
    mated. Centipawn scores: '+1.23' style, positive favouring side to move.
    """
    pov = score.pov(turn)
    if pov.is_mate():
        mate_in = pov.mate()
        return f"Mate in {mate_in}"
    cp = pov.score()
    if cp is None:
        return "0.00"
    return f"{cp / 100.0:+.2f}"


def _analyse_blocking(
    path: str,
    board: chess.Board,
    depth: int,
    time_limit: float,
    multipv: int,
):
    engine = chess.engine.SimpleEngine.popen_uci(path)
    try:
        return engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=time_limit),
            multipv=multipv,
        )
    finally:
        try:
            engine.quit()
        except Exception:  # noqa: BLE001
            engine.close()


def analyse(
    fen: str,
    *,
    depth: int = 18,
    time_limit: float = 1.0,
    multipv: int = 3,
    timeout: float = 5.0,
) -> BestMoveResult:
    """Analyse a position with Stockfish.

    `fen` must already be a valid, legal FEN (validate with chess.Board
    before calling). Raises EngineError if Stockfish is missing or the
    search exceeds `timeout` seconds.
    """
    path = find_stockfish()
    if path is None:
        raise EngineError(
            f"Stockfish was not found. Install it with: {install_hint()}"
        )

    board = chess.Board(fen)

    if board.is_game_over():
        outcome = board.outcome()
        if outcome and outcome.winner is not None:
            text = "Checkmate"
        else:
            text = "Draw"
        return BestMoveResult(
            best_move_uci=None,
            best_move_san=None,
            eval_text=text,
            candidates=[],
            game_over=True,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _analyse_blocking, path, board, depth, time_limit, multipv
        )
        try:
            info = future.result(timeout=timeout)
        except FuturesTimeout as exc:
            raise EngineError(
                f"Stockfish did not respond within {timeout:.0f}s."
            ) from exc
        except chess.engine.EngineError as exc:
            raise EngineError(f"Stockfish failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"Engine error: {exc}") from exc

    # python-chess returns a list when multipv is set, a single dict otherwise.
    infos = info if isinstance(info, list) else [info]

    candidates: list[Candidate] = []
    for entry in infos:
        pv = entry.get("pv")
        score = entry.get("score")
        if not pv or score is None:
            continue
        move = pv[0]
        try:
            line = board.variation_san(pv[:10])
        except ValueError:
            line = board.san(move)
        candidates.append(
            Candidate(
                uci=move.uci(),
                san=board.san(move),
                eval_text=_format_score(score, board.turn),
                line=line,
            )
        )

    if not candidates:
        raise EngineError("Stockfish returned no moves for this position.")

    best = candidates[0]
    return BestMoveResult(
        best_move_uci=best.uci,
        best_move_san=best.san,
        eval_text=best.eval_text,
        candidates=candidates,
    )

"""Chess-logic sanity layer: repair impossible vision outputs.

A piece detector has no concept of chess rules, so its raw grid can be
*impossible*: a pawn on the back rank, two white kings, nine black pawns.
Every such violation is a guaranteed misread, which means it carries
signal we can use:

  - If the cell has a ranked list of alternative classes from the model,
    swap in the best alternative that resolves the violation (and lower
    the cell's confidence to the alternative's score so it still tints
    amber/red for review).
  - If no usable alternative exists, slash the cell's confidence to
    _FLAG_CONF so the UI paints it red — the user cannot miss it.

Rules are 180°-orientation-invariant (ranks 1 and 8 are the two extreme
rows whichever side was closer to the camera), so this can run inside a
backend before the user's orientation choice is applied. 90° rotations
must already be corrected upstream (ModelBackend does this).

Rules applied, in order:
  1. Pawn on rank 1 or rank 8        -> swap to best non-pawn alt, or flag
  2. More than one king per colour   -> keep the most confident; swap or
                                        flag the rest
  3. No king for a colour that has   -> convert the piece whose alts rank
     pieces on the board                that king highest (K<->Q confusion
                                        is the classic misread), or do
                                        nothing if no cell suggests a king
  4. More than 8 pawns per colour    -> flag the least confident surplus
                                        pawns (no auto-swap: which pawn is
                                        wrong is genuinely ambiguous)
"""

from __future__ import annotations

FILES = "abcdefgh"

# Confidence assigned to a cell we know is wrong but cannot repair.
# Below the app's red-tint threshold (0.55) by a wide margin.
_FLAG_CONF = 0.20

# Confidence for empty squares when the whole read is implausible (amber).
_IMPLAUSIBLE_EMPTY_CONF = 0.70

# Minimum class score an alternative needs before we trust an auto-swap.
_MIN_ALT_SCORE = 0.03

# alts[r][c] is None or a ranked list of (symbol, score) for that cell,
# best first, excluding the symbol already chosen in grid[r][c].
Alts = list[list[list[tuple[str, float]] | None]]


def _square(r: int, c: int) -> str:
    return f"{FILES[c]}{8 - r}"


def _best_alt(alts: Alts, r: int, c: int, *, allowed) -> tuple[str, float] | None:
    ranked = alts[r][c] if alts else None
    if not ranked:
        return None
    for sym, score in ranked:
        if score >= _MIN_ALT_SCORE and allowed(sym):
            return sym, float(score)
    return None


def sanitize(
    grid: list[list[str]],
    conf: list[list[float]],
    alts: Alts | None = None,
) -> tuple[list[list[str]], list[list[float]], list[str]]:
    """Repair or flag chess-impossible cells. Mutates copies, not inputs.

    Returns (grid, conf, notes) where notes is a human-readable record of
    every change made (empty for an already-plausible position).
    """
    grid = [row[:] for row in grid]
    conf = [row[:] for row in conf]
    notes: list[str] = []
    if alts is None:
        alts = [[None] * 8 for _ in range(8)]

    # ── 1. pawns on the extreme ranks ────────────────────────────────────
    for r in (0, 7):
        for c in range(8):
            sym = grid[r][c]
            if sym not in ("P", "p"):
                continue
            same_colour = str.isupper if sym == "P" else str.islower
            alt = _best_alt(
                alts, r, c,
                allowed=lambda s: s not in ("P", "p") and same_colour(s),
            )
            if alt:
                grid[r][c] = alt[0]
                conf[r][c] = min(conf[r][c], alt[1])
                notes.append(
                    f"{_square(r, c)}: pawn cannot stand on this rank; "
                    f"reread as {alt[0]}"
                )
            else:
                conf[r][c] = _FLAG_CONF
                notes.append(
                    f"{_square(r, c)}: pawn cannot stand on this rank; "
                    "flagged for review"
                )

    # ── 2. duplicate kings ───────────────────────────────────────────────
    for king in ("K", "k"):
        cells = [
            (conf[r][c], r, c)
            for r in range(8) for c in range(8)
            if grid[r][c] == king
        ]
        if len(cells) <= 1:
            continue
        cells.sort(reverse=True)  # most confident first; that one stays
        for _, r, c in cells[1:]:
            same_colour = str.isupper if king == "K" else str.islower
            alt = _best_alt(
                alts, r, c,
                allowed=lambda s: s not in ("K", "k") and same_colour(s),
            )
            if alt:
                grid[r][c] = alt[0]
                conf[r][c] = min(conf[r][c], alt[1])
                notes.append(
                    f"{_square(r, c)}: second {king} impossible; "
                    f"reread as {alt[0]}"
                )
            else:
                conf[r][c] = _FLAG_CONF
                notes.append(
                    f"{_square(r, c)}: second {king} impossible; "
                    "flagged for review"
                )

    # ── 3. missing king for a colour that has pieces ─────────────────────
    for king in ("K", "k"):
        flat = [sq for row in grid for sq in row]
        if king in flat:
            continue
        same_colour = str.isupper if king == "K" else str.islower
        if not any(sq != "." and same_colour(sq) for sq in flat):
            continue  # colour absent entirely; nothing sensible to do
        # The cell whose alternatives rank this king highest is the most
        # likely misread (typically the K<->Q confusion).
        best: tuple[float, int, int] | None = None
        for r in range(8):
            for c in range(8):
                if grid[r][c] == "." or not same_colour(grid[r][c]):
                    continue
                ranked = alts[r][c]
                if not ranked:
                    continue
                for sym, score in ranked:
                    if sym == king and score >= _MIN_ALT_SCORE:
                        if best is None or score > best[0]:
                            best = (score, r, c)
                        break
        if best:
            score, r, c = best
            notes.append(
                f"{_square(r, c)}: no {king} found anywhere; "
                f"reread {grid[r][c]} as {king}"
            )
            grid[r][c] = king
            conf[r][c] = min(conf[r][c], score)
            continue

        # Fallback when no cell ranks the king as an alt: a sharply trained
        # model can call the king a queen with such high confidence that the
        # K class never enters the top-3 alts. In that case, if the colour
        # has ≥2 queens, the spurious one is almost always on the king's
        # starting file (e). Convert the queen closest to the e-file to a
        # king, capping its confidence so the cell tints amber.
        queen = "Q" if king == "K" else "q"
        queen_cells = [
            (r, c) for r in range(8) for c in range(8) if grid[r][c] == queen
        ]
        if len(queen_cells) >= 2:
            r, c = min(queen_cells, key=lambda rc: (abs(rc[1] - 4), conf[rc[0]][rc[1]]))
            notes.append(
                f"{_square(r, c)}: no {king} found; two {queen}s "
                f"suggest the e-file one is the {king}"
            )
            grid[r][c] = king
            conf[r][c] = min(conf[r][c], _FLAG_CONF)

    # ── 4. more than 8 pawns per colour ──────────────────────────────────
    for pawn in ("P", "p"):
        cells = [
            (conf[r][c], r, c)
            for r in range(8) for c in range(8)
            if grid[r][c] == pawn
        ]
        if len(cells) <= 8:
            continue
        cells.sort()  # least confident first; those are the suspects
        for _, r, c in cells[: len(cells) - 8]:
            conf[r][c] = min(conf[r][c], _FLAG_CONF)
            notes.append(
                f"{_square(r, c)}: {len(cells)} {pawn}-pawns is impossible; "
                "flagged the least confident"
            )

    # ── 5. material that no game can reach ───────────────────────────────
    # Every queen beyond the first, rook/bishop/knight beyond the second,
    # must be a promoted pawn: extras <= 8 - pawns on the board. A read like
    # "11 white queens, 0 pawns" means the model doesn't recognise this
    # piece set -- typically many squares are wrong, with HIGH confidence.
    # We can't tell which, so the whole board is tinted for checking.
    # (Never fires on any of the 500 real positions in samryan18.)
    for problem in material_problems("/".join("".join(r) for r in grid)):
        for r in range(8):
            for c in range(8):
                cap = _FLAG_CONF if grid[r][c] != "." else _IMPLAUSIBLE_EMPTY_CONF
                conf[r][c] = min(conf[r][c], cap)
        notes.append(f"{problem} -- several pieces are misread (this piece set may be "
                     "unfamiliar to the model). Please check every square.")

    return grid, conf, notes


def material_problems(placement: str) -> list[str]:
    """Material that no legal game can reach, per colour, as messages.

    `placement` is the board part of a FEN. Extra pieces (queens beyond 1,
    rooks/bishops/knights beyond 2) must come from promoted pawns, so they
    can't exceed 8 minus the pawns still on the board.
    """
    problems = []
    for colour, upper in (("White", True), ("Black", False)):
        n = {k: 0 for k in "pnbrqk"}
        for ch in placement:
            if ch.isalpha() and ch.isupper() == upper:
                n[ch.lower()] += 1
        if n["p"] > 8:
            continue   # rule 4 handles too many pawns, more precisely
        extra = (max(0, n["q"] - 1) + max(0, n["r"] - 2)
                 + max(0, n["b"] - 2) + max(0, n["n"] - 2))
        if extra > 8 - n["p"]:
            problems.append(
                f"{colour} has {n['q']} queens, {n['r']} rooks, {n['b']} bishops and "
                f"{n['n']} knights with {n['p']} pawns, which no game can reach")
    return problems

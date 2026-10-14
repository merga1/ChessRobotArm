"""
board_renderer.py
-----------------
Pure-Pillow chess board renderer.
Replaces cairosvg / cairocffi which require the native libcairo-2.dll that is
not bundled with Windows Python installations.

Usage:
    from board_renderer import board_to_png
    board_to_png(board, output='output.png', size=900)
    board_to_png(board, output='output.png', size=900, highlight=[sq1, sq2])
"""

import chess
from PIL import Image, ImageDraw, ImageFont

# ── Colours ──────────────────────────────────────────────────────────────────
LIGHT   = (240, 217, 181)
DARK    = (181, 136,  99)
HILITE  = (206, 210, 100)

# ── Piece glyphs (Unicode chess symbols) ─────────────────────────────────────
_GLYPH = {
    (chess.KING,   chess.WHITE): chr(0x2654),
    (chess.QUEEN,  chess.WHITE): chr(0x2655),
    (chess.ROOK,   chess.WHITE): chr(0x2656),
    (chess.BISHOP, chess.WHITE): chr(0x2657),
    (chess.KNIGHT, chess.WHITE): chr(0x2658),
    (chess.PAWN,   chess.WHITE): chr(0x2659),
    (chess.KING,   chess.BLACK): chr(0x265A),
    (chess.QUEEN,  chess.BLACK): chr(0x265B),
    (chess.ROOK,   chess.BLACK): chr(0x265C),
    (chess.BISHOP, chess.BLACK): chr(0x265D),
    (chess.KNIGHT, chess.BLACK): chr(0x265E),
    (chess.PAWN,   chess.BLACK): chr(0x265F),
}

# ── Fallback letter labels when a Unicode font is unavailable ─────────────────
_LETTER = {
    (chess.KING,   chess.WHITE): "K",
    (chess.QUEEN,  chess.WHITE): "Q",
    (chess.ROOK,   chess.WHITE): "R",
    (chess.BISHOP, chess.WHITE): "B",
    (chess.KNIGHT, chess.WHITE): "N",
    (chess.PAWN,   chess.WHITE): "P",
    (chess.KING,   chess.BLACK): "k",
    (chess.QUEEN,  chess.BLACK): "q",
    (chess.ROOK,   chess.BLACK): "r",
    (chess.BISHOP, chess.BLACK): "b",
    (chess.KNIGHT, chess.BLACK): "n",
    (chess.PAWN,   chess.BLACK): "p",
}

# ── Font cache ────────────────────────────────────────────────────────────────
_FONT_CACHE = {}

def _get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "seguisym.ttf",        # Segoe UI Symbol (Windows, covers chess glyphs)
        "NotoSans-Regular.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
    ]
    font = None
    for name in candidates:
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def board_to_png(board, output="output.png", size=900, highlight=None):
    """
    Render *board* as a PNG file at *output*.

    Parameters
    ----------
    board     : chess.Board
    output    : str   – destination PNG path
    size      : int   – total image side length in pixels
    highlight : list  – chess square indices to highlight (e.g. [chess.E2, chess.E4])
    """
    highlight_set = set(highlight) if highlight else set()

    margin = size // 20          # border for rank/file labels
    inner  = size - 2 * margin   # playable area
    sq_px  = inner // 8

    img  = Image.new("RGB", (size, size), (50, 50, 50))
    draw = ImageDraw.Draw(img)

    piece_font = _get_font(int(sq_px * 0.72))
    label_font = _get_font(max(10, sq_px // 5))

    for rank in range(8):
        for file in range(8):
            sq_idx = chess.square(file, rank)

            x0 = margin + file * sq_px
            y0 = margin + (7 - rank) * sq_px

            is_light = (file + rank) % 2 == 0
            base_col = LIGHT if is_light else DARK
            fill_col = HILITE if sq_idx in highlight_set else base_col

            draw.rectangle([x0, y0, x0 + sq_px - 1, y0 + sq_px - 1], fill=fill_col)

            # ── Piece ─────────────────────────────────────────────────
            piece = board.piece_at(sq_idx)
            if piece:
                key   = (piece.piece_type, piece.color)
                glyph = _GLYPH.get(key, _LETTER.get(key, "?"))

                shadow_col = (80,  50,  30) if piece.color == chess.WHITE else (220, 200, 180)
                text_col   = (255, 255, 255) if piece.color == chess.WHITE else (30,  30,  30)

                cx = x0 + sq_px // 2
                cy = y0 + sq_px // 2

                try:
                    bbox = draw.textbbox((0, 0), glyph, font=piece_font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except AttributeError:
                    tw, th = piece_font.getsize(glyph)

                tx = cx - tw // 2
                ty = cy - th // 2

                draw.text((tx + 1, ty + 1), glyph, font=piece_font, fill=shadow_col)
                draw.text((tx,     ty),     glyph, font=piece_font, fill=text_col)

    # ── Rank / file labels ────────────────────────────────────────────
    files = "abcdefgh"
    ranks = "12345678"
    for i in range(8):
        lbl_col = LIGHT if i % 2 == 0 else DARK
        fx = margin + i * sq_px + sq_px // 2 - 4
        fy = margin + 8 * sq_px + 2
        draw.text((fx, fy), files[i], font=label_font, fill=lbl_col)
        rx = 4
        ry = margin + (7 - i) * sq_px + sq_px // 2 - 6
        draw.text((rx, ry), ranks[i], font=label_font, fill=lbl_col)

    img.save(output)

# Play a full game using Computer Vision: ROBOTIC ARM (White) vs YOU (Black).
#
#  * White = Stockfish + Robotic Arm. The arm receives the UCI move and executes it.
#    When the arm finishes it sends back a "done" message over serial.  The program
#    then captures the board and verifies the arm made the right move.  That same
#    capture is kept as the baseline for detecting YOUR reply (no first 'r' needed).
#  * Black = you. The arm's completion capture is already the starting picture.
#    Make your move, take your hand away and press 'r' once: the camera detects it.
#  * Wrong move: the squares turn red. Put the pieces back BY HAND, take your hand
#    away and press 'z'. The camera then captures a fresh starting picture; make a
#    legal move and press 'r'.
from stockfish import Stockfish

import glob
import os
import shutil
import textwrap
import time
import cv2
import numpy as np
import json
import chess
import chess.engine
import serial
from board_renderer import board_to_png
from arm import ChessRobotArm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Settings ----------------
# Set to True if WHITE's pieces are at the TOP of the camera picture (board turned 180 degrees).
# Leave False when White is at the BOTTOM, next to the a1..h1 side you clicked in calibrate.py.
BOARD_ROTATED_180 = False
MIN_SECONDS_BETWEEN_R = 1.5   # a second 'r' sooner than this is ignored (make the move first)
NO_CHANGE_SCORE = 5.0         # if no square changes more than this, both frames are the same picture
CHANGE_MIN_SCORE = 15.0       # a square counts as "clearly changed" above this score ...
CHANGE_FRACTION = 0.3         # ... and above this fraction of the strongest square
CAMERA_INDEX = 0

STOCKFISH_PATH = None         # e.g. r"C:\stockfish\stockfish.exe"; None = search automatically
THINK_TIME_SECONDS = 1.0      # how long Stockfish thinks per move
SKILL_LEVEL = 10              # 0 (weakest) .. 20 (strongest); None = full strength
VERIFY_ENGINE_MOVE = True     # after the arm finishes, the camera checks it made Stockfish's move
ENGINE_COLOR = chess.WHITE    # Stockfish/arm plays White; you play Black

# Robot arm serial port — set to None to run without the arm (simulation mode)
ARM_PORT = '/dev/ttyUSB0'     # e.g. 'COM4' on Windows, '/dev/ttyUSB0' on Linux
# The signal string that the arm controller sends when it has finished a move
ARM_DONE_SIGNAL = "done"

FONT = cv2.FONT_HERSHEY_SIMPLEX

_stockfish = "/usr/games/stockfish"

if _stockfish is None:
    raise SystemExit("Stockfish not found. Put the Stockfish .exe in the 'stockfish' folder next to 'src', "
                     "or set STOCKFISH_PATH at the top of this file.")
engine = chess.engine.SimpleEngine.popen_uci(_stockfish)
if SKILL_LEVEL is not None:
    try:
        engine.configure({"Skill Level": SKILL_LEVEL})
    except Exception as ex:
        print(f"Note: could not set Skill Level ({ex}); Stockfish plays at full strength.")

# Load square polygons mapping (path relative to this script)
_SQDICT_PATH = os.path.join(_SCRIPT_DIR, 'sqdict1.json')
with open(_SQDICT_PATH, 'r') as fp:
    sq_points = json.load(fp)

if BOARD_ROTATED_180:
    # Rename every square as seen from the other side of the board: a1 <-> h8, e2 <-> d7, ...
    def _rotated_name(sq):
        return chr(ord('a') + 7 - (ord(sq[0]) - ord('a'))) + str(9 - int(sq[1]))
    sq_points = {_rotated_name(s): pts for s, pts in sq_points.items()}

# Returns the square given a point within the square
def find_square(x: float, y: float): 
    for square in sq_points:
        points = np.array(sq_points[square], np.int32)
        if cv2.pointPolygonTest(points, (x, y), False) > 0:
            return square
    return None


# ---------------- Move resolver ----------------
def squares_touched(board, move):
    sqs = {move.from_square, move.to_square}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) == 6:
            sqs |= {chess.square(7, rank), chess.square(5, rank)}
        else:
            sqs |= {chess.square(0, rank), chess.square(3, rank)}
    if board.is_en_passant(move):
        sqs.add(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    return sqs

def resolve_move(board, scores, min_total=15.0):
    """Choose the legal move that best explains the per-square change scores."""
    best, best_val = None, 0.0
    for m in board.legal_moves:
        touched = {chess.square_name(s) for s in squares_touched(board, m)}
        explained = sum(scores.get(s, 0.0) for s in touched)
        unexplained = sum(v for s, v in scores.items() if s not in touched)
        val = explained - 0.5 * unexplained
        if val > best_val:
            best, best_val = m, val
    return best if best is not None and best_val >= min_total else None

# ---------------- Wrong-move detection (python-chess rules) ----------------
def changed_squares(scores):
    """Squares that clearly changed, strongest first, plus the full ranking."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    cutoff = max(CHANGE_MIN_SCORE, CHANGE_FRACTION * ranked[0][1])
    return [sq for sq, s in ranked if s >= cutoff], ranked

def explain_illegal(board, frm, to):
    """Why moving the piece on `frm` to `to` is not allowed now (None = it is a legal move)."""
    piece = board.piece_at(frm)
    frm_name, to_name = chess.square_name(frm), chess.square_name(to)
    if piece is None:
        return f"There is no piece on {frm_name}."
    if piece.color != board.turn:
        return (f"That is a {chess.COLOR_NAMES[piece.color]} {chess.piece_name(piece.piece_type)}, "
                f"but it is {chess.COLOR_NAMES[board.turn]}'s turn.")
    target = board.piece_at(to)
    if target is not None and target.color == piece.color:
        return f"You cannot capture your own {chess.piece_name(target.piece_type)} on {to_name}."
    promotion = chess.QUEEN if piece.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7) else None
    move = chess.Move(frm, to, promotion=promotion)
    if move in board.legal_moves:
        return None
    name = chess.piece_name(piece.piece_type)
    if board.is_pseudo_legal(move):          # moves like a piece should, but breaks a king-safety rule
        if piece.piece_type == chess.KING:
            return f"The king cannot go to {to_name}: that square is attacked."
        if board.is_check():
            return "Your king is in check and this move does not get it out of check."
        return f"The {name} on {frm_name} is pinned: moving it would expose your king to check."
    if piece.piece_type == chess.PAWN:
        return ("A pawn moves straight forward one square (two from its start square) onto an empty square, "
                "or one square diagonally forward to capture.")
    alone = chess.Board(None)                # right geometry but something in the way?
    alone.set_piece_at(frm, piece)
    if to in alone.attacks(frm):
        return f"The path from {frm_name} to {to_name} is blocked by another piece."
    rule = {chess.KNIGHT: "A knight jumps in an L-shape (two squares one way, one square the other).",
            chess.BISHOP: "A bishop moves any distance along a diagonal.",
            chess.ROOK: "A rook moves any distance in a straight line along a rank or a file.",
            chess.QUEEN: "A queen moves any distance along a rank, a file or a diagonal.",
            chess.KING: "A king moves one square in any direction."}[piece.piece_type]
    return f"A {name} cannot move from {frm_name} to {to_name}. {rule}"

def explain_castling(board, kingside):
    """Why castling is not allowed now (None = it is allowed)."""
    color = board.turn
    rank = chess.square_rank(board.king(color))
    side = "kingside" if kingside else "queenside"
    has_rights = board.has_kingside_castling_rights(color) if kingside else board.has_queenside_castling_rights(color)
    if not has_rights:
        return f"Castling {side} is no longer allowed: the king or that rook has already moved."
    between = [chess.square(f, rank) for f in ((5, 6) if kingside else (1, 2, 3))]
    if any(board.piece_at(s) for s in between):
        return "There are pieces between the king and the rook."
    if board.is_check():
        return "You cannot castle while your king is in check."
    path = [chess.square(f, rank) for f in ((5, 6) if kingside else (3, 2))]
    if any(board.is_attacked_by(not color, s) for s in path):
        return "The king would pass through or land on an attacked square."
    return None

def diagnose_change(board, scores):
    """The camera saw a change that matches no legal move: say what happened.
    Returns (title, lines, squares_to_mark_red)."""
    changed, ranked = changed_squares(scores)
    shown = (changed or [sq for sq, _ in ranked[:2]])[:6]
    if not changed:
        return "UNCLEAR CHANGE", ["The change was too small to read as a move."], shown
    if len(changed) == 1:
        return "ONLY ONE SQUARE CHANGED", [f"Only {changed[0]} changed: a piece was added, removed or knocked over - "
                                           f"that is not a move."], shown
    if len(changed) == 2:
        a, b = (chess.parse_square(s) for s in changed)
        own = [s for s in (a, b) if board.piece_at(s) and board.piece_at(s).color == board.turn]
        if len(own) == 2:
            return "ILLEGAL MOVE", [f"{changed[0]} and {changed[1]} both hold your own pieces: "
                                    f"you cannot capture or swap with your own piece."], shown
        if len(own) == 1:
            frm, to = own[0], (b if own[0] == a else a)
        else:
            with_piece = [s for s in (a, b) if board.piece_at(s)]
            if not with_piece:
                return "UNCLEAR CHANGE", [f"No piece stands on {changed[0]} or {changed[1]}."], shown
            frm = with_piece[0]
            to = b if frm == a else a
        reason = explain_illegal(board, frm, to)
        title_move = f"{chess.square_name(frm)} -> {chess.square_name(to)}"
        if reason is None:
            return "UNCLEAR MOVE", [f"{title_move} would be legal, but the camera picture was not clear enough "
                                    f"to be sure."], shown
        return f"ILLEGAL MOVE: {title_move}", [reason], shown
    king = board.king(board.turn)
    if len(changed) <= 4 and king is not None and chess.square_file(king) == 4:     # castling attempt?
        rank = chess.square_rank(king)
        for kingside in (True, False):
            castle = {king, chess.square(6 if kingside else 2, rank), chess.square(7 if kingside else 0, rank),
                      chess.square(5 if kingside else 3, rank)}
            if {chess.parse_square(s) for s in changed} <= castle:
                reason = explain_castling(board, kingside)
                if reason:
                    return "ILLEGAL CASTLING", [reason], shown
                return "UNCLEAR CASTLING", ["Castling would be legal, but the camera picture was not clear enough."], shown
    return "MORE THAN ONE MOVE?", [f"{len(changed)} squares changed ({', '.join(changed[:6])}): more than one move "
                                   f"can explain. Did two pieces move, was a piece knocked over, or is a hand still "
                                   f"in the picture?"], shown

def check_clean(board, move, scores):
    """A legal move was found. Check nothing else happened. None = clean, else (title, lines, squares)."""
    changed, _ = changed_squares(scores)
    touched = {chess.square_name(s) for s in squares_touched(board, move)}
    extra = [s for s in changed if s not in touched]
    if extra and len(changed) <= 4:
        return diagnose_change(board, scores)       # e.g. a piece went to a square it cannot reach
    if extra:
        return ("MORE THAN ONE MOVE?",
                [f"{move.uci()} looks legal, but {', '.join(extra)} also changed.",
                 "Did two pieces move, or was a piece knocked over?"], changed[:6])
    if board.is_castling(move):
        missing = [s for s in touched if s not in changed]
        if missing:
            return ("CASTLING NEEDS BOTH PIECES",
                    [f"The king and the rook must both move; no change was seen on {', '.join(sorted(missing))}."],
                    sorted(touched))
    return None

def change_scores(img_before, img_after):
    """Per-square change score between two camera pictures (higher = more changed).
    Returns (scores, cleaned_difference_image)."""
    # Image Differencing with Gaussian blur to suppress noise/flicker
    blur1 = cv2.GaussianBlur(img_before, (5, 5), 0)
    blur2 = cv2.GaussianBlur(img_after, (5, 5), 0)
    gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, diff_thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

    # Clean noise using morphological opening + dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    diff_clean = cv2.dilate(diff_clean, kernel, iterations=2)

    # Square-wise difference scoring across all 64 squares
    scores = {}
    for sq, pts in sq_points.items():
        pts_arr = np.array(pts, np.int32)
        x, y, w, h = cv2.boundingRect(pts_arr)
        if w <= 0 or h <= 0:
            continue
        roi_diff = diff_clean[y:y+h, x:x+w]
        roi_pts = pts_arr - [x, y]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts], 255)
        scores[sq] = cv2.mean(roi_diff, mask=mask)[0]
    return scores, diff_clean


# ---------------- Telling the human how to make Stockfish's move ----------------
def describe_move(board, move):
    """Plain-language steps for making `move` on the real board (call BEFORE pushing the move)."""
    piece = board.piece_at(move.from_square)
    colour = chess.COLOR_NAMES[piece.color]
    other = chess.COLOR_NAMES[not piece.color]
    frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
    steps = []
    if board.is_en_passant(move):
        taken = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        steps.append(f"First remove the {other} pawn on {chess.square_name(taken)} (en passant)")
    elif board.is_capture(move):
        victim = chess.piece_name(board.piece_at(move.to_square).piece_type)
        steps.append(f"First remove the {other} {victim} on {to} (capture)")
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) == 6
        rook_from = chess.square_name(chess.square(7 if kingside else 0, rank))
        rook_to = chess.square_name(chess.square(5 if kingside else 3, rank))
        steps.append(f"Castling: move the {colour} king {frm} -> {to}")
        steps.append(f"then move the {colour} rook {rook_from} -> {rook_to}")
    else:
        steps.append(f"Move the {colour} {chess.piece_name(piece.piece_type)} {frm} -> {to}")
    if move.promotion:
        steps.append(f"Promote: replace the pawn with a {chess.piece_name(move.promotion)}")
    return steps

def put_text(frame, text, y, colour, scale=0.7):
    cv2.putText(frame, text, (20, y), FONT, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (20, y), FONT, scale, colour, 2)

def draw_instruction(frame, board, move, lines) -> None:
    """Colour the squares of Stockfish's move on the camera picture and write the steps."""
    overlay = frame.copy()
    from_sq, to_sq = chess.square_name(move.from_square), chess.square_name(move.to_square)
    for sq in squares_touched(board, move):                    # rook squares, en passant pawn
        name = chess.square_name(sq)
        if name not in (from_sq, to_sq):
            cv2.fillPoly(overlay, [np.array(sq_points[name], np.int32)], (255, 160, 0))
    cv2.fillPoly(overlay, [np.array(sq_points[from_sq], np.int32)], (0, 215, 255))   # yellow = from
    cv2.fillPoly(overlay, [np.array(sq_points[to_sq], np.int32)], (0, 200, 0))       # green  = to
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    centre = lambda sq: tuple(int(v) for v in np.mean(np.array(sq_points[sq], np.float32), axis=0))
    cv2.arrowedLine(frame, centre(from_sq), centre(to_sq), (0, 0, 255), 4, tipLength=0.2)
    y = 80
    put_text(frame, f"ARM (WHITE): {from_sq} -> {to_sq}  [waiting for arm...]", y, (0, 255, 255), 0.9)
    for line in lines:
        y += 32
        put_text(frame, line, y, (255, 255, 255))

# Outline the squares on video frame
def draw_problem(frame, problem) -> None:
    """Mark the squares of a wrong move red and write what is wrong."""
    overlay = frame.copy()
    for name in problem["squares"]:
        cv2.fillPoly(overlay, [np.array(sq_points[name], np.int32)], (0, 0, 255))
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    y = 80
    put_text(frame, problem["title"], y, (80, 80, 255), 0.9)
    for line in problem["lines"] + ["Put the pieces back as they were before your move.", problem["status"]]:
        for part in textwrap.wrap(line, 58):
            y += 32
            put_text(frame, part, y, (255, 255, 255))

def draw_outlines(sq_points: dict, frame, show_text=True) -> None:
    for square in sq_points:
        points = sq_points[square]
        points = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [points], True, (255, 255, 255), thickness=1)
        x, y, w, h = cv2.boundingRect(points)
        if show_text:
            cv2.putText(frame, square, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Show board using pure-Pillow renderer (no Cairo/GTK dependency needed)
def show_board(board: chess.Board, size=900, move=None, message=None) -> None:
    _output = os.path.join(_SCRIPT_DIR, 'output.png')
    if move is not None:
        sq1, sq2 = chess.parse_square(move[:2]), chess.parse_square(move[2:4])
        board_to_png(board, output=_output, size=size, highlight=[sq1, sq2])
    else:
        board_to_png(board, output=_output, size=size)
    image = cv2.imread(_output)
    banner_height = 64
    display = cv2.copyMakeBorder(image, banner_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
    status = message or 'V: show/hide camera frame'
    cv2.putText(display, status, (20, 42), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Game', display)

# Initialize Camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
for _ in range(15):          # let the camera settle (exposure) before the first picture is used
    cap.read()

def fresh_frame(fallback):
    """Read a few frames and return the newest (avoids an old frame waiting in the camera buffer)."""
    newest = None
    for _ in range(3):
        ok, f = cap.read()
        if ok:
            newest = f
    return newest if newest is not None else fallback

# ---------------- Robotic Arm ----------------
robot = ChessRobotArm(22, 22.5, port=ARM_PORT)

def arm_execute_move(board_state, move):
    """Send Stockfish's move to the robotic arm (UCI), then block until the arm signals done."""
    sq1 = chess.square_name(move.from_square)
    sq2 = chess.square_name(move.to_square)
    is_knight = board_state.piece_type_at(move.from_square) == chess.KNIGHT

    # Handle captures: discard the opponent's piece first
    if board_state.is_capture(move):
        if board_state.is_en_passant(move):
            ep_sq = chess.square_name(
                chess.square(chess.square_file(move.to_square),
                             chess.square_rank(move.from_square)))
            robot.discard(ep_sq)
        else:
            robot.discard(sq2)

    # Move the piece
    robot.move(sq1, sq2, knight=is_knight)

    # Handle castling: also move the rook
    if board_state.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) == 6
        rook_from = chess.square_name(chess.square(7 if kingside else 0, rank))
        rook_to   = chess.square_name(chess.square(5 if kingside else 3, rank))
        robot.move(rook_from, rook_to)

    # robot.move / robot.discard already block via wait_until_msg("done") internally,
    # so by the time we return the arm has signalled it is finished.

# -----------------------------------------------------------------------

initial = []
final = []
t_initial = 0.0
bounding_boxes = []
centers = []
highlights = set()

pending_move = None          # Stockfish's move that the arm still has to make
pending_lines = []
pending_initial = []         # camera picture taken just BEFORE the arm's move announcement
# After the arm finishes, arm_post_capture holds the fresh picture of the board.
# It serves double-duty: (1) verify the arm's move, (2) be the starting picture for human's reply.
arm_post_capture = None      # set after the arm signals done; cleared when human's move is accepted

restore = None               # set after a wrong move: waiting for you to put the board back and press 'z'
show_camera_frame = False

board = chess.Board()
cv2.namedWindow('Game', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
show_board(board)
cv2.waitKey(2)

print("--- You (Black) vs Robotic Arm / Stockfish (White) ---")
print("Instructions:")
print("1. WHITE = Robotic Arm (Stockfish). The arm executes the move automatically.")
print("   The camera then verifies the arm moved correctly.")
print("2. BLACK = you. The arm's completion picture is already the starting snapshot.")
print("   Make your move, take your hand away, then press 'r' once.")
print("3. 'm' (Black O-O) / 'n' (Black O-O-O) = castling shortcuts for your moves.")
print("4. WRONG MOVE: the squares turn red. Put the pieces back by hand, take your hand away and press 'z':")
print("   the camera captures a new starting picture, then you make a legal move and press 'r'.")
print("   't' = type your move if the camera got it wrong.")
print("5. 'v' = show/hide the camera frame. 'f' = accept arm's move without the camera check. 'q' = quit.")

try:
    while not board.is_game_over():
        ret, frame = cap.read()
        if not ret:
            break

        # ---- Stockfish / Arm's turn: choose a move and send it to the arm ----
        if board.turn == ENGINE_COLOR and pending_move is None:
            print("\n[WHITE] Stockfish is thinking...")
            pending_move = engine.play(board, chess.engine.Limit(time=THINK_TIME_SECONDS)).move
            pending_lines = describe_move(board, pending_move)
            print(f"[WHITE] STOCKFISH PLAYS: {pending_move.uci()}  ({board.san(pending_move)})")
            for line in pending_lines:
                print(f"        - {line}")

            # Capture board state BEFORE the arm moves (baseline for arm verification)
            pending_initial = fresh_frame(frame)

            show_board(board, move=pending_move.uci(),
                       message=f"ARM: {pending_move.uci()} ({board.san(pending_move)})  |  V: camera frame")

            print("        Sending move to the robotic arm...")
            # Execute on the arm (blocks until arm signals "done")
            arm_execute_move(board, pending_move)
            print("        Arm signalled done.")

            # ------------------------------------------------------------------
            # Arm is done: capture the board NOW.
            # This picture is used for TWO things:
            #   (a) verify the arm made the right move (optional, VERIFY_ENGINE_MOVE)
            #   (b) serve as the initial snapshot for detecting the human's reply
            # ------------------------------------------------------------------
            arm_post_capture = fresh_frame(frame)
            accepted = True

            if VERIFY_ENGINE_MOVE:
                scores, diff_clean = change_scores(pending_initial, arm_post_capture)
                cv2.imshow('Differenced', diff_clean)
                top_squares = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                print("Top changed squares: " + ", ".join(f"{sq} ({s:.1f})" for sq, s in top_squares[:4]))

                if top_squares[0][1] < NO_CHANGE_SCORE:
                    print("Camera check: no change seen yet. Press 'f' to accept anyway or fix and press Enter.")
                    accepted = False
                else:
                    seen = resolve_move(board, scores)
                    if seen is None or squares_touched(board, seen) != squares_touched(board, pending_move):
                        looks = f" (the change looks like {seen.uci()})" if seen else ""
                        print(f"Camera check: board change does not match arm's move "
                              f"{pending_move.uci()}{looks}. Fix the board and press 'f' to accept.")
                        accepted = False

            if accepted:
                board.push(pending_move)
                print(f"[WHITE] Arm move {pending_move.uci()} verified and accepted. Your turn (Black).")
                show_board(board, move=pending_move.uci())
                # arm_post_capture becomes the human's initial snapshot (no first 'r' needed)
                initial = arm_post_capture
                t_initial = time.monotonic()
                pending_move, pending_lines, pending_initial = None, [], []
                arm_post_capture = None
                bounding_boxes = []
                print("[BLACK] Your move. Make it on the board, take your hand away, then press 'r'.")

        raw_frame = frame.copy()   # clean copy, no overlay, used for differencing
        # Draw square grid overlay
        draw_outlines(sq_points, frame, show_text=True)

        # Draw current turn indicator on video feed
        current_turn = "WHITE" if board.turn == chess.WHITE else "BLACK"
        who = "Arm/Stockfish" if board.turn == ENGINE_COLOR else "you"
        cv2.putText(frame, f"Turn: {current_turn} ({who})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if board.turn == chess.WHITE else (0, 0, 255), 2)

        # Show Stockfish's move on the camera picture (while arm is executing or verification failed)
        if pending_move is not None:
            draw_instruction(frame, board, pending_move, pending_lines)

        # Draw status overlay for frame recording
        if len(initial) > 0 and len(final) == 0 and board.turn != ENGINE_COLOR:
            cv2.putText(frame, "Make move & press 'r'", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # After a wrong move: keep the red squares on screen until you press 'z'
        if restore is not None:
            draw_problem(frame, restore)

        # Draw detected bounding boxes
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if show_camera_frame:
            cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('v'):
            show_camera_frame = not show_camera_frame
            if show_camera_frame:
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', frame)
            else:
                cv2.destroyWindow('Frame')

        # 'r' while arm's move is waiting (should not happen normally, arm blocks)
        elif key == ord('r') and pending_move is not None:
            print("It is the arm's turn: wait for it to finish.")

        # 'r' while waiting for you to put the board back after a wrong move
        elif key == ord('r') and restore is not None:
            print("Put the pieces back by hand first, take your hand away, then press 'z'.")

        # 'z' = I have put the pieces back by hand: capture a new starting picture
        elif key == ord('z'):
            if restore is None:
                print("Nothing to restore: press 'r' to confirm your move.")
            else:
                restore = None
                final = []
                centers = []
                bounding_boxes = []
                initial = fresh_frame(raw_frame)
                t_initial = time.monotonic()
                print(f"[{current_turn}] Board restored. Starting picture captured. "
                      f"Make a LEGAL move, take your hand away, then press 'r'.")

        # ---- Record YOUR (Black) move with a SINGLE 'r' press ----
        # The initial snapshot was already set from the arm's completion capture.
        # One 'r' press takes the final frame and processes the move.
        elif key == ord('r'):
            now = time.monotonic()
            if len(initial) == 0:
                # This branch is a fallback: if somehow the initial snapshot was not set
                # (e.g. at the very start of the game or after a z-reset), capture it now
                # and ask the human to press 'r' again.
                initial = fresh_frame(raw_frame)
                t_initial = now
                bounding_boxes = []
                print(f"[{current_turn}] Initial frame captured. Make your physical move on the board, "
                      f"take your hand away, then press 'r' again...")
            elif now - t_initial < MIN_SECONDS_BETWEEN_R:
                print(f"[{current_turn}] Too soon - make your move first, then press 'r' again.")
            elif len(final) == 0:
                print(f"[{current_turn}] Final frame captured. Processing move detection...")
                final = fresh_frame(raw_frame)

                square_scores, diff_clean = change_scores(initial, final)
                cv2.imshow('Differenced', diff_clean)

                top_squares = sorted(square_scores.items(), key=lambda item: item[1], reverse=True)
                top_summary = [f"{sq} ({score:.1f})" for sq, score in top_squares[:4]]
                print(f"Top changed squares: {', '.join(top_summary)}")

                if top_squares[0][1] < NO_CHANGE_SCORE:
                    print("No change detected: both frames look the same. Press 'r', make the move, "
                          "then press 'r' again.")
                    initial = []
                    final = []
                    continue
                # Green boxes in the camera window show the two squares where a change was seen.
                bounding_boxes = [cv2.boundingRect(np.array(sq_points[s], np.int32)) for s, _ in top_squares[:2]]

                detected_move = resolve_move(board, square_scores)
                if detected_move:
                    problem = check_clean(board, detected_move, square_scores)
                else:
                    problem = diagnose_change(board, square_scores)

                if problem is None:
                    board.push(detected_move)
                    print(f"[{current_turn}] Auto-detected legal move: {detected_move.uci()}")
                    show_board(board, move=detected_move.uci())
                else:
                    title, lines, squares = problem
                    print(f"\n[{current_turn}] {title}")
                    for line in lines:
                        print(f"   - {line}")
                    print("   Put the pieces back by hand as they were before your move (the squares are red in the")
                    print("   camera window), take your hand away, then press 'z' - the camera takes a fresh")
                    print("   starting picture. ('t' = type your move if the camera got it wrong)")
                    restore = {"title": title, "lines": lines, "squares": squares,
                               "status": "Put the pieces back, then press 'z'."}

                # Reset detection variables for next turn
                initial = []
                final = []
                centers = []

        # 'f' = accept arm's move without (or despite failed) camera check
        elif key == ord('f'):
            if pending_move is None and arm_post_capture is None:
                print("Nothing to confirm: it is not the arm's turn.")
            elif pending_move is not None:
                # arm_post_capture may or may not exist; use it if available
                post = arm_post_capture if arm_post_capture is not None else fresh_frame(raw_frame)
                board.push(pending_move)
                print(f"[WHITE] Arm move {pending_move.uci()} force-accepted. Your turn (Black).")
                show_board(board, move=pending_move.uci())
                initial = post
                t_initial = time.monotonic()
                pending_move, pending_lines, pending_initial = None, [], []
                arm_post_capture = None
                bounding_boxes = []
                print("[BLACK] Your move. Make it on the board, take your hand away, then press 'r'.")

        # 't' = type your move (if the camera got it wrong)
        elif key == ord('t'):
            if pending_move is not None:
                print("It is the arm's turn: wait for it to finish.")
            else:
                print("\n>>> Type your move in THIS terminal window (for example e7e5), or press Enter to cancel.")
                uci = input("> ").strip()
                if uci:
                    try:
                        board.push_uci(uci)
                        print(f"[{current_turn}] Typed move accepted: {uci}")
                        show_board(board, move=uci)
                        restore = None
                        initial, final = [], []
                        bounding_boxes = []
                    except ValueError as ex:
                        print(f"Invalid or illegal move: {ex}")

        # Keyboard Castling Shortcuts for YOUR moves
        # Black Kingside (0-0)
        elif key == ord('m'):
            if chess.Move.from_uci('e8g8') in board.legal_moves:
                board.push_uci('e8g8')
                show_board(board, move='e8g8')
                bounding_boxes = []
                initial, final = [], []
                print("[BLACK] Castled Kingside (e8g8)")

        # Black Queenside (0-0-0)
        elif key == ord('n'):
            if chess.Move.from_uci('e8c8') in board.legal_moves:
                board.push_uci('e8c8')
                show_board(board, move='e8c8')
                bounding_boxes = []
                initial, final = [], []
                print("[BLACK] Castled Queenside (e8c8)")

        # Exit Command ('q')
        elif key == ord('q'):
            break

finally:
    try:
        engine.quit()
    except Exception:
        pass

# Display final board state
show_board(board)
print("Game over!" if board.is_game_over() else "Stopped.", "Result:", board.result())

cap.release()
cv2.destroyAllWindows()

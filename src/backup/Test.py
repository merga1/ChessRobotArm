# Play a full game using Computer Vision (Human vs Human Mode)

import os
import cv2
import numpy as np
import json
import time
import chess
from board_renderer import board_to_png

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Settings ----------------
# Set to True if WHITE's pieces are at the TOP of the camera picture (board turned 180 degrees).
# Leave False when White is at the BOTTOM, next to the a1..h1 side you clicked in calibrate.py.
BOARD_ROTATED_180 = False
MIN_SECONDS_BETWEEN_R = 1.5   # a second 'r' sooner than this is ignored (make the move first)
NO_CHANGE_SCORE = 5.0         # if no square changes more than this, both frames are the same picture

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


# ---------------- Move resolver (replaces the old pair-matching loop) ----------------
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

def hint_wrong_side(board, scores):
    b2 = board.copy()
    b2.turn = not board.turn
    return resolve_move(b2, scores)

# Outline the squares on video frame
def draw_outlines(sq_points: dict, frame, show_text=True) -> None:
    for square in sq_points:
        points = sq_points[square]
        points = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [points], True, (255, 255, 255), thickness=1)
        x, y, w, h = cv2.boundingRect(points)
        if show_text:
            cv2.putText(frame, square, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Show board using pure-Pillow renderer (no Cairo/GTK dependency needed)
def show_board(board: chess.Board, size=900, move=None) -> None:
    _output = os.path.join(_SCRIPT_DIR, 'output.png')
    if move is not None:
        sq1, sq2 = chess.parse_square(move[:2]), chess.parse_square(move[2:4])
        board_to_png(board, output=_output, size=size, highlight=[sq1, sq2])
    else:
        board_to_png(board, output=_output, size=size)
    cv2.imshow('Game', cv2.imread(_output))

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

def fresh_frame(fallback):
    """Read a few frames and return the newest (avoids an old frame waiting in the camera buffer)."""
    newest = None
    for _ in range(3):
        ok, f = cap.read()
        if ok:
            newest = f
    return newest if newest is not None else fallback

initial = []
final = []
t_initial = 0.0
bounding_boxes = []
centers = []
highlights = set()

board = chess.Board()
show_board(board)
cv2.waitKey(2)

print("--- Human vs Human CV Game Started ---")
print("Instructions:")
print("1. Before moving a piece, press 'r' to record the initial board state.")
print("2. Make the move physically.")
print("3. Press 'r' again to capture the new board state.")
print("4. Use 'm' (Black O-O) / 'n' (Black O-O-O) for castling, or '1'/'2' for White castling.")
print("5. Press 'q' to quit.")

while not board.is_game_over():
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()   # clean copy, no overlay, used for differencing
    # Draw square grid overlay
    draw_outlines(sq_points, frame, show_text=True)

    # Draw current turn indicator on video feed
    current_turn = "WHITE" if board.turn == chess.WHITE else "BLACK"
    cv2.putText(frame, f"Turn: {current_turn}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if board.turn == chess.WHITE else (0, 0, 255), 2)

    # Draw status overlay for frame recording
    if len(initial) > 0 and len(final) == 0:
        cv2.putText(frame, "RECORDING... Make move & press 'r'", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Draw detected bounding boxes
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # Record Move with 'r' Key
    if key == ord('r'):
        now = time.monotonic()
        if len(initial) == 0:
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

            # Image Differencing with Gaussian blur to suppress noise/flicker
            blur1 = cv2.GaussianBlur(initial, (5, 5), 0)
            blur2 = cv2.GaussianBlur(final, (5, 5), 0)
            gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            _, diff_thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

            # Clean noise using morphological opening + dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            diff_clean = cv2.dilate(diff_clean, kernel, iterations=2)
            cv2.imshow('Differenced', diff_clean)

            # Square-wise difference scoring across all 64 squares
            square_scores = {}
            for sq, pts in sq_points.items():
                pts_arr = np.array(pts, np.int32)
                x, y, w, h = cv2.boundingRect(pts_arr)
                if w <= 0 or h <= 0:
                    continue
                roi_diff = diff_clean[y:y+h, x:x+w]
                roi_pts = pts_arr - [x, y]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [roi_pts], 255)
                square_scores[sq] = cv2.mean(roi_diff, mask=mask)[0]

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
                board.push(detected_move)
                print(f"[{current_turn}] Auto-detected legal move: {detected_move.uci()}")
                show_board(board, move=detected_move.uci())
            else:
                other = hint_wrong_side(board, square_scores)
                if other:
                    print(f"HINT: the change looks like {other.uci()}, which is a legal move for the OTHER side. "
                          f"It is {current_turn}'s turn - either the wrong colour moved, or the board is flipped in the camera view "
                          f"(if White is at the TOP of the camera picture, set BOARD_ROTATED_180 = True at the top of Test.py).")
                print("\n>>> Could not auto-resolve a legal move. The camera windows are PAUSED until you answer.")
                print(">>> Click THIS terminal window and type a move such as e2e4, or just press Enter to record again.")
                uci = input("> ").strip()
                if uci:
                    try:
                        board.push_uci(uci)
                        show_board(board, move=uci)
                    except Exception as ex:
                        print(f"Invalid move: {ex}")
                else:
                    print("Re-record: put the pieces back to the position the program expects, then press 'r'.")

            # Reset detection variables for next turn
            initial = []
            final = []
            centers = []

    # Keyboard Castling Shortcuts
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

    # White Kingside (0-0)
    elif key == ord('1'):
        if chess.Move.from_uci('e1g1') in board.legal_moves:
            board.push_uci('e1g1')
            show_board(board, move='e1g1')
            bounding_boxes = []
            initial, final = [], []
            print("[WHITE] Castled Kingside (e1g1)")

    # White Queenside (0-0-0)
    elif key == ord('2'):
        if chess.Move.from_uci('e1c1') in board.legal_moves:
            board.push_uci('e1c1')
            show_board(board, move='e1c1')
            bounding_boxes = []
            initial, final = [], []
            print("[WHITE] Castled Queenside (e1c1)")

    # Exit Command ('q')
    elif key == ord('q'):
        break

# Display final board state
show_board(board)
print("Game Over! Result:", board.result())

cap.release()
cv2.destroyAllWindows()

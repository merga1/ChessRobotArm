#Play a full game using Computer Vision

import os
import cv2
import numpy as np
import json
import chess
import chess.engine
from board_renderer import board_to_png
from arm import ChessRobotArm
import random

# Build absolute path to Stockfish relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STOCKFISH  = os.path.join(_SCRIPT_DIR, "..", "stockfish",
                           "stockfish-windows-2022-x86-64-avx2.exe")
engine = chess.engine.SimpleEngine.popen_uci(_STOCKFISH)

_sq_path = os.path.join(_SCRIPT_DIR, 'sqdict.json')
if not os.path.exists(_sq_path):
    _sq_path = os.path.join(_SCRIPT_DIR, 'sqdict1.json')
with open(_sq_path, 'r') as fp:
    sq_points = json.load(fp)

#Returns the square given a point within the square
def find_square(x: float, y: float): 
    for square in sq_points:
        points = np.array(sq_points[square], np.int32)
        if cv2.pointPolygonTest(points, (x, y), False) > 0:
            return square
    return None

#Outline the squares
def draw_outlines(sq_points: dict, frame, show_text = False) -> None:
    for square in sq_points:
        points = sq_points[square]
        points = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [points], True, (255, 255, 255), thickness=1)
        x, y, w, h = cv2.boundingRect(points)
        if show_text:
            cv2.putText(frame, square, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Show board using pure-Pillow renderer (no Cairo/GTK dependency needed)
def show_board(board: chess.Board, size=900, move=None) -> None:
    _output = os.path.join(_SCRIPT_DIR, 'output.png')
    if move is not None:
        sq1, sq2 = chess.parse_square(move[:2]), chess.parse_square(move[2:4])
        board_to_png(board, output=_output, size=size, highlight=[sq1, sq2])
    else:
        board_to_png(board, output=_output, size=size)
    cv2.imshow('Game', cv2.imread(_output))

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

initial = []
final = []
bounding_boxes = []
centers = []
highlights = set()

board = chess.Board()
robot = ChessRobotArm(22, 22.5, port = 'COM4') #Initialize a 3DOF Robotic Arm with L1=22, L2=21, at port COM4
comp_move = True
show_board(board)
cv2.waitKey(2)

try:
    while not board.is_game_over():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not grab frame from camera.")
            break
        draw_outlines(sq_points, frame)
        cv2.imshow('Frame', frame)

        if comp_move:
            result = engine.play(board, chess.engine.Limit(time=random.random()))
            comp_move = result.move.uci()
            (sq1, sq2) = (comp_move[:2], comp_move[2:4])

            #Check for capture: remove opponent's piece before moving ours
            if board.is_capture(result.move):
                robot.discard(sq2)

            #For straight pawn moves:
            robot.move(sq1, sq2, knight=(board.piece_type_at(chess.parse_square(sq1)) == chess.KNIGHT))

            if comp_move == 'e1g1' and board.piece_type_at(chess.E1) == chess.KING: #Kingside
                robot.move('h1', 'f1')
            elif comp_move == 'e1c1' and board.piece_type_at(chess.E1) == chess.KING: #Queenside
                robot.move('a1', 'd1')
            
            board.push(result.move)
            print('Robot plays', result.move.uci())
            show_board(board, move=str(result.move))
            comp_move = False

        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y) in centers:
            square = find_square(x, y)
            if square:
                highlights.add(square)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if len(initial) == 0:
                initial = frame.copy()
                print("Recording")
            elif len(final) == 0:
                print('Move captured')
                final = frame.copy()

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
                candidate_squares = [sq for sq, score in top_squares[:8] if score > 3.0]

                # Auto-match candidates against python-chess legal moves
                detected_move = None
                for s1 in candidate_squares:
                    p = board.piece_at(chess.parse_square(s1))
                    if p is None or p.color != board.turn:
                        continue
                    for s2 in candidate_squares:
                        if s1 == s2:
                            continue
                        try:
                            m = chess.Move.from_uci(s1 + s2)
                            if m in board.legal_moves:
                                detected_move = m
                                break
                            mp = chess.Move.from_uci(s1 + s2 + 'q')
                            if mp in board.legal_moves:
                                detected_move = mp
                                break
                        except Exception:
                            continue
                    if detected_move:
                        break

                if detected_move:
                    board.push(detected_move)
                    print('User plays', detected_move.uci())
                    show_board(board, move=detected_move.uci())
                    comp_move = True
                else:
                    top_summary = [f"{sq} ({score:.1f})" for sq, score in top_squares[:4]]
                    print(f"Top changed squares: {', '.join(top_summary)}")
                    uci = input("Could not auto-resolve legal move. Enter UCI override (e.g. e7e5): ").strip()
                    if uci:
                        try:
                            board.push_uci(uci)
                            show_board(board, move=uci)
                            comp_move = True
                        except Exception as ex:
                            print(f"Invalid move: {ex}")

                initial = []
                final = []
                centers = []
                highlights = set()

        #Black castles 0-0
        elif key == ord('m'):
            move = 'e8g8'
            board.push_uci('e8g8')
            show_board(board, move=move)
            bounding_boxes = []
            comp_move = True

        #Black castles 0-0-0 
        elif key == ord('n'):
            move = 'e8c8'
            board.push_uci('e8c8')
            show_board(board, move=move)
            bounding_boxes = []
            comp_move = True

        #Exit command
        elif key == ord('q'):
            break

        cv2.imshow('Frame', frame)

finally:
    try:
        engine.quit()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
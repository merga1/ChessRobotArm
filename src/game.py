#Play a chess game without using Computer Vision

import os
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

board = chess.Board()
robot = ChessRobotArm(22, 22.5, port = 'COM4') #Initialize a 3DOF Robotic Arm with L1=22, L2=21, at port COM4

while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=random.random()))
    comp_move = result.move.uci()
    (sq1, sq2) = (comp_move[:2], comp_move[2:4])

    #Check for capture: discard opponent's piece first
    if board.is_capture(result.move):
        robot.discard(sq2)

    robot.move(sq1, sq2, knight=(board.piece_type_at(chess.parse_square(sq1)) == chess.KNIGHT))

    #Check for kingside or queenside castling (White)
    if comp_move == 'e1g1' and board.piece_type_at(chess.E1) == chess.KING: #Kingside
        robot.move('h1', 'f1')
    elif comp_move == 'e1c1' and board.piece_type_at(chess.E1) == chess.KING: #Queenside
        robot.move('a1', 'd1')  # rook moves a1->d1 in queenside castling
    
    board.push(result.move)
    print('Robot plays', result.move.uci())
    print(board)

    while True:
        user_move = input("Make your move (e.g. e7e5, or 'q' to quit): ").strip()
        if user_move.lower() == 'q':
            break
        try:
            move_obj = chess.Move.from_uci(user_move)
            if move_obj in board.legal_moves:
                board.push(move_obj)
                break
            else:
                print(f"Illegal move '{user_move}'. Try again.")
        except Exception:
            print(f"Invalid UCI format '{user_move}'. Format should be like e7e5.")
    if user_move.lower() == 'q':
        break
    print(board)

engine.quit()
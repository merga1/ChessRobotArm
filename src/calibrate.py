# Chessboard Calibration Tool
# Generates 'sqdict.json' by clicking the 4 outer corners of the physical board.

import os
import cv2
import numpy as np
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables to store click points
pts_src = []

def select_corners(event, x, y, flags, param):
    global pts_src
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts_src) < 4:
            pts_src.append([x, y])
            print(f"Corner {len(pts_src)} selected: ({x}, {y})")

def calibrate():
    global pts_src
    
    cap = cv2.VideoCapture(0)
    
    # Lock camera resolution to match your main game script
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", select_corners)

    print("\n--- CHESSBOARD CALIBRATION ---")
    print("Click the 4 OUTER CORNERS of the board in this EXACT order:")
    print("  1. Bottom-Left  (near a1)")
    print("  2. Bottom-Right (near h1)")
    print("  3. Top-Right    (near h8)")
    print("  4. Top-Left     (near a8)")
    print("Press 'c' to calculate and save calibration.")
    print("Press 'r' to reset points if you make a mistake.")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access camera feed.")
            break

        display_frame = frame.copy()

        # Draw selected corners and connecting lines
        for i, pt in enumerate(pts_src):
            cv2.circle(display_frame, (pt[0], pt[1]), 6, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(pts_src) == 4:
            pts_arr = np.array(pts_src, np.int32)
            cv2.polylines(display_frame, [pts_arr], True, (0, 255, 0), 2)

        cv2.imshow("Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Reset points
        if key == ord('r'):
            pts_src = []
            print("Points reset. Click the 4 corners again.")

        # Compute and Save sqdict.json
        elif key == ord('c'):
            if len(pts_src) != 4:
                print("Error: You must select exactly 4 corners first!")
                continue

            print("Calculating 8x8 grid polygons...")

            # Define normalized output square size (e.g., 800x800 pixels for complete board)
            board_dim = 800
            sq_size = board_dim // 8

            pts_dst = np.array([
                [0, board_dim],          # Bottom-Left
                [board_dim, board_dim],  # Bottom-Right
                [board_dim, 0],          # Top-Right
                [0, 0]                   # Top-Left
            ], dtype=np.float32)

            src_mat = np.array(pts_src, dtype=np.float32)

            # Homography matrix mapping 2D camera coordinates to normalized board grid
            H, _ = cv2.findHomography(pts_dst, src_mat)

            files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

            sqdict = {}

            # Generate polygon coordinates for all 64 squares
            for col_idx, f in enumerate(files):
                for row_idx, r in enumerate(ranks):
                    square_name = f + r
                    
                    # Ideal pixel bounds in normalized board space
                    x1 = col_idx * sq_size
                    x2 = (col_idx + 1) * sq_size
                    y1 = (8 - int(r)) * sq_size
                    y2 = (8 - int(r) + 1) * sq_size

                    # Four corner points of individual square
                    sq_pts_norm = np.array([
                        [[x1, y2]],
                        [[x2, y2]],
                        [[x2, y1]],
                        [[x1, y1]]
                    ], dtype=np.float32)

                    # Transform normalized points back into live camera coordinate space
                    sq_pts_cam = cv2.perspectiveTransform(sq_pts_norm, H)
                    
                    # Format points into int list [[x1, y1], [x2, y2], ...]
                    pts_list = sq_pts_cam.reshape(-1, 2).astype(int).tolist()
                    sqdict[square_name] = pts_list

            # Write to JSON file
            out_file = os.path.join(_SCRIPT_DIR, 'sqdict1.json')
            with open(out_file, 'w') as fp:
                json.dump(sqdict, fp, indent=4)

            print(f"Successfully generated and saved '{out_file}'!")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    calibrate()

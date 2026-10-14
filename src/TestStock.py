from stockfish import Stockfish

# Initialize with the correct Ubuntu ARM64 path
stockfish = Stockfish(path="/usr/games/stockfish")

# Set parameters optimized for Jetson Orin's CPU cores 
# (Jetson Orin Nano Super has 8 ARM Cortex-A78AE cores)
stockfish.update_engine_parameters({"Threads": 4, "Hash": 2048})

print("Stockfish is working!")
print(stockfish.get_board_visual())



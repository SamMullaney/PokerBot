import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = os.getenv("DEVICE", "cpu")

CONF_THRES = float(os.getenv("CONF_THRES", 0.7))

MODEL_TABLE_PATH = os.path.join(
    BASE_DIR,
    os.getenv("MODEL_TABLE_PATH", "vision/models/table_detector_v1.pt")
)

CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", 30))
CAPTURE_REGION = None
WINDOW_NAME = "PokerBot Debug (q to quit)"

TABLE_ROIS = {
    "player_card_1": (0.43972, 0.74722, 0.02955, 0.0779),
    "player_card_2": (0.48818, 0.74722, 0.03073, 0.0779),
    "community_card_1": (0.30851, 0.42448, 0.02955, 0.07631),
    "community_card_2": (0.38652, 0.42448, 0.02955, 0.07631),
    "community_card_3": (0.46336, 0.42448, 0.02955, 0.07631),
    "community_card_4": (0.54137, 0.42448, 0.02955, 0.07631),
    "community_card_5": (0.61820, 0.42448, 0.02955, 0.07631),
    "pot_area": (0.45981, 0.60254, 0.08038, 0.03180),
}

SEAT_ROIS = {
    "seat_0": (0.44693, 0.11058, 0.10259, 0.12954),
    "seat_1": (0.80307, 0.21643, 0.10024, 0.13112),
    "seat_2": (0.08962, 0.21643, 0.10259, 0.13112),
    "seat_3": (0.86321, 0.43602, 0.10024, 0.12638),
    "seat_4": (0.02830, 0.43760, 0.10377, 0.12796),
    "seat_5": (0.79599, 0.65087, 0.10967, 0.13270),
    "seat_6": (0.09906, 0.65087, 0.10259, 0.12796),
    "seat_7": (0.57547, 0.10427, 0.10731, 0.13270),
    "seat_8": (0.31722, 0.10742, 0.10613, 0.13112),
    "seat_9": (0.84316, 0.28120, 0.10377, 0.13902),
    "seat_10": (0.04835, 0.28752, 0.10495, 0.12638),
    "seat_11": (0.75000, 0.67615, 0.09906, 0.12954),
    "seat_12": (0.14505, 0.66667, 0.10259, 0.14060),
    "seat_13": (0.77358, 0.18641, 0.10259, 0.13112),
    "seat_14": (0.11675, 0.18167, 0.10613, 0.13586),
    "seat_15": (0.55071, 0.10585, 0.10024, 0.12954),
    "seat_16": (0.34670, 0.10585, 0.09906, 0.12796), 
}

SEAT_EXCLUSION_GROUPS = [
    ["seat_7", "seat_15"],
    ["seat_8", "seat_16"],
    ["seat_5", "seat_11"],
    ["seat_6", "seat_12"],
    ["seat_1", "seat_9", "seat_13"],
    ["seat_2", "seat_10", "seat_14"],
]


 

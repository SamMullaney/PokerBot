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





 

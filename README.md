Temporary main funcion to look at:
from capture.screen_capture import ScreenCapture
from vision.card_classifier import detect_cards
from vision.ocr import read_pot_and_bet
from state.encoder import encode_state
from poker.hero_policy import decide_action
import time

def main():
    capture = ScreenCapture(fps=20)

    try:
        while True:
            frame = capture.get_frame()

            if frame is None:
                continue  # no new frame yet

            # 1. Vision
            community_cards, player_cards = detect_cards(frame)
            pot, bet = read_pot_and_bet(frame)

            # 2. State encoding
            state = encode_state(
                player_cards=player_cards,
                community_cards=community_cards,
                pot=pot,
                bet=bet
            )

            # 3. Poker logic
            action = decide_action(state)

            # 4. Output (for MVP)
            print(action)

            time.sleep(0.05)  # light throttle

    finally:
        capture.stop()

if __name__ == "__main__":
    main()

DETECTION PIPELINE:
DXcam Frame
   ↓
Table Detector (YOLO or similar)
   ↓
For each table:
   ├── Normalize table geometry
   ├── Crop fixed card regions
   │     └── Card classifier (CNN / ViT)
   ├── Crop pot/bet text regions
   │     └── OCR
   ├── Detect dealer button / blinds
   │     └── Rule-based logic
   ↓
Structured game state



GTO Main Algoritms:

- Ficticious Play
- Counterfactual regret minimization

These solvers only work with a certain amount of bet sizes. The more bet size intervals added the more likley the solver will "break". Computation becomes to lengthy or nearly impossible

Memory issues with a solver:

- must run on a cloud
- Programming lang might have to vary for the math behind the system
- 
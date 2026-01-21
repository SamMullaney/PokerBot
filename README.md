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



table_x1 = 696
table_y1 = 186
table_x2 = 1542
table_y2 = 815

table_w = 846
table_h = 629

player_card_one_x1 = 1068
player_car_one_y1 = 656
player_card_one_x2 = 1093
player_card_one_y2 = 705

player_card_two_x1 = 1109
player_card_two_y1 = 656
player_card_two_x2 = 1135
player_card_two_y2 = 706

- First Player card bboxes percentages
x_pct = (pcox1 - tablex1) / table_w = 0.43972
y_pct = (pcoy1 - tabley1) / table_h = 0.74722
w_pct = (pcox2 - pcox1) / table_w = 0.02955
h_pct = (pcoy2 - pcoy1) / table_h = 0.0779

- Second player card bboxes percentages
x_pct = 0.48818
y_pct = 0.74722
w_pct = 0.03073
h_pct = 0.0779




Player position possible rois:

table_x1 = 
table_y1 =
table_x2 = 
table_y2 = 

table_w = 
table_h = 
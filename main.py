from capture.screen_capture import ScreenCapture
from vision.table_detector import TableDetector
from vision.player_detector import PlayerDetector
from vision.card_detector import CardClassifier
from vision.draw import draw_tables, draw_roi, draw_players
from app.debug_viewer import DebugViewer
import config
import cv2

def main():
    cap = ScreenCapture(fps=config.CAPTURE_FPS, region=config.CAPTURE_REGION, output_color="BGR")
    detector = TableDetector(
        model_path=config.MODEL_TABLE_PATH,
        conf_thres=config.CONF_THRES,
        device=config.DEVICE
    )
    player_detector = PlayerDetector(
        edge_ratio_threshold=0.1,
        laplacian_var_threshold=100.0,
        nms_overlap_threshold=0.45
    )
    card_model_path = "vision/models/tiny_corner_net_best_cardv4.pt"
    card_clf = CardClassifier(weights_path=card_model_path, device="cpu")  # or "cuda"

    viewer = DebugViewer(config.WINDOW_NAME)

    print("Starting PokerBot. Press 'q' to quit.")

    try:
        while True:
            frame = cap.get_frame()
            if frame is None:
                continue

            tables = detector.detect(frame)
            
            annotated = draw_tables(frame, tables)
            all_detected_cards = {}

            for table in tables:
                if table is not None and table.w > 0 and table.h > 0:
                    # Draw table ROIs
                    for roi_name, (x_pct, y_pct, w_pct, h_pct) in config.TABLE_ROIS.items():
                        roi = table.roi_from_rel(x_pct=x_pct, y_pct=y_pct, w_pct=w_pct, h_pct=h_pct)
                        draw_roi(annotated, roi, roi_name)
                    
                    # Detect and draw players
                    players = player_detector.detect(frame, table)
                    draw_players(annotated, players, color=(0, 165, 255))  # Orange color for players
                    
                    # Detect cards in player and community card ROIs
                    for roi_name, (x_pct, y_pct, w_pct, h_pct) in config.TABLE_ROIS.items():
                        if "card" in roi_name:
                            x1, y1, x2, y2 = table.roi_from_rel(x_pct=x_pct, y_pct=y_pct, w_pct=w_pct, h_pct=h_pct)
                            
                            # Extract the card region from the frame
                            if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                                card_region = frame[y1:y2, x1:x2]
                                
                                if card_region.size > 0:
                                    # Classify the card using the tiny CNN model
                                    prediction = card_clf.predict_corner(card_region)
                                    
                                    # Apply confidence threshold
                                    CONFIDENCE_THRESHOLD = 0.6
                                    if prediction.card_conf < CONFIDENCE_THRESHOLD:
                                        label = "NO_CARD"
                                        card_conf = prediction.card_conf
                                    else:
                                        label = prediction.label
                                        card_conf = prediction.card_conf
                                    
                                    # Store the result
                                    all_detected_cards[roi_name] = {
                                        "label": label,
                                        "rank_conf": prediction.rank_conf,
                                        "suit_conf": prediction.suit_conf,
                                        "card_conf": card_conf
                                    }
                                    
                                    # Draw the card label on the frame
                                    text = f"{label} ({card_conf:.2f})"
                                    cv2.putText(annotated, text, (x1, y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


            key = viewer.show(annotated, detected_cards=all_detected_cards)
            viewer.log_fps()

            if key == ord("q"):
                break

    finally:
        cap.stop()
        viewer.close()
        print("Stopped.")


if __name__ == "__main__":
    main()

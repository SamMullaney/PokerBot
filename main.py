from capture.screen_capture import ScreenCapture
from vision.table_detector import TableDetector
from vision.player_detector import PlayerDetector
from vision.draw import draw_tables, draw_roi, draw_players
from app.debug_viewer import DebugViewer
import config


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
    viewer = DebugViewer(config.WINDOW_NAME)

    print("Starting PokerBot. Press 'q' to quit.")

    try:
        while True:
            frame = cap.get_frame()
            if frame is None:
                continue

            tables = detector.detect(frame)
            
            annotated = draw_tables(frame, tables)

            for table in tables:
                if table is not None and table.w > 0 and table.h > 0:
                    # Draw table ROIs
                    for roi_name, (x_pct, y_pct, w_pct, h_pct) in config.TABLE_ROIS.items():
                        roi = table.roi_from_rel(x_pct=x_pct, y_pct=y_pct, w_pct=w_pct, h_pct=h_pct)
                        draw_roi(annotated, roi, roi_name)
                    
                    # Detect and draw players
                    players = player_detector.detect(frame, table)
                    draw_players(annotated, players, color=(0, 165, 255))  # Orange color for players

            key = viewer.show(annotated)
            viewer.log_fps()

            if key == ord("q"):
                break

    finally:
        cap.stop()
        viewer.close()
        print("Stopped.")


if __name__ == "__main__":
    main()

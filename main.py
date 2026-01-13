from capture.screen_capture import ScreenCapture
from vision.table_detector import TableDetector
from vision.draw import draw_tables, draw_roi
from app.debug_viewer import DebugViewer
import config


def main():
    cap = ScreenCapture(fps=config.CAPTURE_FPS, region=config.CAPTURE_REGION, output_color="BGR")
    detector = TableDetector(
        model_path=config.MODEL_TABLE_PATH,
        conf_thres=config.CONF_THRES,
        device=config.DEVICE
    )
    viewer = DebugViewer(config.WINDOW_NAME)

    print("Starting PokerBot. Press 'q' to quit.")

    try:
        while True:
            frame = cap.get_frame()
            if frame is None:
                continue

            tables = detector.detect(frame)
            primary = detector.pick_primary(tables)

            annotated = draw_tables(frame, tables, highlight=primary)

            # Example: show a test ROI inside the primary table (for future OCR/cards)
            # These are dummy values just to prove the ROI scaling works.
            if primary is not None and primary.w > 0 and primary.h > 0:
                pot_roi = primary.roi_from_rel(x_pct=0.55, y_pct=0.40, w_pct=0.12, h_pct=0.05)
                draw_roi(annotated, pot_roi, "pot_roi(example)")

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

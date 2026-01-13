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
            

            annotated = draw_tables(frame, tables)


            for a in tables:
                if a is not None and a.w > 0 and a.h > 0:
                    test_roi = a.roi_from_rel(x_pct=0.1, y_pct=0.1, w_pct=0.3, h_pct=0.2)
                    draw_roi(annotated, test_roi, "test_roi")

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

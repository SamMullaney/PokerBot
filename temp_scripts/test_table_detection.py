import time
import cv2
import numpy as np
import dxcam
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"vision\models\table_detector_v1.pt"
WINDOW_NAME = "PokerBot Live (press q to quit)"
CONF_THRES = 0.35
TARGET_FPS = 30  # lower = easier to use desktop while testing

# If you know the screen region, set it here to reduce load:
# region = (left, top, right, bottom)
REGION = None

# -----------------------------
# LOAD MODEL (force CPU to avoid CUDA driver warning)
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# START CAPTURE
# -----------------------------
camera = dxcam.create(output_color="BGR")  # BGR so OpenCV displays correctly
camera.start(target_fps=TARGET_FPS, region=REGION, video_mode=True)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("Starting live detection. Press 'q' to quit.")

last_t = time.time()
frames = 0

try:
    while True:
        frame = camera.get_latest_frame()  # blocks until a new frame
        if frame is None:
            continue

        # Run detection (CPU)
        results = model.predict(frame, conf=CONF_THRES, device="cpu", verbose=False)

        # Draw boxes
        annotated = frame.copy()
        r0 = results[0]
        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                name = model.names.get(cls, str(cls))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{name} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # Show one window only
        cv2.imshow(WINDOW_NAME, annotated)

        # MUST have waitKey for UI events; also lets you press q
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # simple fps log
        frames += 1
        now = time.time()
        if now - last_t >= 1.0:
            print(f"Live FPS: {frames / (now - last_t):.1f}")
            last_t = now
            frames = 0

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Stopped.")

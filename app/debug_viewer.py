import time
import cv2


class DebugViewer:
    def __init__(self, window_name: str):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self._last_t = time.time()
        self._frames = 0

    def show(self, frame):
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key

    def log_fps(self, prefix="Live FPS"):
        self._frames += 1
        now = time.time()
        if now - self._last_t >= 1.0:
            print(f"{prefix}: {self._frames / (now - self._last_t):.1f}")
            self._last_t = now
            self._frames = 0

    def close(self):
        cv2.destroyAllWindows()

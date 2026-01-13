from typing import List, Optional
from ultralytics import YOLO
from state.table_state import TableBox


class TableDetector:
    #Test confidence thresholds. Seems very low
    def __init__(self, model_path: str, conf_thres: float = 0.7, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.device = device

    def detect(self, frame) -> List[TableBox]:
        """
        Returns all detected poker tables in the frame.
        """
        results = self.model.predict(frame, conf=self.conf_thres, device=self.device, verbose=False)
        r0 = results[0]

        tables: List[TableBox] = []
        if r0.boxes is None or len(r0.boxes) == 0:
            return tables

        for b in r0.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            # cls = int(b.cls[0])  # you can check class if you add more later
            tables.append(TableBox(x1, y1, x2, y2, conf))

        # Optional: sort left-to-right then top-to-bottom (useful when multiple tables)
        tables.sort(key=lambda t: (t.y1, t.x1))
        return tables
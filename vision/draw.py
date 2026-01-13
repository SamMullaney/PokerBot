import cv2
from typing import List, Optional, Tuple
from state.table_state import TableBox


def draw_tables(frame, tables: List[TableBox], highlight: Optional[TableBox] = None):
    annotated = frame.copy()

    for t in tables:
        x1, y1, x2, y2 = t.as_xyxy()
        
        color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"table {t.conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    return annotated


def draw_roi(frame, roi_xyxy: Tuple[int, int, int, int], label: str = "roi"):
    x1, y1, x2, y2 = roi_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

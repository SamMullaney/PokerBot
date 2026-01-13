from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TableBox:
    # pixel coords in full-screen image
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def crop(self, frame):
        # frame is a numpy array (H,W,3)
        return frame[self.y1:self.y2, self.x1:self.x2]

    def roi_from_rel(self, x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> Tuple[int, int, int, int]:
        """
        Convert a table-relative ROI (0..1) into full-frame pixel bbox (x1,y1,x2,y2).
        """
        x = int(self.x1 + x_pct * self.w)
        y = int(self.y1 + y_pct * self.h)
        w = int(w_pct * self.w)
        h = int(h_pct * self.h)
        return (x, y, x + w, y + h)

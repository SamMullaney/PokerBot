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


def draw_players(frame, players: List, color: Tuple[int, int, int] = (0, 165, 255)):
    """
    Draw detected players on the frame.
    Only draws players that are marked as occupied.
    Includes drawing additional ROIs (name, VPIP, stack, bet, pos) for occupied seats.
    
    Args:
        frame: Frame to draw on
        players: List of PlayerSeat objects
        color: BGR color for drawing (default: orange)
    """
    # Color map for different ROI types
    roi_colors = {
        "name": (0, 255, 255),      # Cyan
        "VPIP": (255, 0, 255),      # Magenta
        "stack": (255, 255, 0),     # Yellow
        "bet": (0, 255, 0),         # Green
        "pos": (255, 128, 0),       # Light Blue
    }
    
    for player in players:
        if player.is_occupied:
            x1, y1, x2, y2 = player.x1, player.y1, player.x2, player.y2
            
            # Draw rectangle with thicker line to distinguish from other ROIs
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with seat info
            label = f"{player.seat_name} ({player.confidence:.2f})"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # Draw additional ROIs for this occupied seat
            for roi_name, (roi_x1, roi_y1, roi_x2, roi_y2) in player.rois.items():
                roi_color = roi_colors.get(roi_name, (128, 128, 128))  # Default gray
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)
                cv2.putText(frame, roi_name, (roi_x1, max(20, roi_y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)

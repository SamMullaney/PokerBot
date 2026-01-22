import cv2
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# These need to be set to your table boundaries (from first roi calibration)
TABLE_X1 = 537
TABLE_Y1 = 199
TABLE_X2 = 1385
TABLE_Y2 = 832

TABLE_W = TABLE_X2 - TABLE_X1
TABLE_H = TABLE_Y2 - TABLE_Y1

# IMAGE_PATH to calibrate against
IMAGE_PATH = "C:\\Users\\xvasc\\OneDrive\\Pictures\\Screenshots\\9playersseat16.png"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
display = img.copy()
clicks = []

# Draw table boundaries on image for reference
"""
cv2.rectangle(display, (TABLE_X1, TABLE_Y1), (TABLE_X2, TABLE_Y2), (0, 255, 0), 2)
cv2.putText(display, "Table Bounds (green)", (TABLE_X1, TABLE_Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
"""

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        click_num = len(clicks)
        roi_num = (click_num - 1) // 2 + 1
        click_in_roi = (click_num - 1) % 2 + 1
        
        print(f"\nClick #{click_num} (ROI #{roi_num}, point {click_in_roi}/2): x={x}, y={y}")
        
        # Visual marker with text
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display, f"#{click_num}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow("Image", display)
        
        # Every 2 clicks, calculate and display the ROI
        if len(clicks) % 2 == 0:
            roi_index = len(clicks) // 2
            x1, y1 = clicks[-2]
            x2, y2 = clicks[-1]
            
            # Calculate percentages
            x_pct = (x1 - TABLE_X1) / TABLE_W
            y_pct = (y1 - TABLE_Y1) / TABLE_H
            w_pct = (x2 - x1) / TABLE_W
            h_pct = (y2 - y1) / TABLE_H
            
            print(f"\n✓ ROI #{roi_index} completed:")
            print(f"  x_pct={x_pct:.5f}")
            print(f"  y_pct={y_pct:.5f}")
            print(f"  w_pct={w_pct:.5f}")
            print(f"  h_pct={h_pct:.5f}")
            print(f"  Add to config.py: \"roi_{roi_index}\": ({x_pct:.5f}, {y_pct:.5f}, {w_pct:.5f}, {h_pct:.5f})")
            
            # Draw the ROI rectangle on display
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display, f"ROI_{roi_index}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("Image", display)

# Instructions
print("="*60)
print("ROI CALIBRATION TOOL")
print("="*60)
print(f"Table boundaries: ({TABLE_X1}, {TABLE_Y1}) to ({TABLE_X2}, {TABLE_Y2})")
print(f"Table dimensions: {TABLE_W} x {TABLE_H}")
print("\nCLICK ORDER for each ROI (2 clicks per ROI):")
print("  1. Top-left corner of ROI")
print("  2. Bottom-right corner of ROI")
print("\nThen repeat for the next ROI")
print("Press 'q' to quit and see summary")
print("="*60 + "\n")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", mouse_callback)
cv2.imshow("Image", display)

# Event loop
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if clicks:
    print(f"\n\nTotal clicks: {len(clicks)}")
    print("\nCopy the following ROIs to TABLE_ROIS in config.py:")
    print("-" * 60)
    
    for i in range(0, len(clicks), 2):
        if i + 1 < len(clicks):  # Need at least 2 points
            x1, y1 = clicks[i]
            x2, y2 = clicks[i + 1]
            
            x_pct = (x1 - TABLE_X1) / TABLE_W
            y_pct = (y1 - TABLE_Y1) / TABLE_H
            w_pct = (x2 - x1) / TABLE_W
            h_pct = (y2 - y1) / TABLE_H
            
            roi_num = i // 2 + 1
            print(f'    "roi_{roi_num}": ({x_pct:.5f}, {y_pct:.5f}, {w_pct:.5f}, {h_pct:.5f}),')

cv2.destroyAllWindows()

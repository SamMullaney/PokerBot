"""
Measure a table's pixel bounding box on a screenshot.

Click the table's top-left corner, then its bottom-right corner; the
coordinates are printed. Feed them to tools/calibrate_rois.py.

Usage (from the project root):
    python -m tools.measure_table_bounds path/to/screenshot.png
"""
import argparse
import cv2

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("image", help="Path to a screenshot of the poker client")
IMAGE_PATH = parser.parse_args().image

# STATE
points = []

def click(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        # Once two points are clicked, print table coords
        if len(points) == 2:
            print_table_coords(points)
            points = []  # reset for next box


def print_table_coords(pts):
    (x1, y1), (x2, y2) = pts

    # Ensure correct ordering
    left   = min(x1, x2)
    right  = max(x1, x2)
    top    = min(y1, y2)
    bottom = max(y1, y2)

    print("\n" + "="*40)
    print("TABLE COORDINATES:")
    print(f"x1 = {left}")
    print(f"y1 = {top}")
    print(f"x2 = {right}")
    print(f"y2 = {bottom}")
    print(f"width = {right - left}")
    print(f"height = {bottom - top}")
    print("="*40 + "\n")


# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

cv2.namedWindow("Click top-left, then bottom-right", cv2.WINDOW_NORMAL)
cv2.imshow("Click top-left, then bottom-right", image)
cv2.setMouseCallback("Click top-left, then bottom-right", click)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "C:\\Users\\xvasc\\OneDrive\\Pictures\\Screenshots\\Screenshot 2026-01-13 190121.png"   # path to your uploaded screenshot

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

display = img.copy()

# -----------------------------
# Mouse callback
# -----------------------------
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked pixel at: x={x}, y={y}")

        # Visual marker
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", display)

# -----------------------------
# Show window
# -----------------------------
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", mouse_callback)
cv2.imshow("Image", display)

print("Click on the image to get pixel coordinates.")
print("Press 'q' to quit.")

# -----------------------------
# Event loop
# -----------------------------
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()

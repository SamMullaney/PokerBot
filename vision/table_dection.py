from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight and fast

model.train(
    data="C:/Users/xvasc/PokerBot/vision/table_detection_images/data.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    workers=8,
    device=0
)

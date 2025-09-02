from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    name="cow_pose_yolov8l",
    device="cpu",
    patience=10  # כדי שיעצור אם אין שיפור ב-10 אפוקים רצופים
)


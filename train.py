from ultralytics import YOLO

# טען את תצורת המודל ליולו מסוג pose (אפשר גם s/m/l)
model = YOLO("yolov8n-pose.pt")  # או yolov8s-pose.yaml לפי משאבים

# הרצת האימון
model.train(
    data="data.yaml",      # הנתיב לקובץ data.yaml
    epochs=50,            # כמה איטרציות (שנה לפי גודל הדאטה)
    imgsz=640,             # גודל תמונה (ברירת מחדל 640)
    name="cow_pose_model"  # שם לריצה בתיקיית runs/
)

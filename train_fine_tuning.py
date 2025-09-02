from ultralytics import YOLO

# טוען את המודל המאומן הקודם
model = YOLO(r"C:\Users\yehudah\OneDrive - ARO Volcani Center\Desktop\train on other dataset\runs\pose\fine_tuned_on_my_data3\weights\best.pt")

# מאמן רק על התמונות שלי (fine-tuning)
model.train(
    data="data_for_fine_tuned.yaml",
    epochs=50,                # פחות אפוקים לבדיקות מהירות
    imgsz=416,                # גודל תמונה קטן יותר - מאיץ את האימון
    batch=32,                 # נצול הזיכרון שלך (אפשר לבדוק גם 32)
    workers=4,                # תאוצה בטעינת נתונים (לא להשאיר 0)
    cache=True,               # טעינה מהירה יותר של התמונות
    name="fine_tuned_on_my_data",
    device="cpu",            # אין לי מעבד גראפי CUDA
    patience=10,             # early stopping אחרי 10 אפוקים בלי שיפור
)

from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os



model_path = 'C:\\Users\yehudah\Desktop\cows_detectoion\\runs\detect\head_detection7\weights\\best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"לא נמצא מודל בנתיב: {model_path}")
model = YOLO(model_path)

# טוען את המודל
# model = YOLO('yolov8n.pt')  # המודל הכללי של יולו
model = YOLO('C:\\Users\yehudah\Desktop\cows_detectoion\\runs\detect\head_detection7\weights\\best.pt') # המודל שלי

def upload_image():
    """ פונקציה להעלאת תמונה """
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        process_and_display(image)

def process_and_display(image):
    """פונקציה לעיבוד התמונה ויצירת הצג"""
    # המרת התמונה ל-BGR (כפי שדרוש עבור YOLO)
    results = model(image)  # הרצת המודל על התמונה

    # הצגת התמונה עם תוצאות הזיהוי (עבור התוצאה הראשונה ברשימה)
    results[0].show()  # מציג את התמונה עם תוצאות הזיהוי של התמונה הראשונה

# יצירת ממשק GUI
root = tk.Tk()
root.title("זיהוי אובייקטים עם YOLOv8")

# כפתור להעלאת תמונה
upload_button = tk.Button(root, text="העלאת תמונה", command=upload_image)
upload_button.pack()

# הרצת הממשק
root.mainloop()

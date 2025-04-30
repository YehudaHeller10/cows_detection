import os
from PIL import Image

# קביעת גודל חדש שתואם ל-YOLO (למשל 1280x720)
CROP_WIDTH = 640
CROP_HEIGHT = 640
OUTPUT_DIR = "output_cutted_for_yolo"

# יצירת תיקיית יעד אם לא קיימת
os.makedirs(OUTPUT_DIR, exist_ok=True)

# עובר על כל התמונות בתקייה הנוכחית
for filename in os.listdir("."):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(filename)
        width, height = image.size

        if width < CROP_WIDTH or height < CROP_HEIGHT:
            print(f"Skipping {filename} (too small for crop)")
            continue

        # חיתוך מימין לשמאל
        num_parts = width // CROP_WIDTH  # מספר החלקים שמתקבלים
        for i in range(num_parts):
            left = i * CROP_WIDTH
            top = (height - CROP_HEIGHT) // 2  # חיתוך ממרכז התמונה
            right = left + CROP_WIDTH
            bottom = top + CROP_HEIGHT

            cropped_image = image.crop((left, top, right, bottom))

            # יצירת שם קובץ חדש לכל חתיכה
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}_part{i+1}.jpg"
            output_path = os.path.join(OUTPUT_DIR, new_filename)
            cropped_image.save(output_path, quality=100)

            print(f"Saved cropped image to {output_path}")

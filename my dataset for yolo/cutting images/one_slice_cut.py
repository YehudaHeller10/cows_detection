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

        # חיתוך ממרכז התמונה
        left = (width - CROP_WIDTH) // 2
        top = (height - CROP_HEIGHT) // 2
        right = left + CROP_WIDTH
        bottom = top + CROP_HEIGHT

        cropped_image = image.crop((left, top, right, bottom))

        # שמירה באיכות מקורית
        output_path = os.path.join(OUTPUT_DIR, filename)
        cropped_image.save(output_path, quality=100)

        print(f"Saved cropped image to {output_path}")

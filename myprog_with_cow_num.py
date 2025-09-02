import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os

# --- Constants ---
MODEL_PATH = r"C:\Users\yehudah\OneDrive - ARO Volcani Center\Desktop\train on other dataset\runs\pose\fine_tuned_on_my_data3\weights\best.pt"  # the fine tuned model
#MODEL_PATH = r"C:\Users\yehudah\OneDrive - ARO Volcani Center\Desktop\train on other dataset\runs\pose\cow_pose_yolov8m\weights\best.pt"  # the model on other dataset
STATIONS_CSV_PATH = "C:\\Users\yehudah\OneDrive - ARO Volcani Center\Desktop\\train on other dataset\COW_STATION_LOCATIONS_PART1.csv"  # Make sure this file exists

KPT_NAMES = ["left_eye", "right_eye", "center_nose", "left_mouth", "right_mouth"]

# Colors and Fonts
BOX_COLOR = (255, 0, 0)  # Red for box in regular processing
AGGREGATE_BOX_COLOR = (0, 0, 255)  # Blue for aggregated boxes
KPT_COLOR = (0, 255, 0)  # Green for keypoints
TEXT_COLOR = (255, 255, 255)  # White for keypoint text
LABEL_COLOR = (255, 0, 0)  # Red for label text
NUMBER_COLOR = (0, 255, 255)  # Cyan for cow number
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_BOX = 0.6
FONT_SCALE_KPT = 0.4
FONT_SCALE_NUMBER = 1.0
CIRCLE_RADIUS = 5
LINE_THICKNESS = 2

# Zoom parameters
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0
ZOOM_FACTOR = 1.1

# NMS parameters - default values
DEFAULT_CONF_THRESHOLD = 0.15
DEFAULT_IOU_THRESHOLD = 0.45
MERGE_IOU_THRESHOLD = 0.6  # IoU threshold for merging detections in deep search

# Parameters for dynamic inference size selection
THRESHOLD_FOR_LARGE_IMG_SIZE_ACTIVATION = 800
LARGE_IMG_SIZE = 1280


# --- Lighting Effect Functions ---
def apply_original(image):
    return image


def apply_darker(image):
    return cv2.convertScaleAbs(image, alpha=0.7, beta=-40)


def apply_brighter(image):
    return cv2.convertScaleAbs(image, alpha=1.2, beta=20)


def apply_higher_contrast(image):
    return cv2.convertScaleAbs(image, alpha=1.5, beta=0)


def apply_lower_contrast(image):
    return cv2.convertScaleAbs(image, alpha=0.7, beta=0)


def apply_warm_tone(image):
    b, g, r = cv2.split(image)
    r_new = np.clip(r * 1.15, 0, 255).astype(np.uint8)
    b_new = np.clip(b * 0.9, 0, 255).astype(np.uint8)
    return cv2.merge((b_new, g, r_new))


def apply_cool_tone(image):
    b, g, r = cv2.split(image)
    b_new = np.clip(b * 1.15, 0, 255).astype(np.uint8)
    r_new = np.clip(r * 0.9, 0, 255).astype(np.uint8)
    return cv2.merge((b_new, g, r_new))


def apply_grainy_effect(image):
    noise_intensity = 20
    noise_mat = np.zeros_like(image, dtype=np.int16)
    cv2.randn(noise_mat, (0, 0, 0), (noise_intensity, noise_intensity, noise_intensity))
    noisy_image = cv2.add(image.astype(np.int16), noise_mat)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def calculate_iou(box1, box2):
    # box format: (x1, y1, x2, y2)
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


class App:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Cow Pose Viewer - Enhanced with Station Numbering")
        self.root.geometry("1300x800")  # Adjusted width for buttons

        self.stations_data = self.load_stations_data()
        self.lighting_presets = [
            {'name': 'Original', 'function': apply_original},
            {'name': 'Darker', 'function': apply_darker},
            {'name': 'Brighter', 'function': apply_brighter},
            {'name': 'Higher Contrast', 'function': apply_higher_contrast},
            {'name': 'Lower Contrast', 'function': apply_lower_contrast},
            {'name': 'Warm Tone', 'function': apply_warm_tone},
            {'name': 'Cool Tone', 'function': apply_cool_tone},
            {'name': 'Grainy Effect', 'function': apply_grainy_effect},
        ]
        self.current_lighting_preset_index = 0

        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load YOLO model from path:\n{MODEL_PATH}\nError: {e}")
            self.root.quit()
            return

        self.control_frame = Frame(root_window)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Label(self.control_frame, text="Confidence:").pack(side=tk.LEFT, padx=(5, 0))
        self.conf_scale = Scale(self.control_frame, from_=0.05, to=0.95, resolution=0.05, orient=tk.HORIZONTAL,
                                length=120)
        self.conf_scale.set(DEFAULT_CONF_THRESHOLD)
        self.conf_scale.pack(side=tk.LEFT, padx=(0, 5))
        self.conf_scale.bind("<ButtonRelease-1>", self.reprocess_current_image_event)

        tk.Label(self.control_frame, text="IOU:").pack(side=tk.LEFT, padx=(5, 0))
        self.iou_scale = Scale(self.control_frame, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL, length=120)
        self.iou_scale.set(DEFAULT_IOU_THRESHOLD)
        self.iou_scale.pack(side=tk.LEFT, padx=(0, 5))
        self.iou_scale.bind("<ButtonRelease-1>", self.reprocess_current_image_event)

        tk.Button(self.control_frame, text="Reprocess", command=self.reprocess_current_image_event).pack(side=tk.LEFT,
                                                                                                         padx=5)
        tk.Button(self.control_frame, text="Cycle Lighting", command=self.cycle_lighting_and_reprocess).pack(
            side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Deep Aggregate Search", command=self.deep_aggregate_search).pack(
            side=tk.LEFT, padx=5)

        self.current_effect_label = tk.Label(self.control_frame, text="Effect: Original")
        self.current_effect_label.pack(side=tk.LEFT, padx=5)

        self.stats_label = tk.Label(self.control_frame, text="Cows Detected: 0")
        self.stats_label.pack(side=tk.RIGHT, padx=10)
        self.stations_label = tk.Label(self.control_frame,
                                       text=f"Stations: {len(self.stations_data) if self.stations_data else 'Not Loaded'}")
        self.stations_label.pack(side=tk.RIGHT, padx=10)

        self.canvas = tk.Canvas(root_window, bg='grey10')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        menu = tk.Menu(root_window)
        root_window.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.open_image_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        self.canvas.bind("<MouseWheel>", self.zoom_image_event)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.canvas.config(cursor="arrow"))

        self.current_scale = 1.0
        self.tk_img = None
        self.original_cv2_img = None
        self.current_display_cv2_img = None
        self.pil_img_for_display = None
        self.img_x_on_canvas = 0
        self.img_y_on_canvas = 0
        self._last_drag_x = 0
        self._last_drag_y = 0

    def load_stations_data(self):
        try:
            if not os.path.exists(STATIONS_CSV_PATH):
                messagebox.showwarning("Stations File Not Found",
                                       f"Stations file {STATIONS_CSV_PATH} not found.\nSystem will operate without station numbering.")
                return None
            df = pd.read_csv(STATIONS_CSV_PATH)
            stations = []
            for _, row in df.iterrows():
                # Ensure all necessary columns exist in the CSV or handle missing ones
                if not all(col in row for col in ['label_name', 'bbox_x', 'bbox_width', 'bbox_y', 'bbox_height']):
                    print(f"Skipping row due to missing columns: {row}")
                    continue
                stations.append({
                    'station_number': int(row['label_name']),
                    'center_x': row['bbox_x'] + row['bbox_width'] / 2,
                    'center_y': row['bbox_y'] + row['bbox_height'] / 2,
                    'bbox_x': row['bbox_x'],  # Store original bbox for assign_cow_to_station
                    'bbox_y': row['bbox_y'],
                    'bbox_width': row['bbox_width'],
                    'bbox_height': row['bbox_height']
                })
            stations.sort(key=lambda x: x['center_x'], reverse=True)
            print(f"Successfully loaded {len(stations)} stations.")
            return stations
        except Exception as e:
            messagebox.showerror("Error Loading Stations",
                                 f"Error loading stations file:\n{e}")
            return None

    def assign_cow_to_station(self, cow_bbox_xyxy):
        if self.stations_data is None:
            return None

        x1, y1, x2, y2 = cow_bbox_xyxy
        cow_center_x = (x1 + x2) / 2
        cow_center_y = (y1 + y2) / 2

        min_distance = float('inf')
        best_station_number = None
        margin = 100

        for station in self.stations_data:
            if (station['bbox_x'] - margin <= cow_center_x <= station['bbox_x'] + station['bbox_width'] + margin and
                    station['bbox_y'] - margin <= cow_center_y <= station['bbox_y'] + station['bbox_height'] + margin):
                distance = np.sqrt((cow_center_x - station['center_x']) ** 2 +
                                   (cow_center_y - station['center_y']) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_station_number = station['station_number']
        return best_station_number

    def update_effect_label(self):
        if hasattr(self, 'lighting_presets') and self.lighting_presets:
            preset_name = self.lighting_presets[self.current_lighting_preset_index]['name']
            self.current_effect_label.config(text=f"Effect: {preset_name}")
        else:
            self.current_effect_label.config(text="Effect: -")

    def cycle_lighting_and_reprocess(self):
        if self.original_cv2_img is None:
            messagebox.showinfo("Information", "Please load an image first.")
            return
        self.current_lighting_preset_index = (self.current_lighting_preset_index + 1) % len(self.lighting_presets)
        self.update_effect_label()
        self.reprocess_current_image()

    def reprocess_current_image_event(self, event=None):
        self.reprocess_current_image()

    def reprocess_current_image(self):
        if self.original_cv2_img is None:
            return
        try:
            effect_function = self.lighting_presets[self.current_lighting_preset_index]['function']
            img_with_effect = effect_function(self.original_cv2_img.copy())
            self.current_display_cv2_img = self.process_image_for_yolo(img_with_effect, draw_keypoint_names=True)
            self.display_image()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reprocessing the image: {e}")
            self.current_display_cv2_img = self.original_cv2_img.copy() if self.original_cv2_img is not None else None
            self.display_image()

    def open_image_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return

        try:
            raw_cv2_img = cv2.imread(path)
            if raw_cv2_img is None:
                messagebox.showerror("Image Load Error", f"Could not load image from path: {path}")
                return

            self.original_cv2_img = raw_cv2_img
            self.current_lighting_preset_index = 0
            self.update_effect_label()
            self.current_scale = 1.0
            self.img_x_on_canvas = 0
            self.img_y_on_canvas = 0
            self.reprocess_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the image: {e}")
            self.original_cv2_img = None
            self.current_display_cv2_img = None
            self.display_image()

    def process_image_for_yolo(self, cv2_img_to_process, draw_keypoint_names=False):
        annotated_img = cv2_img_to_process.copy()
        original_h, original_w = cv2_img_to_process.shape[:2]

        conf_threshold = self.conf_scale.get()
        iou_threshold = self.iou_scale.get()

        model_args = {
            "verbose": False, "conf": conf_threshold, "iou": iou_threshold, "classes": [0]
        }
        if max(original_h, original_w) > THRESHOLD_FOR_LARGE_IMG_SIZE_ACTIVATION:
            model_args["imgsz"] = LARGE_IMG_SIZE

        results_list = self.model(cv2_img_to_process, **model_args)
        num_detections_on_this_img = 0

        if results_list and results_list[0].boxes is not None and len(results_list[0].boxes) > 0:
            results = results_list[0]
            num_detections_on_this_img = len(results.boxes.xyxy)

            for i in range(num_detections_on_this_img):
                box = results.boxes.xyxy[i]
                x1, y1, x2, y2 = map(int, box)
                confidence = float(results.boxes.conf[i])
                station_number = self.assign_cow_to_station((x1, y1, x2, y2))

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), BOX_COLOR, LINE_THICKNESS)
                label_text_parts = [f"C:{confidence:.2f}"]

                if station_number is not None:
                    label_text_parts.insert(0, f"S{station_number}")
                    number_text = str(station_number)
                    (text_w, text_h), _ = cv2.getTextSize(number_text, FONT_FACE, FONT_SCALE_NUMBER, LINE_THICKNESS * 2)
                    number_x = x1 + (x2 - x1 - text_w) // 2
                    number_y = y1 - 10
                    cv2.rectangle(annotated_img, (number_x - 5, number_y - text_h - 5),
                                  (number_x + text_w + 5, number_y + 5), (0, 0, 0), -1)
                    cv2.putText(annotated_img, number_text, (number_x, number_y),
                                FONT_FACE, FONT_SCALE_NUMBER, NUMBER_COLOR, LINE_THICKNESS * 2)
                else:
                    label_text_parts.insert(0, "NoSt")

                final_label_text = " ".join(label_text_parts)
                cv2.putText(annotated_img, final_label_text, (x1, y2 + 20),
                            FONT_FACE, FONT_SCALE_BOX, LABEL_COLOR, LINE_THICKNESS)

                if results.keypoints is not None and i < len(results.keypoints.data):
                    kpts_tensor = results.keypoints.data[i]
                    for kp_idx, (x_kp, y_kp, v_kp) in enumerate(kpts_tensor):
                        if v_kp > 0:
                            x_kp, y_kp = int(x_kp), int(y_kp)
                            cv2.circle(annotated_img, (x_kp, y_kp), CIRCLE_RADIUS, KPT_COLOR, -1)
                            if draw_keypoint_names and kp_idx < len(KPT_NAMES):
                                kp_name = KPT_NAMES[kp_idx]
                                cv2.putText(annotated_img, kp_name, (x_kp + 4, y_kp - 4),
                                            FONT_FACE, FONT_SCALE_KPT, TEXT_COLOR, 1)

        self.stats_label.config(text=f"Cows (current view): {num_detections_on_this_img}")
        return annotated_img

    def deep_aggregate_search(self):
        if self.original_cv2_img is None:
            messagebox.showinfo("Information", "Please load an image first to perform deep search.")
            return

        messagebox.showinfo("Deep Search", "Starting deep aggregate search. This may take a moment...")

        all_found_cows = []
        conf_threshold = self.conf_scale.get()
        iou_threshold = self.iou_scale.get()
        base_image_for_drawing = self.original_cv2_img.copy()

        for i, preset in enumerate(self.lighting_presets):
            self.current_effect_label.config(text=f"Searching: {preset['name']} ({i + 1}/{len(self.lighting_presets)})")
            self.root.update_idletasks()

            effect_function = preset['function']
            img_with_effect = effect_function(self.original_cv2_img.copy())
            original_h, original_w = img_with_effect.shape[:2]
            model_args = {
                "verbose": False, "conf": conf_threshold, "iou": iou_threshold, "classes": [0]
            }
            if max(original_h, original_w) > THRESHOLD_FOR_LARGE_IMG_SIZE_ACTIVATION:
                model_args["imgsz"] = LARGE_IMG_SIZE

            results_list = self.model(img_with_effect, **model_args)

            if results_list and results_list[0].boxes is not None:
                results = results_list[0]
                for j in range(len(results.boxes.xyxy)):
                    box = tuple(map(int, results.boxes.xyxy[j]))
                    confidence = float(results.boxes.conf[j])
                    station = self.assign_cow_to_station(box)
                    keypoints_for_this_cow = results.keypoints.data[j] if results.keypoints is not None and j < len(
                        results.keypoints.data) else None
                    all_found_cows.append({
                        'box': box,
                        'station': station,
                        'confidence': confidence,
                        'keypoints': keypoints_for_this_cow
                    })

        all_found_cows.sort(key=lambda x: x['confidence'], reverse=True)
        unique_cows_to_draw = []
        for cow_detection in all_found_cows:
            is_unique = True
            for existing_cow in unique_cows_to_draw:
                if calculate_iou(cow_detection['box'], existing_cow['box']) > MERGE_IOU_THRESHOLD:
                    is_unique = False
                    break
            if is_unique:
                unique_cows_to_draw.append(cow_detection)

        print(
            f"Deep Search: Found {len(all_found_cows)} total detections, {len(unique_cows_to_draw)} unique cows after merging.")

        for cow in unique_cows_to_draw:
            x1, y1, x2, y2 = cow['box']
            station_number = cow['station']
            confidence = cow['confidence']
            kpts_tensor = cow['keypoints']

            cv2.rectangle(base_image_for_drawing, (x1, y1), (x2, y2), AGGREGATE_BOX_COLOR, LINE_THICKNESS)
            label_text_parts = [f"C:{confidence:.2f}"]
            if station_number is not None:
                label_text_parts.insert(0, f"S{station_number}")
                number_text = str(station_number)
                (text_w, text_h), _ = cv2.getTextSize(number_text, FONT_FACE, FONT_SCALE_NUMBER, LINE_THICKNESS * 2)
                number_x = x1 + (x2 - x1 - text_w) // 2
                number_y = y1 - 10
                cv2.rectangle(base_image_for_drawing, (number_x - 5, number_y - text_h - 5),
                              (number_x + text_w + 5, number_y + 5), (0, 0, 0), -1)
                cv2.putText(base_image_for_drawing, number_text, (number_x, number_y),
                            FONT_FACE, FONT_SCALE_NUMBER, NUMBER_COLOR, LINE_THICKNESS * 2)
            else:
                label_text_parts.insert(0, "NoSt")

            final_label_text = " ".join(label_text_parts)
            cv2.putText(base_image_for_drawing, final_label_text, (x1, y2 + 20),
                        FONT_FACE, FONT_SCALE_BOX, LABEL_COLOR, LINE_THICKNESS)

            if kpts_tensor is not None:
                for kp_idx, (x_kp, y_kp, v_kp) in enumerate(kpts_tensor):
                    if v_kp > 0:
                        x_kp, y_kp = int(x_kp), int(y_kp)
                        cv2.circle(base_image_for_drawing, (x_kp, y_kp), CIRCLE_RADIUS, KPT_COLOR, -1)
                        if kp_idx < len(KPT_NAMES):
                            kp_name = KPT_NAMES[kp_idx]
                            cv2.putText(base_image_for_drawing, kp_name, (x_kp + 4, y_kp - 4),
                                        FONT_FACE, FONT_SCALE_KPT, TEXT_COLOR, 1)

        self.current_display_cv2_img = base_image_for_drawing
        self.stats_label.config(text=f"Aggregated Cows: {len(unique_cows_to_draw)}")
        self.current_effect_label.config(text="Effect: Aggregated Result")
        self.display_image()
        messagebox.showinfo("Deep Search Complete",
                            f"Deep aggregate search finished.\nFound {len(unique_cows_to_draw)} unique cows.")

    def display_image(self):
        if self.current_display_cv2_img is None:
            self.canvas.delete("all")
            self.tk_img = None
            if self.original_cv2_img is not None:  # If an original was loaded but processing failed or no detections
                self.canvas.create_text(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                        text="No image to display or error in processing.", fill="white")
            return

        img_rgb = cv2.cvtColor(self.current_display_cv2_img, cv2.COLOR_BGR2RGB)
        pil_img_original = Image.fromarray(img_rgb)

        new_width = int(pil_img_original.width * self.current_scale)
        new_height = int(pil_img_original.height * self.current_scale)

        if new_width <= 0 or new_height <= 0:
            return

        try:
            resized_pil_img = pil_img_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error resizing image for display: {e}")
            return

        self.pil_img_for_display = resized_pil_img
        self.tk_img = ImageTk.PhotoImage(resized_pil_img)

        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # Smart centering / clamping for pan
        if new_width < canvas_w:  # Center if image is smaller than canvas
            self.img_x_on_canvas = (canvas_w - new_width) / 2
        else:  # Clamp pan to keep image edges within or at canvas bounds
            self.img_x_on_canvas = min(0, max(self.img_x_on_canvas, canvas_w - new_width))

        if new_height < canvas_h:
            self.img_y_on_canvas = (canvas_h - new_height) / 2
        else:
            self.img_y_on_canvas = min(0, max(self.img_y_on_canvas, canvas_h - new_height))

        self.canvas.create_image(self.img_x_on_canvas, self.img_y_on_canvas, anchor="nw", image=self.tk_img)

    def zoom_image_event(self, event):
        if self.pil_img_for_display is None:
            return

        canvas_x_mouse = self.canvas.canvasx(event.x)
        canvas_y_mouse = self.canvas.canvasy(event.y)

        img_coord_x_before_zoom = (canvas_x_mouse - self.img_x_on_canvas) / self.current_scale
        img_coord_y_before_zoom = (canvas_y_mouse - self.img_y_on_canvas) / self.current_scale

        old_scale = self.current_scale
        if event.delta > 0:
            self.current_scale *= ZOOM_FACTOR
        else:
            self.current_scale /= ZOOM_FACTOR
        self.current_scale = max(MIN_ZOOM, min(self.current_scale, MAX_ZOOM))

        if abs(self.current_scale - old_scale) < 1e-9:
            return

        self.img_x_on_canvas = canvas_x_mouse - (img_coord_x_before_zoom * self.current_scale)
        self.img_y_on_canvas = canvas_y_mouse - (img_coord_y_before_zoom * self.current_scale)

        self.display_image()

    def on_canvas_press(self, event):
        if self.pil_img_for_display is None:
            return
        self._last_drag_x = event.x
        self._last_drag_y = event.y
        self.canvas.config(cursor="fleur")

    def on_canvas_motion(self, event):
        if self.pil_img_for_display is None:
            return

        dx = event.x - self._last_drag_x
        dy = event.y - self._last_drag_y

        # Only pan if image is larger than canvas in that dimension
        if self.pil_img_for_display.width * self.current_scale > self.canvas.winfo_width():
            self.img_x_on_canvas += dx
        if self.pil_img_for_display.height * self.current_scale > self.canvas.winfo_height():
            self.img_y_on_canvas += dy

        self._last_drag_x = event.x
        self._last_drag_y = event.y

        self.display_image()
        self.canvas.config(cursor="fleur")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
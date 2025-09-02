import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, Frame
from PIL import Image, ImageTk  # PIL is still used for displaying images in Tkinter
from ultralytics import YOLO
import numpy as np
import time

# --- Constants ---
# IMPORTANT: Update this path to where your model file is located
#MODEL_PATH = r"C:\Users\yehudah\OneDrive - ARO Volcani Center\Desktop\train on other dataset\runs\pose\cow_pose_yolov8m\weights\best.pt"
MODEL_PATH = r"C:\Users\yehudah\OneDrive - ARO Volcani Center\Desktop\train on other dataset\runs\pose\fine_tuned_on_my_data3\weights\best.pt"


KPT_NAMES = ["left_eye", "right_eye", "center_nose", "left_mouth", "right_mouth"]

# Colors and Fonts (for OpenCV)
BOX_COLOR = (255, 0, 0)  # Blue for regular boxes
ACCUMULATED_BOX_COLOR_PALETTE = [
    (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (255, 128, 0)  # Added more colors for more effects
]
KPT_COLOR = (0, 255, 0)  # Green for keypoints
TEXT_COLOR = (255, 255, 255)  # White for text
# LABEL_COLOR removed as background color is taken from current_box_color
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_BOX = 0.5
FONT_SCALE_KPT = 0.4
CIRCLE_RADIUS = 5
LINE_THICKNESS = 2

# Zoom parameters
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0
ZOOM_FACTOR = 1.1

# NMS parameters - default values
DEFAULT_CONF_THRESHOLD = 0.15
DEFAULT_IOU_THRESHOLD = 0.45

# Dynamic inference size parameters
THRESHOLD_FOR_LARGE_IMG_SIZE_ACTIVATION = 800
LARGE_IMG_SIZE = 1280


# --- Effect Functions ---
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


def apply_clahe(image):  # Contrast Limited Adaptive Histogram Equalization
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    updated_lab_img = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(updated_lab_img, cv2.COLOR_LAB2BGR)
    return clahe_img


def apply_median_blur(image):
    """Applies Median Blur to reduce noise."""
    return cv2.medianBlur(image, 5)  # Kernel size 5, can be 3


def apply_sharpen(image):
    """Applies a sharpening filter."""
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


class App:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Cow Pose Viewer - Deep Scan Version")
        self.root.geometry("1300x850")  # Adjusted width for potentially longer effect names

        self.lighting_presets = [
            {'name': 'Original', 'function': apply_original},
            {'name': 'Darker', 'function': apply_darker},
            {'name': 'Brighter', 'function': apply_brighter},
            {'name': 'High Contrast', 'function': apply_higher_contrast},
            {'name': 'Low Contrast', 'function': apply_lower_contrast},
            {'name': 'Warm Tone', 'function': apply_warm_tone},
            {'name': 'Cool Tone', 'function': apply_cool_tone},
            {'name': 'Grainy Effect', 'function': apply_grainy_effect},
            {'name': 'CLAHE', 'function': apply_clahe},
            {'name': 'Median Blur', 'function': apply_median_blur},  # New effect
            {'name': 'Sharpen', 'function': apply_sharpen},  # New effect
        ]
        self.current_lighting_preset_index = 0
        self.all_collected_detections = []
        self.active_image_path = None
        self.is_deep_scan_results_displayed = False

        try:
            self.model = YOLO(MODEL_PATH)
            print(f"YOLO Model loaded successfully from: {MODEL_PATH}")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load YOLO model from {MODEL_PATH}:\n{e}")
            self.root.quit()
            return

        self.control_frame = Frame(root_window)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.conf_label = tk.Label(self.control_frame, text="Confidence Thresh:")
        self.conf_label.pack(side=tk.LEFT, padx=5)
        self.conf_scale = Scale(self.control_frame, from_=0.005, to=0.95, resolution=0.05, orient=tk.HORIZONTAL,
                                length=120)
        self.conf_scale.set(DEFAULT_CONF_THRESHOLD)
        self.conf_scale.pack(side=tk.LEFT, padx=5)
        self.conf_scale.bind("<ButtonRelease-1>", self.reprocess_with_current_settings)

        self.iou_label = tk.Label(self.control_frame, text="IOU Thresh:")
        self.iou_label.pack(side=tk.LEFT, padx=5)
        self.iou_scale = Scale(self.control_frame, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL, length=120)
        self.iou_scale.set(DEFAULT_IOU_THRESHOLD)
        self.iou_scale.pack(side=tk.LEFT, padx=5)
        self.iou_scale.bind("<ButtonRelease-1>", self.reprocess_with_current_settings)

        self.reprocess_button = tk.Button(self.control_frame, text="Reprocess (Current NMS)",
                                          command=self.reprocess_with_current_settings)
        self.reprocess_button.pack(side=tk.LEFT, padx=10)

        self.lighting_effect_button = tk.Button(self.control_frame, text="Next Effect + Detect",
                                                command=self.cycle_lighting_and_reprocess_single_effect)
        self.lighting_effect_button.pack(side=tk.LEFT, padx=10)

        self.deep_scan_button = tk.Button(self.control_frame, text="Perform Deep Scan", command=self.perform_deep_scan)
        self.deep_scan_button.pack(side=tk.LEFT, padx=10)

        self.current_effect_label = tk.Label(self.control_frame, text="Effect: Original")
        self.current_effect_label.pack(side=tk.LEFT, padx=5)

        self.stats_label = tk.Label(self.control_frame, text="Cows Detected: 0")
        self.stats_label.pack(side=tk.RIGHT, padx=10)

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

        self.current_scale = 1.0
        self.tk_img = None
        self.original_cv2_img = None
        self.processed_cv2_img_for_display = None
        self.pil_img_for_display = None
        self.img_x_on_canvas = 0
        self.img_y_on_canvas = 0
        self._last_drag_x = 0
        self._last_drag_y = 0

    def update_effect_label(self):
        if hasattr(self, 'lighting_presets') and self.lighting_presets:
            preset_name = self.lighting_presets[self.current_lighting_preset_index]['name']
            self.current_effect_label.config(text=f"Effect: {preset_name}")
        else:
            self.current_effect_label.config(text="Effect: -")

    def open_image_dialog(self):
        path = filedialog.askopenfilename(title="Select Image File",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not path: return

        self.all_collected_detections = []
        self.is_deep_scan_results_displayed = False
        self.active_image_path = path

        try:
            raw_cv2_img = cv2.imread(path)
            if raw_cv2_img is None:
                messagebox.showerror("Image Load Error", f"Could not load image from path: {path}")
                return
            self.original_cv2_img = raw_cv2_img.copy()
            self.current_lighting_preset_index = 0  # Reset to original effect
            self.update_effect_label()
            self.reprocess_with_current_settings()  # Initial processing with original effect
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the image: {e}")
            self.original_cv2_img = None
            self.processed_cv2_img_for_display = None
            self.display_image()

    def cycle_lighting_and_reprocess_single_effect(self):
        if self.original_cv2_img is None:
            messagebox.showinfo("Information", "Please load an image first.")
            return
        self.current_lighting_preset_index = (self.current_lighting_preset_index + 1) % len(self.lighting_presets)
        self.update_effect_label()
        self.is_deep_scan_results_displayed = False  # Results are for single effect, not deep scan
        self.reprocess_with_current_settings()

    def reprocess_with_current_settings(self, event=None):
        """Reprocesses with the current effect and NMS settings (not accumulating)."""
        if self.original_cv2_img is None:
            if event and not self.active_image_path:  # Called from slider before image load
                return
            elif not self.active_image_path:
                messagebox.showinfo("Information", "No image to reprocess. Please load an image first.")
                return
        try:
            self.is_deep_scan_results_displayed = False  # Not deep scan results
            preset_dict = self.lighting_presets[self.current_lighting_preset_index]
            effect_function = preset_dict['function']

            img_with_effect = effect_function(self.original_cv2_img.copy())

            # Run YOLO on the image with the current effect
            current_detections = self.run_yolo_on_image(img_with_effect)

            # Draw current detections on the image with effect
            self.processed_cv2_img_for_display = self.draw_detections_on_image(
                img_with_effect.copy(),  # Draw on a fresh copy of the effected image
                current_detections,
                is_accumulated=False
            )
            self.stats_label.config(text=f"Cows Detected (Current Effect): {len(current_detections)}")
            self.display_image()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reprocessing the image: {e}")

    def perform_deep_scan(self):
        if self.original_cv2_img is None:
            messagebox.showinfo("Information", "Please load an image first.")
            return

        start_time = time.time()
        self.all_collected_detections = []  # Reset before new deep scan
        self.is_deep_scan_results_displayed = True  # Mark that next results are from deep scan

        original_preset_index_before_scan = self.current_lighting_preset_index  # Save current index

        num_effects = len(self.lighting_presets)
        for i, preset in enumerate(self.lighting_presets):
            self.current_lighting_preset_index = i  # Set current effect for potential UI update
            self.update_effect_label()
            # Update UI to show progress
            progress_text = f"Scanning with effect: {preset['name']} ({i + 1}/{num_effects})"
            self.current_effect_label.config(text=progress_text)
            self.root.update_idletasks()  # Force UI update

            img_to_process = self.original_cv2_img.copy()
            effect_function = preset['function']
            img_with_effect = effect_function(img_to_process)

            current_detections_data = self.run_yolo_on_image(img_with_effect)

            for det_data in current_detections_data:
                self.all_collected_detections.append({
                    "box": det_data["box"],
                    "keypoints": det_data["keypoints"],
                    "conf": det_data["conf"],
                    "effect_name": preset['name'],  # Save effect name
                    "effect_index": i  # Save effect index for unique color
                })

        # Filter accumulated detections using NMS with IOU from slider
        iou_threshold_for_filtering = self.iou_scale.get()
        final_detections_to_draw = self.filter_accumulated_detections(
            self.all_collected_detections,
            iou_thresh=iou_threshold_for_filtering
        )
        print(
            f"Deep scan: {len(self.all_collected_detections)} detections before NMS, {len(final_detections_to_draw)} after NMS with IOU threshold {iou_threshold_for_filtering}")

        # Revert to original effect (or the one before scan) for background display
        self.current_lighting_preset_index = 0  # Display on original image by default
        # self.current_lighting_preset_index = original_preset_index_before_scan # Or revert to previous effect
        self.update_effect_label()

        # Prepare base image on which all accumulated detections will be drawn
        base_img_for_drawing_accumulated = self.lighting_presets[self.current_lighting_preset_index]['function'](
            self.original_cv2_img.copy())

        self.processed_cv2_img_for_display = self.draw_detections_on_image(
            base_img_for_drawing_accumulated,
            final_detections_to_draw,
            is_accumulated=True
        )

        end_time = time.time()
        duration = end_time - start_time
        self.stats_label.config(text=f"Cows Detected (Deep Scan): {len(final_detections_to_draw)}")
        self.display_image()
        messagebox.showinfo("Deep Scan Complete",
                            f"Collected and filtered {len(final_detections_to_draw)} potential detections.\nScan duration: {duration:.2f} seconds.")

    def run_yolo_on_image(self, cv2_img_for_yolo):
        """Runs YOLO on a given image and returns a list of detection data."""
        detections_data_list = []
        original_h, original_w = cv2_img_for_yolo.shape[:2]
        longest_side = max(original_h, original_w)

        conf_threshold = self.conf_scale.get()
        iou_threshold = self.iou_scale.get()  # This IOU is for YOLO's internal NMS per image

        # Assuming class 0 is 'cow' or the target object
        model_args = {"verbose": False, "conf": conf_threshold, "iou": iou_threshold, "classes": [0]}
        if longest_side > THRESHOLD_FOR_LARGE_IMG_SIZE_ACTIVATION:
            model_args["imgsz"] = LARGE_IMG_SIZE

        try:
            results_list = self.model(cv2_img_for_yolo, **model_args)
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            # Potentially show a messagebox or update a status bar if critical
            return detections_data_list

        if not results_list or results_list[0].boxes is None or len(results_list[0].boxes) == 0:
            return detections_data_list

        results = results_list[0]  # Assuming single image processing
        keypoints_data = results.keypoints.data if results.keypoints is not None else []

        for i in range(len(results.boxes.xyxy)):
            box = results.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
            confidence = float(results.boxes.conf[i])
            kpts_list = []
            if results.keypoints is not None and i < len(keypoints_data):
                kpts_tensor = keypoints_data[i]  # Keypoints for the i-th detection
                if hasattr(kpts_tensor, 'reshape'):  # Check if it's a tensor
                    kpts_list = kpts_tensor.reshape(-1, 3).tolist()  # [[x,y,conf], ...]

            detections_data_list.append({
                "box": box, "keypoints": kpts_list, "conf": confidence
            })
        return detections_data_list

    def draw_detections_on_image(self, image_to_draw_on, detections_to_draw, is_accumulated=False):
        annotated_img_local = image_to_draw_on.copy()
        for i, det_data in enumerate(detections_to_draw):
            x1, y1, x2, y2 = map(int, det_data["box"])
            confidence = det_data["conf"]
            kpts_list = det_data.get("keypoints", [])

            current_box_color = BOX_COLOR  # Default box color
            label_text = f"Cow: {confidence:.2f}"  # English label

            if is_accumulated and "effect_name" in det_data:
                label_text += f" ({det_data['effect_name']})"
                # Assign unique color from palette based on effect index
                effect_idx = det_data.get("effect_index", i)
                current_box_color = ACCUMULATED_BOX_COLOR_PALETTE[effect_idx % len(ACCUMULATED_BOX_COLOR_PALETTE)]

            cv2.rectangle(annotated_img_local, (x1, y1), (x2, y2), current_box_color, LINE_THICKNESS)

            # Text label positioning
            (text_width, text_height), baseline = cv2.getTextSize(label_text, FONT_FACE, FONT_SCALE_BOX, LINE_THICKNESS)
            text_y_position = y1 - 10 if y1 - 10 > text_height else y1 + text_height + baseline  # Position above box or below if too close to top

            # Background for text for better readability
            cv2.rectangle(annotated_img_local, (x1, text_y_position - text_height - baseline),
                          (x1 + text_width, text_y_position + baseline), current_box_color, -1)  # Filled rectangle
            cv2.putText(annotated_img_local, label_text, (x1, text_y_position), FONT_FACE, FONT_SCALE_BOX,
                        TEXT_COLOR, LINE_THICKNESS)

            # Draw keypoints
            for kp_idx, kp_data in enumerate(kpts_list):
                if len(kp_data) == 3:  # Ensure three values (x, y, visibility/confidence)
                    x_kp, y_kp, v_kp = kp_data
                    if v_kp > 0:  # Only draw visible/confident keypoints
                        x_kp, y_kp = int(x_kp), int(y_kp)
                        cv2.circle(annotated_img_local, (x_kp, y_kp), CIRCLE_RADIUS, KPT_COLOR, -1)  # Filled circle
                        kp_name = KPT_NAMES[kp_idx] if kp_idx < len(KPT_NAMES) else f"kp_{kp_idx}"
                        cv2.putText(annotated_img_local, kp_name, (x_kp + 4, y_kp - 4), FONT_FACE, FONT_SCALE_KPT,
                                    TEXT_COLOR, 1)
        return annotated_img_local

    def calculate_iou(self, box1, box2):
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
        # box format: [x1, y1, x2, y2]
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
        if union_area == 0: return 0
        return inter_area / union_area

    def filter_accumulated_detections(self, detections_list, iou_thresh):
        """
        Filters a list of accumulated detections to reduce clear duplicates.
        Keeps the detection with the highest confidence in case of high overlap.
        """
        if not detections_list: return []

        # Sort by confidence in descending order (important for NMS logic)
        detections_list.sort(key=lambda d: d['conf'], reverse=True)

        filtered_detections = []
        suppressed = [False] * len(detections_list)  # Keep track of suppressed boxes

        for i in range(len(detections_list)):
            if suppressed[i]:
                continue
            filtered_detections.append(detections_list[i])  # Keep this detection
            # Suppress other detections that significantly overlap with this one
            for j in range(i + 1, len(detections_list)):
                if suppressed[j]:
                    continue
                iou = self.calculate_iou(detections_list[i]['box'], detections_list[j]['box'])
                if iou > iou_thresh:
                    # If high overlap, the detection 'j' (which has lower or equal confidence
                    # because of sorting) is suppressed.
                    suppressed[j] = True

        return filtered_detections

    def display_image(self):
        if self.processed_cv2_img_for_display is None:
            self.canvas.delete("all")  # Clear canvas if no image
            self.tk_img = None
            return

        img_rgb = cv2.cvtColor(self.processed_cv2_img_for_display, cv2.COLOR_BGR2RGB)
        pil_img_original = Image.fromarray(img_rgb)

        new_width = int(pil_img_original.width * self.current_scale)
        new_height = int(pil_img_original.height * self.current_scale)

        if new_width <= 0 or new_height <= 0: return  # Avoid error with zero or negative size
        try:
            # For Pillow >= 9.1.0, use Image.Resampling.LANCZOS
            # For older versions, use Image.LANCZOS
            if hasattr(Image.Resampling, 'LANCZOS'):
                resample_filter = Image.Resampling.LANCZOS
            else:
                resample_filter = Image.LANCZOS  # For older Pillow versions
            resized_pil_img = pil_img_original.resize((new_width, new_height), resample_filter)
        except Exception as e:
            print(f"Error resizing image: {e}")
            return

        self.pil_img_for_display = resized_pil_img  # Keep reference to PIL image
        self.tk_img = ImageTk.PhotoImage(resized_pil_img)  # Convert to Tkinter PhotoImage
        self.canvas.delete("all")  # Clear previous image from canvas

        # Draw the image on the canvas at its current position (img_x_on_canvas, img_y_on_canvas)
        self.canvas.create_image(self.img_x_on_canvas, self.img_y_on_canvas, anchor="nw", image=self.tk_img)
        # self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL)) # This might interfere with scan_dragto

    def zoom_image_event(self, event):
        if self.processed_cv2_img_for_display is None or self.pil_img_for_display is None: return

        # Mouse coordinates on the canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Point on the original image (before zoom) under the mouse
        img_coord_x_before_zoom = (canvas_x - self.img_x_on_canvas) / self.current_scale
        img_coord_y_before_zoom = (canvas_y - self.img_y_on_canvas) / self.current_scale

        old_scale = self.current_scale
        if event.delta > 0:  # Zoom in
            self.current_scale *= ZOOM_FACTOR
        else:  # Zoom out
            self.current_scale /= ZOOM_FACTOR
        self.current_scale = max(MIN_ZOOM, min(self.current_scale, MAX_ZOOM))  # Clamp zoom scale

        if abs(self.current_scale - old_scale) < 1e-9: return  # No significant change in zoom

        # Update top-left corner of the image so the point under mouse stays in the same place
        self.img_x_on_canvas = canvas_x - (img_coord_x_before_zoom * self.current_scale)
        self.img_y_on_canvas = canvas_y - (img_coord_y_before_zoom * self.current_scale)

        self.display_image()

    def on_canvas_press(self, event):
        if self.processed_cv2_img_for_display is None: return
        self.canvas.scan_mark(event.x, event.y)
        self._last_drag_x = event.x  # Store for calculating drag delta
        self._last_drag_y = event.y

    def on_canvas_motion(self, event):
        if self.processed_cv2_img_for_display is None: return
        # Use scan_dragto for canvas dragging
        self.canvas.scan_dragto(event.x, event.y, gain=1)

        # Update self.img_x_on_canvas and self.img_y_on_canvas to reflect the drag
        # Since scan_dragto moves the canvas's "viewpoint",
        # we need to update the coordinates of the image's top-left corner accordingly.
        dx = event.x - self._last_drag_x
        dy = event.y - self._last_drag_y

        # img_x_on_canvas and img_y_on_canvas are the coordinates of the image's top-left corner
        # within the canvas's "world". When the canvas is dragged, these coordinates change.
        self.img_x_on_canvas += dx
        self.img_y_on_canvas += dy

        self._last_drag_x = event.x  # Update last drag position
        self._last_drag_y = event.y
        # No need to call display_image() here as scan_dragto updates the view.
        # However, it's important that img_x/y_on_canvas are updated for future zoom calculations.


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
def canvas_release(self, event):
    """Handle mouse release events."""
    if self.current_image is None:
        return

    if self.moving_keypoint:
        self.moving_keypoint = False
        self.selected_keypoint = None
        self.status_label.config(text="Keypoint moved - ready")

    if self.drawing_bbox and self.manual_mode:
        self.drawing_bbox = False

        # Make sure x1,y1 #!/usr/bin/env python


# -*- coding: utf-8 -*-

"""
YOLOv8 Cow Pose Annotator - Improved Version

Features:
- YOLO model detection
- Manual annotation for images without detections
- Drag and reposition keypoints
- Manual bounding box creation

Installation:
pip install ultralytics opencv-python pillow tk
"""

import os
import glob
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Button, Label, Canvas, Toplevel
from PIL import Image, ImageTk
from ultralytics import YOLO

# Path to the YOLOv8 Pose model weights
MODEL_PATH = r"C:\Users\yehudah\OneDrive - ARO Volcani Center\Desktop\train on other dataset\runs\pose\cow_pose_yolov8m\weights\best.pt"

# Keypoint configuration
KEYPOINT_NAMES = ["left_eye", "right_eye", "center_nose", "left_mouth", "right_mouth"]
# BGR color format for OpenCV
KEYPOINT_COLORS = [
    (0, 0, 255),  # left_eye: Red
    (0, 255, 0),  # right_eye: Green
    (255, 0, 0),  # center_nose: Blue
    (0, 255, 255),  # left_mouth: Yellow
    (255, 0, 255)  # right_mouth: Magenta
]

# Convert to RGB for tkinter
KEYPOINT_COLORS_RGB = [
    "#FF0000",  # left_eye: Red
    "#00FF00",  # right_eye: Green
    "#0000FF",  # center_nose: Blue
    "#FFFF00",  # left_mouth: Yellow
    "#FF00FF"  # right_mouth: Magenta
]


class CowPoseAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Cow Pose Annotator")
        self.root.geometry("1000x700")

        # Initialize variables
        self.model = None
        self.image_files = []
        self.current_image_idx = -1
        self.input_folder = ""
        self.output_folder = ""
        self.current_image = None
        self.results = None
        self.current_image_original = None
        self.display_scale = 1.0

        # Manual annotation variables
        self.manual_mode = False
        self.current_bbox = None  # [x1, y1, x2, y2]
        self.current_keypoints = []  # [(x, y, v), ...]
        self.selected_keypoint = None
        self.drawing_bbox = False
        self.bbox_start = None
        self.selected_tool = None
        self.active_keypoint_idx = -1
        self.moving_keypoint = False
        self.move_start_pos = None

        # Try to load the model
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model from {MODEL_PATH}.\nError: {str(e)}")
            self.root.destroy()
            return

        # Setup UI Components
        self.setup_ui()

        # Bind keyboard shortcuts
        self.root.bind("<Right>", lambda event: self.next_image())  # Right arrow key for next image
        self.root.bind("<Down>", lambda event: self.save_and_next())  # Down arrow key for save and next

    def setup_ui(self):
        # Top toolbar
        top_frame = Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        Button(top_frame, text="Select Images Folder", command=self.select_images_folder).pack(side=tk.LEFT, padx=5)
        Button(top_frame, text="Select Output Folder", command=self.select_output_folder).pack(side=tk.LEFT, padx=5)

        # Toggle between auto and manual modes
        self.manual_mode_btn = Button(top_frame, text="Toggle Manual Mode", command=self.toggle_manual_mode)
        self.manual_mode_btn.pack(side=tk.LEFT, padx=5)

        # Status label to indicate movement mode
        self.status_label = Label(top_frame, text="Movement: Point & Drag to move keypoints", fg="blue")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Main content frame with canvas and tools
        content_frame = Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tool panel (left side)
        self.tool_panel = Frame(content_frame, width=150, bg="#f0f0f0", relief=tk.RAISED, borderwidth=1)
        self.tool_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.tool_panel.pack_propagate(False)  # Prevent shrinking

        # Tool panel title
        Label(self.tool_panel, text="Tools", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=(10, 5))

        # Tool buttons
        self.bbox_btn = Button(self.tool_panel, text="Draw Box", command=lambda: self.select_tool("bbox"))
        self.bbox_btn.pack(fill=tk.X, padx=5, pady=2)

        # Keypoint buttons
        Label(self.tool_panel, text="Keypoints", bg="#f0f0f0").pack(pady=(10, 5))
        self.keypoint_btns = []
        for i, (name, color) in enumerate(zip(KEYPOINT_NAMES, KEYPOINT_COLORS_RGB)):
            btn = Button(self.tool_panel, text=name, bg=color,
                         command=lambda idx=i: self.select_tool(f"keypoint_{idx}"))
            btn.pack(fill=tk.X, padx=5, pady=2)
            self.keypoint_btns.append(btn)

        # Canvas for the image
        self.canvas_frame = Frame(content_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = Canvas(self.canvas_frame, bg="gray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas event bindings
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # Current image info
        self.image_info = Label(self.root, text="No image loaded")
        self.image_info.pack(fill=tk.X, padx=10)

        # Bottom toolbar
        bottom_frame = Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        Button(bottom_frame, text="<< Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        Button(bottom_frame, text="Save", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        Button(bottom_frame, text="Save & Next", command=self.save_and_next).pack(side=tk.LEFT, padx=5)
        Button(bottom_frame, text="Next >>", command=self.next_image).pack(side=tk.LEFT, padx=5)
        Button(bottom_frame, text="Reset", command=self.reset_annotations).pack(side=tk.LEFT, padx=5)

        # Keyboard shortcuts info
        shortcuts_label = Label(bottom_frame, text="Shortcuts: → Next image, ↓ Save & Next")
        shortcuts_label.pack(side=tk.RIGHT, padx=5)

        # Initially hide the tool panel
        self.tool_panel.pack_forget()

    def toggle_manual_mode(self):
        self.manual_mode = not self.manual_mode

        if self.manual_mode:
            self.manual_mode_btn.config(relief=tk.SUNKEN, text="Manual Mode: ON")
            self.tool_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        else:
            self.manual_mode_btn.config(relief=tk.RAISED, text="Manual Mode: OFF")
            self.tool_panel.pack_forget()

        # Redisplay the current image
        self.display_annotated_image()

    def select_tool(self, tool_name):
        """Select a drawing tool."""
        self.selected_tool = tool_name

        # Reset all buttons
        self.bbox_btn.config(relief=tk.RAISED)
        for btn in self.keypoint_btns:
            btn.config(relief=tk.RAISED)

        # Highlight the selected button
        if tool_name == "bbox":
            self.bbox_btn.config(relief=tk.SUNKEN)
        elif tool_name.startswith("keypoint_"):
            idx = int(tool_name.split("_")[1])
            self.keypoint_btns[idx].config(relief=tk.SUNKEN)
            self.active_keypoint_idx = idx

    def canvas_click(self, event):
        """Handle canvas click events."""
        if self.current_image is None:
            return

        # Convert canvas coordinates to original image coordinates
        x, y = self.canvas_to_image_coords(event.x, event.y)

        # First check if we're clicking on an existing keypoint to move it
        if self.check_keypoint_click(event.x, event.y):
            # If we clicked a keypoint, no need to do anything else
            self.status_label.config(text="Keypoint selected - Drag to move")
            return

        # If in manual mode, handle manual annotation tools
        if self.manual_mode:
            if self.selected_tool == "bbox":
                # Start drawing a bounding box
                self.drawing_bbox = True
                self.bbox_start = (x, y)
                self.current_bbox = [x, y, x, y]  # Initial box is just a point
                self.status_label.config(text="Drawing box - Drag to resize")

            elif self.selected_tool and self.selected_tool.startswith("keypoint_"):
                # We need a bounding box first
                if not self.current_bbox:
                    messagebox.showwarning("Warning", "Please draw a bounding box first!")
                    return

                idx = int(self.selected_tool.split("_")[1])

                # Update or add the keypoint
                if idx < len(self.current_keypoints):
                    self.current_keypoints[idx] = (x, y, 2)  # 2 means visible
                else:
                    # Make sure we have placeholders for all preceding keypoints
                    while len(self.current_keypoints) < idx:
                        self.current_keypoints.append((0, 0, 0))  # Add invisible keypoints
                    self.current_keypoints.append((x, y, 2))

                self.status_label.config(text=f"Added keypoint: {KEYPOINT_NAMES[idx]}")

                # Redraw the annotated image
                self.display_annotated_image()

    def check_keypoint_click(self, canvas_x, canvas_y):
        """Check if user clicked on a keypoint and prepare to move it."""
        # Set a larger click radius for better usability
        click_radius = 10  # pixels

        # First check if we have any detections
        if self.results and hasattr(self.results, 'keypoints'):
            # Check existing model keypoints
            for i, keypoints_obj in enumerate(self.results.keypoints.data):
                for j, kpt in enumerate(keypoints_obj[:len(KEYPOINT_NAMES)]):
                    x, y, conf = map(float, kpt)
                    if conf > 0:  # Visible keypoint
                        # Convert to canvas coordinates
                        cx, cy = self.image_to_canvas_coords(x, y)
                        # Check if clicked within keypoint radius
                        if ((canvas_x - cx) ** 2 + (canvas_y - cy) ** 2) <= click_radius ** 2:
                            self.moving_keypoint = True
                            self.move_start_pos = (x, y)
                            self.selected_keypoint = (i, j)  # (detection_idx, keypoint_idx)
                            print(f"Selected keypoint: detection {i}, keypoint {j} at ({x}, {y})")
                            return True

        # Check manual keypoints
        if self.current_keypoints:  # Always check manual keypoints regardless of mode
            for j, kpt in enumerate(self.current_keypoints):
                if j < len(KEYPOINT_NAMES) and len(kpt) >= 3 and kpt[2] > 0:
                    # Convert to canvas coordinates
                    cx, cy = self.image_to_canvas_coords(kpt[0], kpt[1])
                    # Check if clicked within keypoint radius
                    if ((canvas_x - cx) ** 2 + (canvas_y - cy) ** 2) <= click_radius ** 2:
                        self.moving_keypoint = True
                        self.move_start_pos = (kpt[0], kpt[1])
                        self.selected_keypoint = (-1, j)  # (-1 means manual keypoint)
                        print(f"Selected manual keypoint {j} at ({kpt[0]}, {kpt[1]})")
                        return True

        return False

    def canvas_drag(self, event):
        """Handle mouse drag events."""
        if self.current_image is None:
            return

        # Convert canvas coordinates to original image coordinates
        x, y = self.canvas_to_image_coords(event.x, event.y)

        # Ensure coordinates are within image bounds
        h, w = self.current_image.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        if self.moving_keypoint and self.selected_keypoint is not None:
            # Moving a keypoint
            detection_idx, keypoint_idx = self.selected_keypoint

            if detection_idx == -1:
                # Manual keypoint
                if keypoint_idx < len(self.current_keypoints):
                    self.current_keypoints[keypoint_idx] = (x, y, 2)  # Update position
                    print(f"Moving manual keypoint {keypoint_idx} to ({x}, {y})")
            else:
                # Model detected keypoint - make a clone before modifying
                try:
                    # Create a copy of the keypoints tensor for this detection
                    if not hasattr(self, 'keypoints_copy') or self.keypoints_copy is None:
                        self.keypoints_copy = self.results.keypoints.data.clone()

                    # Update the copy
                    self.keypoints_copy[detection_idx][keypoint_idx][0] = x
                    self.keypoints_copy[detection_idx][keypoint_idx][1] = y

                    print(f"Moving detection {detection_idx}, keypoint {keypoint_idx} to ({x}, {y})")
                except Exception as e:
                    print(f"Error updating keypoint: {e}")

            # Redraw
            self.display_annotated_image()

        elif self.drawing_bbox and self.manual_mode:
            # Update the bounding box as we drag
            self.current_bbox[2] = x
            self.current_bbox[3] = y

            # Redraw
            self.display_annotated_image()

    def canvas_release(self, event):
        """Handle mouse release events."""
        if not self.manual_mode:
            return

        if self.moving_keypoint:
            self.moving_keypoint = False
            self.selected_keypoint = None

        if self.drawing_bbox:
            self.drawing_bbox = False

            # Make sure x1,y1 is the top-left corner and x2,y2 is bottom-right
            x1, y1, x2, y2 = self.current_bbox
            self.current_bbox = [
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2)
            ]

            # Reset keypoints when drawing a new box
            self.current_keypoints = []

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to original image coordinates."""
        if self.display_scale == 0:
            return (0, 0)
        return (canvas_x / self.display_scale, canvas_y / self.display_scale)

    def image_to_canvas_coords(self, img_x, img_y):
        """Convert original image coordinates to canvas coordinates."""
        return (img_x * self.display_scale, img_y * self.display_scale)

    def select_images_folder(self):
        """Open a dialog to select the folder containing images."""
        folder = filedialog.askdirectory(title="Select Images Folder")
        if not folder:
            return

        self.input_folder = folder

        # Get all image files
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(folder, ext)))

        # Sort the files for consistent navigation
        self.image_files.sort()

        if not self.image_files:
            messagebox.showwarning("No Images", "No image files found in the selected folder!")
            return

        # Reset index and load first image
        self.current_image_idx = 0
        self.load_current_image()
        messagebox.showinfo("Images Loaded", f"Found {len(self.image_files)} images in the folder.")

    def select_output_folder(self):
        """Open a dialog to select the output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if not folder:
            return

        self.output_folder = folder

        # Create required subfolders
        images_folder = os.path.join(folder, "images")
        labels_folder = os.path.join(folder, "labels")

        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        messagebox.showinfo("Output Folder",
                            f"Output folder set to {folder}\nCreated subfolders for images and labels.")

    def load_current_image(self):
        """Load the current image and run inference."""
        if not self.image_files or self.current_image_idx < 0 or self.current_image_idx >= len(self.image_files):
            return

        image_path = self.image_files[self.current_image_idx]
        filename = os.path.basename(image_path)

        try:
            # Load and display original image
            self.current_image = cv2.imread(image_path)
            self.current_image_original = self.current_image.copy()  # Keep a clean copy

            if self.current_image is None:
                messagebox.showerror("Error", f"Failed to load image: {image_path}")
                return

            # Update image info
            self.image_info.config(text=f"Image {self.current_image_idx + 1}/{len(self.image_files)}: {filename}")

            # Reset manual annotations
            self.current_bbox = None
            self.current_keypoints = []

            # Reset keypoints copy
            self.keypoints_copy = None

            # Run inference if not in manual mode
            self.results = self.model(self.current_image)[0]

            # Draw boxes and keypoints
            self.display_annotated_image()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def display_annotated_image(self):
        """Draw bounding boxes and keypoints on the image and display it."""
        if self.current_image is None:
            return

        # Make a fresh copy of the original image for drawing
        img_display = self.current_image_original.copy()

        # Get image dimensions
        img_height, img_width = img_display.shape[:2]

        # Draw automatic detections if we have results
        if self.results and hasattr(self.results, 'boxes') and len(self.results.boxes) > 0:
            # Process each detected object
            for i, box in enumerate(self.results.boxes):
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw keypoints - use our copy if available, otherwise use original
                keypoints = self.keypoints_copy[i] if hasattr(self,
                                                              'keypoints_copy') and self.keypoints_copy is not None else \
                self.results.keypoints.data[i]

                if keypoints.shape[0] > 0:
                    for j, kpt in enumerate(keypoints[:len(KEYPOINT_NAMES)]):
                        x, y, conf = map(float, kpt)
                        x, y = int(x), int(y)

                        if conf > 0:  # Visible keypoint
                            # Draw circle for the keypoint
                            cv2.circle(img_display, (x, y), 5, KEYPOINT_COLORS[j], -1)

                            # Add keypoint name
                            cv2.putText(img_display, KEYPOINT_NAMES[j], (x + 5, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEYPOINT_COLORS[j], 1)

        # Draw manual annotations if they exist
        if self.current_bbox:
            x1, y1, x2, y2 = map(int, self.current_bbox)
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the keypoints (always show manual keypoints if they exist)
        for j, kpt in enumerate(self.current_keypoints):
            if j < len(KEYPOINT_NAMES) and len(kpt) >= 3 and kpt[2] > 0:
                x, y = map(int, kpt[:2])

                # Draw circle for the keypoint
                cv2.circle(img_display, (x, y), 5, KEYPOINT_COLORS[j], -1)

                # Add keypoint name
                cv2.putText(img_display, KEYPOINT_NAMES[j], (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEYPOINT_COLORS[j], 1)

        # Convert to RGB for PIL
        img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        # Resize if needed to fit the window (keeping aspect ratio)
        display_height, display_width = 600, 800  # Maximum display size
        h, w = img_display_rgb.shape[:2]

        # Calculate resize factor
        scale = min(display_width / w, display_height / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img_display_rgb = cv2.resize(img_display_rgb, (new_w, new_h))
            self.display_scale = scale
        else:
            self.display_scale = 1.0

        # Clear previous canvas content
        self.canvas.delete("all")

        # Convert to PhotoImage for display
        img_pil = Image.fromarray(img_display_rgb)
        self.img_tk = ImageTk.PhotoImage(image=img_pil)

        # Update the canvas with the new image
        self.canvas.config(width=img_pil.width, height=img_pil.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # Update status based on selected action
        if self.moving_keypoint:
            self.status_label.config(text="Moving keypoint - Drag to continue")
        elif self.selected_keypoint:
            self.status_label.config(text="Keypoint selected")
        else:
            self.status_label.config(text="Movement: Point & Drag to move keypoints")

    def reset_annotations(self):
        """Reset all annotations on the current image."""
        if self.manual_mode:
            self.current_bbox = None
            self.current_keypoints = []
        else:
            # In auto mode, just reload the image to get fresh detections
            self.load_current_image()

        self.display_annotated_image()

    def save_annotations(self):
        """Save the annotated image and YOLO format labels."""
        if self.current_image is None or not self.output_folder:
            messagebox.showwarning("Warning", "No image to save or output folder not selected!")
            return

        # Check if we have any annotations to save
        has_auto_annotations = not self.manual_mode and self.results and hasattr(self.results, 'boxes') and len(
            self.results.boxes) > 0
        has_manual_annotations = self.manual_mode and self.current_bbox is not None

        if not (has_auto_annotations or has_manual_annotations):
            messagebox.showwarning("Warning", "No annotations to save!")
            return

        # Get the current image filename without extension
        current_filename = os.path.basename(self.image_files[self.current_image_idx])
        filename_no_ext = os.path.splitext(current_filename)[0]

        try:
            # Save the original image without annotations
            img_output_path = os.path.join(self.output_folder, "images", f"{filename_no_ext}.jpg")

            # Get dimensions for the label format
            img_height, img_width = self.current_image.shape[:2]

            # Save the original image (no annotations)
            cv2.imwrite(img_output_path, self.current_image_original)

            # Save YOLO format labels
            label_output_path = os.path.join(self.output_folder, "labels", f"{filename_no_ext}.txt")

            with open(label_output_path, 'w') as f:
                if has_auto_annotations:
                    # Save automatic detections
                    for i, box in enumerate(self.results.boxes):
                        # Get normalized bounding box coordinates (center_x, center_y, width, height)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Convert to YOLO format (normalized)
                        center_x = (x1 + x2) / (2 * img_width)
                        center_y = (y1 + y2) / (2 * img_height)
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        # Start the line with class_id (0 for cow_face)
                        line = f"0 {center_x:.5f} {center_y:.5f} {width:.5f} {height:.5f}"

                        # Use our updated keypoints if available, otherwise use the original
                        keypoints = self.keypoints_copy[i] if hasattr(self,
                                                                      'keypoints_copy') and self.keypoints_copy is not None else \
                        self.results.keypoints.data[i]

                        # Add keypoints in format: x y v (visibility 0=not labeled, 1=labeled but not visible, 2=labeled and visible)
                        for kpt in keypoints[:len(KEYPOINT_NAMES)]:
                            x, y, conf = map(float, kpt)

                            # Normalize coordinates
                            x_norm = x / img_width
                            y_norm = y / img_height

                            # Set visibility value (v): 2 if detected, 0 if not
                            v = 2 if conf > 0 else 0

                            line += f" {x_norm:.5f} {y_norm:.5f} {int(v)}"

                        f.write(line + "\n")

                elif has_manual_annotations:
                    # Save manual annotations
                    x1, y1, x2, y2 = self.current_bbox

                    # Convert to YOLO format (normalized)
                    center_x = (x1 + x2) / (2 * img_width)
                    center_y = (y1 + y2) / (2 * img_height)
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # Start the line with class_id (0 for cow_face)
                    line = f"0 {center_x:.5f} {center_y:.5f} {width:.5f} {height:.5f}"

                    # Add keypoints
                    for j in range(len(KEYPOINT_NAMES)):
                        if j < len(self.current_keypoints) and len(self.current_keypoints[j]) >= 3:
                            x, y, v = self.current_keypoints[j]

                            # Normalize coordinates
                            x_norm = x / img_width
                            y_norm = y / img_height

                            line += f" {x_norm:.5f} {y_norm:.5f} {int(v)}"
                        else:
                            # Add placeholder for missing keypoint
                            line += f" 0.00000 0.00000 0"

                    f.write(line + "\n")

            messagebox.showinfo("Success", "Annotations saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

    def save_and_next(self):
        """Save annotations and move to the next image."""
        self.save_annotations()
        self.next_image()

    def next_image(self):
        """Load the next image."""
        if not self.image_files:
            return

        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.load_current_image()
        else:
            messagebox.showinfo("End of Dataset", "You've reached the last image in the folder.")

    def prev_image(self):
        """Load the previous image."""
        if not self.image_files:
            return

        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_current_image()
        else:
            messagebox.showinfo("Beginning of Dataset", "You're at the first image in the folder.")


if __name__ == "__main__":
    # Add PyTorch import for tensor operations when moving keypoints
    try:
        import torch
    except ImportError:
        messagebox.showerror("Missing Dependency", "PyTorch is required. Please install it with:\npip install torch")
        exit(1)

    root = tk.Tk()
    app = CowPoseAnnotator(root)
    root.mainloop()
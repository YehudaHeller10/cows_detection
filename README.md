Cow Pose / Station Detection
=============================

Overview
--------
A compact and practical toolkit for detecting cow face bounding boxes and facial keypoints (pose), assigning detections to fixed "station" regions, and exporting results as CSV. Built around Ultralytics YOLOv8 (pose models), OpenCV for visualization and image processing, and lightweight Tkinter GUIs and CLI tools for inspection and batch extraction.

What this repo contains
-----------------------
- Training scripts: train.py, train_fine_tuning.py, train_stronger_model.py
- Inference / utilities:
  - Cow_location_extractor*.py — multiprocessing CLI to run a YOLO model over a folder and save CSV results
  - myprog_with_cow_num.py / myProg5-deep_search_colors.py — GUI tools (Tkinter) for interactive inspection, lighting-augmented deep searches and station numbering
  - my_prog_to_test_model*.py — quick test harnesses
- Data configs: data.yaml, data_for_fine_tuned.yaml (kpt names & shapes, dataset paths)
- Models: yolov8{n,m,l}-pose.pt and fine-tuned weights under runs/pose/
- Utilities: CSV with station bounding boxes (COW_STATION_LOCATIONS_PART1.csv)

Key capabilities
----------------
- Pose detection (5 keypoints): left_eye, right_eye, center_nose, left_mouth, right_mouth
- Automatic assignment of detected cows to pre-defined station bounding boxes with configurable margin
- Batch extraction with multiprocessing and an output CSV per input folder
- Interactive GUI for visual inspection, dynamic NMS tuning, lighting-effect augmentation and deep-aggregate detection across multiple effects
- Fine-tuning scripts for adapting pretrained pose models to small / custom datasets

Quick start
-----------
1) Install (recommended in a virtualenv):
   pip install -r requirements.txt
   - If no requirements.txt, at least install: ultralytics, torch (matching CUDA if available), opencv-python, numpy, pandas, pillow, tqdm

2) Train or fine-tune:
   - Train from scratch / baseline:
     python train.py
   - Fine-tune existing best weights:
     python train_fine_tuning.py
   - Use a heavier model for better accuracy (more resources):
     python train_stronger_model.py

3) Run batch extraction (CLI):
   - Start the multiprocessing extractor and save CSV:
     python Cow_location_extractor_cli.py
   - The script will ask for: stations CSV path, model (.pt) path, and images folder path.

4) Inspect images interactively:
   - Launch the GUI and open images to inspect detections, toggle lighting presets, run deep scan and visualize aggregated detections:
     python myprog_with_cow_num.py
     or
     python myProg5-deep_search_colors.py

Important configuration
-----------------------
- data.yaml / data_for_fine_tuned.yaml: set correct 'path', 'train' and 'val' locations and keypoint info before training
- Station CSV: must contain label_name, bbox_x, bbox_y, bbox_width, bbox_height
- MODEL_PATH constants in GUI scripts point to the weights file to use for inference
- Device: set device='cuda' where possible for speed; code falls back to CPU automatically if CUDA unavailable

Performance & tuning notes
--------------------------
- Use GPU (CUDA) for both training and inference. The CLI limits to 1 worker when using CUDA to avoid multiple processes loading the GPU concurrently.
- Adjust conf / iou sliders in the GUI, and conf/iou args in the CLI for desired precision/recall balance
- The deep-aggregate search runs the model multiple times with different image-effects to find missed detections; use filtering IoU to reduce duplicates
- For large images, the code increases inference resolution (imgsz) to improve small-keypoint detection

Troubleshooting
---------------
- Model loading errors: check ultralytics / torch versions and that the .pt file exists
- No detections: verify the model was trained for "pose" (keypoint-enabled) and that class indices match (class 0 expected)
- Memory errors: reduce batch size, use smaller model (yolov8n/s), or enable cache=False where applicable

Project maintenance
-------------------
- Keep weights under runs/pose/ and update MODEL_PATH constants in GUIs when you replace best.pt
- Add or regenerate station CSV if camera setup or barn layout changes

License
-------
Include your preferred license file (e.g., MIT) in the repo root. Models and code are otherwise provided without warranty—follow licenses for pretrained model weights you use.

Contact / Notes
----------------
- Designed for rapid iteration: train -> fine-tune -> test -> batch-extract
- If you want, I can add: example commands for exporting visualized outputs, a requirements.txt, or a short notebook demonstrating training logs and mAP/keypoint metrics.

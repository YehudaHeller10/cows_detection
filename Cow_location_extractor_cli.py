import os
import sys
import csv
import gc
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm

# ----------------
# Configuration
# ----------------
KEYPOINT_NAMES = ["left_eye", "right_eye", "center_nose", "left_mouth", "right_mouth"]
LARGE_IMG_SIZE = 1280
RESIZE_THRESHOLD = 800

# ----------------
# Prompt user input
# ----------------
stations_path = input("Enter path to station CSV file: ").strip()
model_path = input("Enter path to YOLO model (.pt): ").strip()
images_path = input("Enter path to image folder: ").strip()

# ----------------
# Infer output file name from folder
# ----------------
def infer_output_name(images_path):
    comps = os.path.normpath(images_path).split(os.sep)
    if len(comps) >= 2:
        base = comps[-2] + "_" + comps[-1]
    else:
        base = comps[-1]
    return f"output_{base}.csv"

output_path = infer_output_name(images_path)

# ----------------
# Helper functions
# ----------------
def load_station_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    stations = []
    for _, row in df.iterrows():
        x1 = float(row['bbox_x'])
        y1 = float(row['bbox_y'])
        x2 = x1 + float(row['bbox_width'])
        y2 = y1 + float(row['bbox_height'])
        stations.append({
            "station_number": int(row['label_name']),
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2)/2, (y1 + y2)/2)
        })
    return stations

def assign_station(cow_box, stations, margin=100):
    x1, y1, x2, y2 = cow_box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    best = 'unidentified'
    mind = float('inf')
    for st in stations:
        sx1, sy1, sx2, sy2 = (st['bbox'][0]-margin,
                              st['bbox'][1]-margin,
                              st['bbox'][2]+margin,
                              st['bbox'][3]+margin)
        if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
            dist = np.hypot(cx-st['center'][0], cy-st['center'][1])
            if dist < mind:
                mind = dist
                best = st['station_number']
    return best

# ----------------
# Globals for worker processes
# ----------------
model = None
stations = None
device = 'cpu'

# ----------------
# Worker initializer and function
# ----------------
def init_worker(model_path, stations_path, dev):
    global model, stations, device
    import torch
    from ultralytics import YOLO
    device = dev
    model = YOLO(model_path)
    model.to(device)
    model.model.eval()
    torch.set_grad_enabled(False)
    stations = load_station_csv(stations_path)

def process_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return []

        h, w = img.shape[:2]
        pred_kwargs = dict(source=img, device=device, conf=0.15, iou=0.45, classes=[0], verbose=False)
        if max(h, w) > RESIZE_THRESHOLD:
            pred_kwargs["imgsz"] = LARGE_IMG_SIZE

        with torch.inference_mode():
            results = model.predict(**pred_kwargs)

        det = results[0]
        boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else []
        kps_obj = getattr(det, 'keypoints', None)
        kps = kps_obj.data.cpu().numpy() if (kps_obj is not None and getattr(kps_obj, 'data', None) is not None) else []

        rows = []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = map(int, b)
            station_num = assign_station((x1, y1, x2, y2), stations)
            if station_num == 'unidentified':
                continue

            row = {
                'image_name': os.path.basename(path),
                'station_number': station_num,
                'cow_face_x1': x1, 'cow_face_y1': y1,
                'cow_face_x2': x2, 'cow_face_y2': y2
            }

            kpt = kps[i] if i < len(kps) else None
            for j, nm in enumerate(KEYPOINT_NAMES):
                if kpt is not None and len(kpt) > j and kpt[j][2] > 0:
                    row[f"{nm}_x"] = int(kpt[j][0])
                    row[f"{nm}_y"] = int(kpt[j][1])
                else:
                    row[f"{nm}_x"] = ''
                    row[f"{nm}_y"] = ''
            rows.append(row)

            # שחרור נקודות מפתח מיידית
            del kpt

        # שחרור אובייקטים גדולים מהזיכרון
        del img, results, det, boxes, kps_obj, kps
        torch.cuda.empty_cache()
        gc.collect()

        return rows

    except Exception as e:
        print(f"Error processing {path}: {e}", file=sys.stderr)
        return []


# ----------------
# Main
# ----------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA available: {torch.cuda.is_available()}, using device: {device}")

    image_files = []
    for r, _, fs in os.walk(images_path):
        for f in fs:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                image_files.append(os.path.join(r, f))

    print(f"Found {len(image_files)} images to process.")

    header = ['image_name','station_number','cow_face_x1','cow_face_y1','cow_face_x2','cow_face_y2']
    for k in KEYPOINT_NAMES:
        header += [f"{k}_x", f"{k}_y"]
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    f_out = open(output_path, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=header)
    writer.writeheader()

    workers = min(cpu_count(), len(image_files))
    ctx = get_context('spawn') if device == 'cuda' else get_context()
    if device == 'cuda':
        workers = min(workers, 1)

    print(f"Starting processing with {workers} worker(s) using start method '{ctx.get_start_method()}'...")
    with ctx.Pool(workers, initializer=init_worker,
                  initargs=(model_path, stations_path, device)) as pool:
        for rows in tqdm(pool.imap_unordered(process_image, image_files, chunksize=8), total=len(image_files)):
            for r in rows:
                writer.writerow(r)
            f_out.flush()

    f_out.close()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Processing complete. Results saved to {output_path}")

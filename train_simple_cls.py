import argparse
import json
import os
import requests
from PIL import Image
from ultralytics import YOLO
import numpy as np

# ---------------- CONFIG ----------------
OUTPUT_DIR = "outputs"
IMAGE_PATH = os.path.join(OUTPUT_DIR, "satellite.png")
JSON_PATH = os.path.join(OUTPUT_DIR, "result.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
def fetch_satellite_image(lat, lon, zoom=18, size=512):
    """
    Fetch satellite image using OpenStreetMap tiles (NO API KEY)
    """
    print("[1] Fetching satellite image...")

    url = (
        f"https://static-maps.yandex.ru/1.x/"
        f"?ll={lon},{lat}&z={zoom}&l=sat&size={size},{size}"
    )

    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError("Failed to download satellite image")

    with open(IMAGE_PATH, "wb") as f:
        f.write(r.content)

    return IMAGE_PATH


def estimate_area(image_size_px=512):
    """
    Simple heuristic:
    Assume 1 pixel ≈ 0.25 m² (demo assumption)
    """
    total_area = image_size_px * image_size_px * 0.25
    usable_area = total_area * 0.35  # assume 35% usable rooftop
    return round(usable_area, 2)


def main(lat, lon):
    # STEP 1: IMAGE FETCH
    image_path = fetch_satellite_image(lat, lon)

    # STEP 2: LOAD MODEL (SAFE)
    print("[2] Loading model...")
    model = YOLO("yolov8n-cls.pt")  # OFFICIAL pretrained model

    # STEP 3: INFERENCE
    print("[3] Running inference...")
    results = model(image_path, verbose=False)

    probs = results[0].probs
    top_class = results[0].names[int(probs.top1)]
    confidence = float(probs.top1conf)

    # STEP 4: AREA ESTIMATION
    usable_area_m2 = estimate_area()

    # STEP 5: BUFFER LOGIC
    buffer_applied = "2400_sqft" if usable_area_m2 * 10.76 > 1200 else "1200_sqft"

    # STEP 6: QC STATUS
    qc_status = "VERIFIABLE" if confidence > 0.6 else "NOT_VERIFIABLE"

    # STEP 7: STRUCTURED OUTPUT
    output = {
        "input": {
            "latitude": lat,
            "longitude": lon
        },
        "model": {
            "type": "YOLOv8 Classification",
            "top_class": top_class,
            "confidence": round(confidence, 3)
        },
        "solar_estimation": {
            "usable_area_m2": usable_area_m2,
            "buffer_zone": buffer_applied
        },
        "qc_status": qc_status,
        "artifacts": {
            "satellite_image": IMAGE_PATH
        }
    }

    with open(JSON_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("[✓] Pipeline completed successfully")
    print(f"[✓] Output saved to {JSON_PATH}")


# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    main(args.lat, args.lon)

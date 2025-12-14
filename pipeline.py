import argparse
import json
import requests
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "models/solar_cls.pt"
OUTPUT_JSON = "result.json"


def fetch_satellite_image(lat, lon):
    print("[1] Fetching satellite image (ESRI – free)")

    delta = 0.001
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export"
        f"?bbox={bbox}"
        "&bboxSR=4326"
        "&imageSR=4326"
        "&size=640,640"
        "&format=png"
        "&f=image"
    )

    out_path = "satellite.png"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print("[✓] Satellite image downloaded")
        return out_path
    except Exception:
        raise RuntimeError("Satellite image fetch failed (check internet)")


def estimate_area(label):
    if label.lower() == "suitable":
        return {
            "usable_roof_area_m2": 60,
            "buffer_m2": 30,
            "final_area_m2": 30
        }
    else:
        return {
            "usable_roof_area_m2": 0,
            "buffer_m2": 0,
            "final_area_m2": 0
        }


def main(lat, lon):
    img_path = fetch_satellite_image(lat, lon)

    print("[2] Loading YOLO model")
    model = YOLO(MODEL_PATH)

    print("[3] Running classification")
    result = model(img_path)[0]

    probs = result.probs
    cls_id = probs.top1
    confidence = float(probs.top1conf)
    label = model.names[cls_id]

    area_info = estimate_area(label)

    qc_status = "VERIFIABLE" if confidence >= 0.6 else "NOT_VERIFIABLE"

    output = {
        "latitude": lat,
        "longitude": lon,
        "prediction": label,
        "confidence": round(confidence, 3),
        "qc_status": qc_status,
        "area_estimation": area_info,
        "model": "YOLOv8 Classification",
        "imagery_source": "ESRI World Imagery (Free)",
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=4)

    print("[✓] Pipeline completed")
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    main(args.lat, args.lon)

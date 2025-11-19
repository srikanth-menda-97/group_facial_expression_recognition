import os, csv, json
import pandas as pd

RES_DIR = "results"
OUT = os.path.join(RES_DIR, "ground_truth.csv")
PRED_FILES = [
    os.path.join(RES_DIR, "group_emotions_baseline.csv"),
    os.path.join(RES_DIR, "group_emotions_confidence.csv"),  # or group_emotions_conf.csv if that's your filename
    os.path.join(RES_DIR, "group_emotions_spatial.csv"),
    os.path.join(RES_DIR, "group_emotions_hybrid.csv"),
]

def collect_images():
    imgs = set()
    for p in PRED_FILES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "image" in df.columns:
                imgs.update(df["image"].astype(str).tolist())
    return sorted(imgs)

def main():
    os.makedirs(RES_DIR, exist_ok=True)
    images = collect_images()
    if not images:
        print("No prediction CSVs found; falling back to data/group_images/")
        IMG_DIR = "data/group_images"
        if os.path.exists(IMG_DIR):
            images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
    rows = [{"image": im, "gt_label": ""} for im in images]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote template {OUT} with {len(rows)} rows — fill gt_label and re-run evaluation.")

if __name__ == "__main__":
    main()

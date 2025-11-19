import os, json, math
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

import os as _os
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

IMG_DIR = "data/group_images"
OUT_CSV = "results/group_emotions_spatial.csv"
ALPHA = 0.5  # blend between confidence and spatial (0..1)

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def analyze_face(face_rgb):
    res = DeepFace.analyze(img_path=face_rgb,
                           actions=["emotion"],
                           enforce_detection=False,
                           detector_backend="skip")
    if isinstance(res, list):
        res = res[0]
    emo = res["emotion"]
    probs = {k: float(emo[k]) for k in emo}
    s = sum(probs.values()) or 1.0
    probs = {k: v / s for k, v in probs.items()}
    dom = max(probs, key=probs.get)
    conf = probs[dom]
    return probs, conf

def spatial_score(x,y,w,h,W,H):
    area = (w*h) / max(1.0, W*H)
    cx = x + w/2.0; cy = y + h/2.0
    cW = W/2.0; cH = H/2.0
    dist = math.hypot(cx - cW, cy - cH)
    max_dist = math.hypot(cW, cH)
    center = 1.0 - min(1.0, dist / max_dist)
    return 0.5*area + 0.5*center  # [0..1]

def aggregate(scores):
    agg = {e: 0.0 for e in EMOTIONS}
    for probs, weight in scores:
        for e in EMOTIONS:
            agg[e] += weight * probs.get(e, 0.0)
    total = sum(agg.values()) or 1.0
    agg = {e: v/total for e, v in agg.items()}
    label = max(agg, key=agg.get)
    return label, agg

def process_image(path, detector):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return {"image": os.path.basename(path), "n_faces": 0, "group_label": "load_error", "group_scores": {}}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H,W = img_rgb.shape[:2]
    faces = detector.detect_faces(img_rgb)
    scored = []
    for f in faces:
        x,y,w,h = f["box"]
        x = max(0,x); y = max(0,y); w = max(1,w); h = max(1,h)
        x2 = min(W, x+w); y2 = min(H, y+h)
        crop = img_rgb[y:y2, x:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (224,224))
        try:
            probs, conf = analyze_face(crop)
        except Exception:
            continue
        spatial = spatial_score(x,y,w,h,W,H)
        weight = ALPHA*conf + (1.0-ALPHA)*spatial
        scored.append((probs, weight))
    if not scored:
        return {"image": os.path.basename(path), "n_faces": 0, "group_label": "no_face", "group_scores": {}}
    label, agg = aggregate(scored)
    return {"image": os.path.basename(path), "n_faces": len(scored), "group_label": label, "group_scores": agg}

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    detector = MTCNN()
    images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","n_faces","group_label","group_scores_json","alpha"])
        for p in images:
            rec = process_image(p, detector)
            w.writerow([rec["image"], rec["n_faces"], rec["group_label"], json.dumps(rec["group_scores"]), ALPHA])
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()

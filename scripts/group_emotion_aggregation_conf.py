import os, json
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

# Optional: help with Keras 3 environments
import os as _os
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

IMG_DIR = "data/group_images"
OUT_CSV = "results/group_emotions_conf.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def analyze_face_conf(face_rgb):
    # DeepFace returns scores as dict, sometimes wrapped in list
    res = DeepFace.analyze(img_path=face_rgb,
                           actions=["emotion"],
                           enforce_detection=False,
                           detector_backend="skip")
    if isinstance(res, list):  # take first result if list
        res = res[0]
    emo = res["emotion"]
    # Convert to probabilities in [0,1]
    probs = {k: float(emo[k]) for k in emo}
    s = sum(probs.values()) or 1.0
    probs = {k: v / s for k, v in probs.items()}
    dom = max(probs, key=probs.get)
    conf = probs[dom]
    return probs, dom, conf

def aggregate_conf(scores_and_conf):
    agg = {e: 0.0 for e in EMOTIONS}
    for probs, _, conf in scores_and_conf:
        for e in EMOTIONS:
            agg[e] += conf * probs.get(e, 0.0)
    # Normalize
    total = sum(agg.values()) or 1.0
    agg = {e: v/total for e, v in agg.items()}
    label = max(agg, key=agg.get)
    return label, agg

def process_image(path, detector):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return {"image": os.path.basename(path), "n_faces": 0, "group_label": "load_error", "group_scores": {}}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    scores = []
    for f in faces:
        x,y,w,h = f["box"]
        # clamp to image bounds
        H,W = img_rgb.shape[:2]
        x = max(0,x); y = max(0,y); w = max(1,w); h = max(1,h)
        x2 = min(W, x+w); y2 = min(H, y+h)
        crop = img_rgb[y:y2, x:x2]
        if crop.size == 0:
            continue
        # resize for stability
        crop = cv2.resize(crop, (224,224))
        try:
            probs, dom, conf = analyze_face_conf(crop)
            scores.append((probs, dom, conf))
        except Exception as e:
            # skip problematic faces
            continue
    if not scores:
        return {"image": os.path.basename(path), "n_faces": 0, "group_label": "no_face", "group_scores": {}}
    label, agg = aggregate_conf(scores)
    return {"image": os.path.basename(path), "n_faces": len(scores), "group_label": label, "group_scores": agg}

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    detector = MTCNN()
    images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","n_faces","group_label","group_scores_json"])
        for p in images:
            rec = process_image(p, detector)
            w.writerow([rec["image"], rec["n_faces"], rec["group_label"], json.dumps(rec["group_scores"])])
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()

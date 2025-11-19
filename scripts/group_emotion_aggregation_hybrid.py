import os, json, csv
from collections import defaultdict
import cv2
from aggregation_utils import (
    EMOTIONS, get_detector, read_rgb, detect_faces, face_crop,
    deepface_emotion_probs, spatial_prominence,
    agg_majority, agg_confidence, agg_spatial, agg_hybrid
)

IMG_DIR = "data/group_images"
RES_DIR = "results"
ALPHA = 0.5  # blend factor for hybrid

os.makedirs(RES_DIR, exist_ok=True)

def process_image(path, detector):
    img_rgb = read_rgb(path)
    if img_rgb is None:
        return {"image": os.path.basename(path), "n_faces": 0, "status": "load_error"}
    H, W = img_rgb.shape[:2]
    faces = detect_faces(img_rgb, detector)
    per_face_probs, confs, spatials = [], [], []

    for f in faces:
        box = f["box"]
        crop = face_crop(img_rgb, box)
        if crop is None:
            continue
        try:
            probs, dom, conf = deepface_emotion_probs(crop)
        except Exception:
            continue
        per_face_probs.append(probs)
        confs.append(conf)
        spatials.append(spatial_prominence((box[0], box[1], box[2], box[3]), W, H))

    if not per_face_probs:
        return {
            "image": os.path.basename(path),
            "n_faces": 0,
            "status": "no_face"
        }

    # Compute all aggregations
    label_base, agg_base = agg_majority(per_face_probs)
    label_conf, agg_conf = agg_confidence(per_face_probs, confs)
    label_spat, agg_spat = agg_spatial(per_face_probs, spatials)
    label_hyb, agg_hyb = agg_hybrid(per_face_probs, confs, spatials, alpha=ALPHA)

    return {
        "image": os.path.basename(path),
        "n_faces": len(per_face_probs),
        "status": "ok",
        "baseline": {"label": label_base, "scores": agg_base},
        "confidence": {"label": label_conf, "scores": agg_conf},
        "spatial": {"label": label_spat, "scores": agg_spat},
        "hybrid": {"label": label_hyb, "scores": agg_hyb, "alpha": ALPHA},
    }

def write_csv(path, rows, method):
    out = os.path.join(RES_DIR, f"group_emotions_{method}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_faces", "group_label", "group_scores_json"] + (["alpha"] if method=="hybrid" else []))
        for r in rows:
            data = r[method]
            extras = [r["hybrid"]["alpha"]] if method == "hybrid" else []
            w.writerow([r["image"], r["n_faces"], data["label"], json.dumps(data["scores"])] + extras)
    print(f"Wrote {out}")

def main():
    det = get_detector()
    images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    results = []
    for p in images:
        rec = process_image(p, det)
        if rec.get("status") == "ok":
            results.append(rec)
    if not results:
        print("No results; check input images.")
        return
    for m in ["baseline", "confidence", "spatial", "hybrid"]:
        write_csv(IMG_DIR, results, m)

if __name__ == "__main__":
    main()

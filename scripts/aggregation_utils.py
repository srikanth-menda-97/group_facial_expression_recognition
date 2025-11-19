import os
import cv2
import math
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

# Optional: improve compatibility when using TF 2.15 + legacy Keras backends
import os as _os
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def get_detector():
    return MTCNN()

def read_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def clamp_box(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y); w = max(1, w); h = max(1, h)
    x2 = min(W, x + w); y2 = min(H, y + h)
    return x, y, x2, y2

def detect_faces(img_rgb, detector):
    # Returns MTCNN face dicts with 'box' key
    return detector.detect_faces(img_rgb)

def face_crop(img_rgb, box, size=(224, 224)):
    x, y, w, h = box
    H, W = img_rgb.shape[:2]
    x, y, x2, y2 = clamp_box(x, y, w, h, W, H)
    crop = img_rgb[y:y2, x:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, size)

def deepface_emotion_probs(face_rgb):
    # Accepts an RGB array; use detector_backend="skip" to avoid redetection
    res = DeepFace.analyze(img_path=face_rgb,
                           actions=["emotion"],
                           enforce_detection=False,
                           detector_backend="skip")
    if isinstance(res, list):
        res = res[0]
    emo = res["emotion"]
    # Convert to normalized probabilities
    raw = {k.lower(): float(emo[k]) for k in emo}
    s = sum(raw.values()) or 1.0
    probs = {k: raw.get(k, 0.0) / s for k in EMOTIONS}
    dom = max(probs, key=probs.get)
    conf = probs[dom]
    return probs, dom, conf

def spatial_prominence(box, frame_w, frame_h):
    x, y, w, h = box
    # Area score
    area = (w * h) / max(1.0, frame_w * frame_h)
    # Center proximity score
    cx, cy = x + w / 2.0, y + h / 2.0
    cW, cH = frame_w / 2.0, frame_h / 2.0
    dist = math.hypot(cx - cW, cy - cH)
    max_dist = math.hypot(cW, cH)
    center = 1.0 - min(1.0, dist / max_dist)
    return 0.5 * area + 0.5 * center

def agg_majority(per_face_probs):
    # Equal weights for all faces (baseline)
    agg = {e: 0.0 for e in EMOTIONS}
    for probs in per_face_probs:
        for e in EMOTIONS:
            agg[e] += probs.get(e, 0.0)
    total = sum(agg.values()) or 1.0
    agg = {e: v / total for e, v in agg.items()}
    label = max(agg, key=agg.get)
    return label, agg

def agg_confidence(per_face_probs, confs):
    agg = {e: 0.0 for e in EMOTIONS}
    for probs, c in zip(per_face_probs, confs):
        for e in EMOTIONS:
            agg[e] += c * probs.get(e, 0.0)
    total = sum(agg.values()) or 1.0
    agg = {e: v / total for e, v in agg.items()}
    label = max(agg, key=agg.get)
    return label, agg

def agg_spatial(per_face_probs, spatial_weights):
    agg = {e: 0.0 for e in EMOTIONS}
    for probs, s in zip(per_face_probs, spatial_weights):
        for e in EMOTIONS:
            agg[e] += s * probs.get(e, 0.0)
    total = sum(agg.values()) or 1.0
    agg = {e: v / total for e, v in agg.items()}
    label = max(agg, key=agg.get)
    return label, agg

def agg_hybrid(per_face_probs, confs, spatial_weights, alpha=0.5):
    # weight = alpha*confidence + (1-alpha)*spatial
    agg = {e: 0.0 for e in EMOTIONS}
    for probs, c, s in zip(per_face_probs, confs, spatial_weights):
        w = alpha * c + (1.0 - alpha) * s
        for e in EMOTIONS:
            agg[e] += w * probs.get(e, 0.0)
    total = sum(agg.values()) or 1.0
    agg = {e: v / total for e, v in agg.items()}
    label = max(agg, key=agg.get)
    return label, agg

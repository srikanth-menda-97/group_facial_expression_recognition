import os, json, time
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

import os as _os
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]
LAMBDA = 0.8  # smoothing factor

def face_probs(face_rgb):
    res = DeepFace.analyze(img_path=face_rgb,
                           actions=["emotion"],
                           enforce_detection=False,
                           detector_backend="skip")
    if isinstance(res, list):
        res = res[0]
    emo = res["emotion"]
    probs = {k: float(emo[k]) for k in emo}
    s = sum(probs.values()) or 1.0
    return {k: v/s for k, v in probs.items()}

def group_probs(frame_rgb, detector):
    faces = detector.detect_faces(frame_rgb)
    if not faces:
        return {e: 0.0 for e in EMOTIONS}, []
    agg = {e: 0.0 for e in EMOTIONS}
    boxes = []
    for f in faces:
        x,y,w,h = f["box"]
        H,W = frame_rgb.shape[:2]
        x = max(0,x); y = max(0,y); w = max(1,w); h = max(1,h)
        x2 = min(W, x+w); y2 = min(H, y+h)
        crop = frame_rgb[y:y2, x:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (224,224))
        try:
            p = face_probs(crop)
        except Exception:
            continue
        for e in EMOTIONS:
            agg[e] += p.get(e,0.0)
        boxes.append((x,y,w,h,max(p, key=p.get)))
    total = sum(agg.values()) or 1.0
    agg = {e: v/total for e, v in agg.items()}
    return agg, boxes

def smooth(prev, curr):
    if prev is None:
        return curr
    return {e: LAMBDA*prev.get(e,0.0) + (1.0-LAMBDA)*curr.get(e,0.0) for e in EMOTIONS}

def main():
    detector = MTCNN()
    cap = cv2.VideoCapture(0)  # use integer for webcam or path for video file
    if not cap.isOpened():
        print("Could not open webcam; try giving a video path to VideoCapture.")
        return
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gp, boxes = group_probs(frame_rgb, detector)
        prev = smooth(prev, gp)
        label = max(prev, key=prev.get)
        # draw
        for (x,y,w,h,fe) in boxes:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, fe, (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, f"Group: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow("Realtime Group Emotion", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

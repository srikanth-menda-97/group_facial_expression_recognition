import time
import numpy as np
import cv2
from aggregation_utils import (
    EMOTIONS, get_detector, detect_faces, face_crop,
    deepface_emotion_probs, spatial_prominence, agg_hybrid
)

LAMBDA = 0.8  # EMA smoothing factor
ALPHA = 0.5   # hybrid blend factor

def draw_group_prob_strip(frame, group_scores, origin=(10, 50), width=240, height=120):
    # Draw horizontal bars for each emotion
    x0, y0 = origin
    bar_h = int(height / len(EMOTIONS))
    for i, e in enumerate(EMOTIONS):
        val = float(group_scores.get(e, 0.0))
        w = int(val * width)
        y = y0 + i * bar_h
        cv2.rectangle(frame, (x0, y), (x0 + width, y + bar_h - 2), (60,60,60), 1)
        cv2.rectangle(frame, (x0, y), (x0 + w, y + bar_h - 2), (0, 140, 255), -1)
        cv2.putText(frame, f"{e}: {val:.2f}", (x0 + width + 10, y + bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def ema(prev, curr, lam=LAMBDA):
    if prev is None:
        return curr
    return {e: lam * prev.get(e, 0.0) + (1 - lam) * curr.get(e, 0.0) for e in EMOTIONS}

def main():
    det = get_detector()
    cap = cv2.VideoCapture(0)  # change to a file path for video
    if not cap.isOpened():
        print("Could not open webcam/video.")
        return

    smoothed = None
    t_prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame_rgb.shape[:2]
        faces = detect_faces(frame_rgb, det)

        per_face_probs, confs, spatials = [], [], []
        for f in faces:
            x,y,w,h = f["box"]
            crop = face_crop(frame_rgb, (x,y,w,h))
            if crop is None:
                continue
            try:
                probs, dom, conf = deepface_emotion_probs(crop)
            except Exception:
                continue
            per_face_probs.append(probs)
            confs.append(conf)
            spatials.append(spatial_prominence((x,y,w,h), W, H))
            # Draw face box + dominant label
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, dom, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if per_face_probs:
            label, scores = agg_hybrid(per_face_probs, confs, spatials, alpha=ALPHA)
        else:
            label, scores = "no_face", {e: 0.0 for e in EMOTIONS}

        smoothed = ema(smoothed, scores, LAMBDA)
        group_label = max(smoothed, key=smoothed.get) if smoothed else label

        # Overlays: title, group label, probability strip, FPS
        cv2.putText(frame, f"Group: {group_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        draw_group_prob_strip(frame, smoothed, origin=(10, 60))
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.imshow("Realtime Group Emotion (Hybrid + EMA)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

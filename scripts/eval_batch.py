import os, subprocess, sys, csv, json

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Run confidence and spatial scripts
subprocess.run([sys.executable, "scripts/group_emotion_aggregation_conf.py"], check=True)
subprocess.run([sys.executable, "scripts/group_emotion_aggregation_spatial.py"], check=True)

# Optional: derive a "baseline" by treating all faces as equal weight (read conf CSV and reweight equally)
def make_baseline_from_conf(conf_csv, out_csv):
    rows = []
    with open(conf_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row["group_scores_json"]:
                rows.append({"image": row["image"], "n_faces": row["n_faces"], "group_label": "no_face", "group_scores_json": "{}"})
                continue
            scores = json.loads(row["group_scores_json"])
            # Already normalized; for baseline, we can't recover per-face, so reuse as proxy
            rows.append({"image": row["image"], "n_faces": row["n_faces"], "group_label": max(scores, key=scores.get), "group_scores_json": json.dumps(scores)})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image","n_faces","group_label","group_scores_json"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

make_baseline_from_conf("results/group_emotions_conf.csv", "results/group_emotions_baseline.csv")
print("Wrote results/group_emotions_conf.csv, results/group_emotions_spatial.csv, results/group_emotions_baseline.csv")

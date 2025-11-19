import os, json, csv
import pandas as pd

RES_DIR = "results"
BASE = os.path.join(RES_DIR, "group_emotions_baseline.csv")
CONF = os.path.join(RES_DIR, "group_emotions_confidence.csv")  # will create below
SPAT = os.path.join(RES_DIR, "group_emotions_spatial.csv")
HYBR = os.path.join(RES_DIR, "group_emotions_hybrid.csv")
GT   = os.path.join(RES_DIR, "ground_truth.csv")  # optional: image,gt_label

def rename_if_needed():
    # Harmonize method filenames from hybrid runner if needed
    mapping = {
        os.path.join(RES_DIR, "group_emotions_baseline.csv"): BASE,
        os.path.join(RES_DIR, "group_emotions_conf.csv"): CONF,
        os.path.join(RES_DIR, "group_emotions_spatial.csv"): SPAT,
        os.path.join(RES_DIR, "group_emotions_hybrid.csv"): HYBR,
    }
    for src, dst in mapping.items():
        if os.path.exists(src) and src != dst:
            os.replace(src, dst)

def load_labels(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["image","group_label"])
    df = pd.read_csv(path)
    df = df[["image","group_label"]].copy()
    df["image"] = df["image"].astype(str)
    df["group_label"] = df["group_label"].astype(str).str.strip().str.lower()
    return df

def agreement_rate(a, b):
    merged = a.merge(b, on="image", suffixes=("_a","_b"))
    if merged.empty:
        return 0.0, 0
    agree = (merged["group_label_a"] == merged["group_label_b"]).sum()
    return agree / len(merged), len(merged)

def accuracy(pred, gt):
    merged = pred.merge(gt, on="image")
    if merged.empty:
        return None, 0
    acc = (merged["group_label"] == merged["gt_label"].str.strip().str.lower()).mean()
    return acc, len(merged)

def main():
    rename_if_needed()
    base = load_labels(BASE)
    conf = load_labels(CONF)
    spat = load_labels(SPAT)
    hybr = load_labels(HYBR)
    gt   = None
    if os.path.exists(GT):
        gt = pd.read_csv(GT)
        gt["image"] = gt["image"].astype(str)
        gt["gt_label"] = gt["gt_label"].astype(str).str.strip().str.lower()

    # Agreement between methods
    pairs = [
        ("baseline","confidence", base, conf),
        ("baseline","spatial", base, spat),
        ("baseline","hybrid", base, hybr),
        ("confidence","spatial", conf, spat),
        ("confidence","hybrid", conf, hybr),
        ("spatial","hybrid", spat, hybr),
    ]
    rows = []
    for n1, n2, d1, d2 in pairs:
        rate, n = agreement_rate(d1, d2)
        rows.append({"pair": f"{n1} vs {n2}", "agreement": rate, "n": n})

    # Optional accuracy (if ground-truth provided)
    acc_rows = []
    if gt is not None:
        for name, df in [("baseline", base), ("confidence", conf), ("spatial", spat), ("hybrid", hybr)]:
            acc, n = accuracy(df, gt)
            if acc is not None:
                acc_rows.append({"method": name, "accuracy": acc, "n": n})

    os.makedirs(RES_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, "method_agreement.csv"), index=False)
    if acc_rows:
        pd.DataFrame(acc_rows).to_csv(os.path.join(RES_DIR, "method_accuracy.csv"), index=False)
    print("Wrote results/method_agreement.csv and (if GT provided) results/method_accuracy.csv")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RES_DIR = "results"
AGREE_CSV = os.path.join(RES_DIR, "method_agreement.csv")   # columns: pair,agreement,n
ACC_CSV   = os.path.join(RES_DIR, "method_accuracy.csv")    # columns: method,accuracy,n  (optional)

os.makedirs(RES_DIR, exist_ok=True)

def _add_vertical_bar_labels(ax, values, containers, fmt="{:.1f}%"):
    # Places vertical percentage labels above each bar using the true values and a small padding. 
    for j, cont in enumerate(containers):
        for i, bar in enumerate(cont):
            if i >= len(values[j]):
                continue
            v = values[j][i]
            ax.annotate(fmt.format(v),
                        (bar.get_x() + bar.get_width()/2.0, bar.get_height()),
                        ha="center", va="bottom",
                        rotation=90, rotation_mode="anchor",
                        fontsize=8, xytext=(0, 2), textcoords="offset points")

def plot_agreement():
    if not os.path.exists(AGREE_CSV):
        print(f"Missing {AGREE_CSV} — run evaluation first.")
        return None
    df = pd.read_csv(AGREE_CSV)
    if df.empty:
        print("Agreement CSV is empty.")
        return None

    # Prepare data
    df["pair"] = df["pair"].astype(str)
    df = df.sort_values("pair").reset_index(drop=True)
    labels = df["pair"].tolist()
    perc = (df["agreement"].astype(float) * 100.0).tolist()
    ns   = df["n"].astype(int).tolist()

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, perc, color="#4C78A8")
    ax.set_title("Method Agreement (pairwise)") 
    ax.set_ylabel("Agreement (%)")
    ax.set_xlabel("Method pairs")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylim(0, 100)  # full 0–100% scale
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Vertical labels above bars and sample sizes below x-ticks
    for bar, p, n in zip(bars, perc, ns):
        ax.annotate(f"{p:.1f}%",
                    (bar.get_x() + bar.get_width()/2.0, bar.get_height()),
                    ha="center", va="bottom",
                    rotation=90, rotation_mode="anchor",
                    fontsize=8, xytext=(0, 2), textcoords="offset points")
    # Add N under ticks
    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([f"N={n}" for n in ns], rotation=30, ha="right")
    ax2.tick_params(length=0)
    plt.tight_layout()

    out_path = os.path.join(RES_DIR, "method_agreement_plot.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    return df

def plot_accuracy():
    if not os.path.exists(ACC_CSV):
        print(f"No accuracy CSV found at {ACC_CSV} — skipping accuracy plot.")
        return None
    df = pd.read_csv(ACC_CSV)
    if df.empty:
        print("Accuracy CSV is empty.")
        return None

    df["method"] = df["method"].astype(str)
    df = df.sort_values("method").reset_index(drop=True)
    labels = df["method"].tolist()
    perc = (df["accuracy"].astype(float) * 100.0).tolist()
    ns   = df["n"].astype(int).tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, perc, color="#F58518")
    ax.set_title("Method Accuracy (with ground truth)") 
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Method")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, p in zip(bars, perc):
        ax.annotate(f"{p:.1f}%",
                    (bar.get_x() + bar.get_width()/2.0, bar.get_height()),
                    ha="center", va="bottom",
                    rotation=90, rotation_mode="anchor",
                    fontsize=8, xytext=(0, 2), textcoords="offset points")
    # Show N beneath each bar
    for bar, n in zip(bars, ns):
        ax.annotate(f"N={n}",
                    (bar.get_x() + bar.get_width()/2.0, 0),
                    ha="center", va="top", fontsize=8, xytext=(0, 14),
                    textcoords="offset points", rotation=90)
    plt.tight_layout()

    out_path = os.path.join(RES_DIR, "method_accuracy_plot.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    return df

def plot_agreement_matrix(agree_df):
    # Build symmetric matrix from pairwise agreement rates
    # Methods set inferred from pair names like "baseline vs confidence"
    methods = set()
    for pair in agree_df["pair"]:
        a, b = [s.strip() for s in pair.split("vs")]
        methods.add(a); methods.add(b)
    methods = sorted(methods)
    idx = {m:i for i,m in enumerate(methods)}
    mat = np.eye(len(methods), dtype=float)

    for _, row in agree_df.iterrows():
        a, b = [s.strip() for s in row["pair"].split("vs")]
        r = float(row["agreement"])
        i, j = idx[a], idx[b]
        mat[i, j] = r
        mat[j, i] = r

    # Plot heatmap with numeric annotations (percentage)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat * 100.0, vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(len(methods))); ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticklabels(methods)
    ax.set_title("Pairwise Agreement Matrix (%)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("%", rotation=90, va="bottom")
    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{mat[i,j]*100:.1f}",
                    ha="center", va="center", color=("black" if mat[i,j] < 0.7 else "white"), fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RES_DIR, "method_agreement_matrix.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    agree_df = plot_agreement()
    acc_df = plot_accuracy()
    if agree_df is not None and not agree_df.empty:
        plot_agreement_matrix(agree_df)

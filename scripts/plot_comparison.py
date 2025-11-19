import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE_CSV = "results/group_emotions_baseline.csv"
CONF_CSV = "results/group_emotions_conf.csv"
SPAT_CSV = "results/group_emotions_spatial.csv"

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def label_distribution(csv_path, method_name):
    if not os.path.exists(csv_path):
        return pd.Series(index=EMOTIONS, data=0.0, name=method_name, dtype=float)
    df = pd.read_csv(csv_path)
    if "group_label" not in df.columns or df.empty:
        return pd.Series(index=EMOTIONS, data=0.0, name=method_name, dtype=float)
    labels = df["group_label"].astype(str).str.strip().str.lower()
    counts = labels.value_counts(dropna=False).reindex(EMOTIONS, fill_value=0)
    perc = (counts / counts.sum() * 100.0) if counts.sum() > 0 else counts.astype(float)
    perc.name = method_name
    return perc

series_list = []
if os.path.exists(BASE_CSV):
    series_list.append(label_distribution(BASE_CSV, "baseline"))
series_list.append(label_distribution(CONF_CSV, "confidence"))
series_list.append(label_distribution(SPAT_CSV, "spatial"))

dist_df = pd.concat(series_list, axis=1).fillna(0.0)

if dist_df.empty:
    print("No data to plot. Run the aggregation scripts to generate CSVs first.")
else:
    # No epsilon floor: bars reflect true percentages (0 stays 0)
    ax = dist_df.plot(kind="bar", rot=45, figsize=(10,6))
    ax.set_title("Group Label Distribution by Aggregation Method")
    ax.set_ylabel("Percentage of images (%)")
    ax.set_xlabel("Group label")
    ax.legend(title="Method")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Put vertical percentage labels ABOVE each bar using true values
    # For zero-height bars, use a small pixel offset so text sits above the baseline
    for j, container in enumerate(ax.containers):
        true_vals = [dist_df.iloc[i, j] for i in range(len(dist_df.index))]
        labels = [f"{v:.1f}%" for v in true_vals]
        # bar_label with label_type='edge' places text at the bar's top; padding is in points
        # For zero-height bars, bar_label anchors at y=0, so the padding lifts the text above the baseline
        ax.bar_label(container, labels=labels, label_type='edge',
                     padding=2, rotation=90, fontsize=8)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/aggregation_method_comparison.png", dpi=200)
    plt.show()

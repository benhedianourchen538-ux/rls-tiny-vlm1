"""Plot S1: images/s vs batch size for every throughput_*.csv in this folder (mean +- std over repeats)."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent


def main():
    series = defaultdict(lambda: defaultdict(list))          # (device, mode) -> batch -> [images/s]
    for path in sorted(HERE.glob("throughput_*.csv")):
        for r in csv.DictReader(open(path)):
            series[(r["device"], r["mode"])][int(r["batch_size"])].append(float(r["images_per_s"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for (dev, mode), by_bs in series.items():
        bs = sorted(by_bs)
        mean = np.array([np.mean(by_bs[b]) for b in bs]); std = np.array([np.std(by_bs[b]) for b in bs])
        ax.errorbar(bs, mean, yerr=std, marker="o", capsize=3, label=f"{dev} / {mode}")
    ax.set(xscale="log", yscale="log", xlabel="batch size", ylabel="training throughput (images/s)",
           title="S1: training throughput vs batch size")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(HERE / "throughput.png", dpi=150)
    print("saved", HERE / "throughput.png")


if __name__ == "__main__":
    main()

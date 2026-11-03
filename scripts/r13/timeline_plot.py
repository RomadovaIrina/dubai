#!/usr/bin/env python3
"""r13 round 3: GPU timeline of e2e jobs: NVML device util (10 Hz, 2-s mean) + every job's stage bands. Single job vs a worker queue.
    python scripts/r13/timeline_plot.py --out fig.png --panel "single old03" NVML.jsonl JOBDIR [JOBDIR ...] --panel "queue" NVML.jsonl JOBDIR ..."""
import argparse, json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"precheck": "#bbbbbb", "wait_heavy": "#eeeeee", "audio": "#4c78a8", "sep": "#72b7b2", "face_gen": "#e45756", "wait_comp": "#eeeeee",
       "face_comp": "#f58518", "subs": "#b279a2", "mux": "#54a24b", "metrics": "#bbbbbb", "upright": "#bbbbbb", "polyglot": "#9d755d", "pace_fix": "#9d755d"}


def stages(d):
    rows = [l.rstrip("\n").split("\t") for l in open(pathlib.Path(d) / "work" / "stages.tsv") if "\t" in l]
    return [(n, float(a), float(b)) for (n, a), (_, b) in zip(rows, rows[1:])]


def main():
    argv = sys.argv[1:]; out = argv[argv.index("--out") + 1]; panels = []
    i = 0
    while i < len(argv):
        if argv[i] == "--panel":
            j = i + 1; name = argv[j]; nv = argv[j + 1]; dirs = []
            k = j + 2
            while k < len(argv) and not argv[k].startswith("--"): dirs.append(argv[k]); k += 1
            panels.append((name, nv, dirs)); i = k
        else: i += 1
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3.2 * len(panels)), squeeze=False)
    for ax, (name, nv, dirs) in zip(axes[:, 0], panels):
        S = [json.loads(l) for l in open(nv) if l.strip()]; t = np.array([s["t"] for s in S]); u = np.array([float(s.get("util", 0)) for s in S])
        st = {pathlib.Path(d).name: stages(d) for d in dirs}; t0 = min(a for v in st.values() for _, a, _ in v); t1 = max(b for v in st.values() for _, _, b in v)
        m = (t >= t0) & (t <= t1); tt, uu = t[m] - t0, u[m]; k = 20; us = np.convolve(uu, np.ones(k) / k, "same") if len(uu) > k else uu
        ax.fill_between(tt, 0, us, color="#333333", alpha=0.35, lw=0); ax.plot(tt, us, color="#222222", lw=0.6)
        for r, (job, ss) in enumerate(st.items()):
            for n, a, b in ss:
                ax.add_patch(plt.Rectangle((a - t0, 102 + r * 9), b - a, 8, color=COL.get(n, "#999999"), lw=0))
                if b - a > 25 and n in ("audio", "face_gen", "face_comp"): ax.text((a + b) / 2 - t0, 106 + r * 9, n, ha="center", va="center", fontsize=6.5, color="white")
            ax.text(-5, 106 + r * 9, job, ha="right", va="center", fontsize=8)
        busy = float(np.mean(uu > 10)) if len(uu) else 0
        ax.set_title(f"{name}: {tt.max():.0f} s wall, GPU busy (util > 10 %) {100 * busy:.0f} % of the wall", fontsize=10, loc="left")
        ax.set_ylim(0, 104 + 9 * len(st)); ax.set_xlim(-60, tt.max() + 5); ax.set_ylabel("GPU util %"); ax.set_yticks([0, 50, 100])
    axes[-1, 0].set_xlabel("seconds")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("audio", "sep", "face_gen", "face_comp", "mux")]
    fig.legend(handles, ["audio", "sep", "face_gen (LatentSync)", "face_comp (composite, CPU-heavy)", "mux"], loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, dpi=110); print("saved", out)


if __name__ == "__main__":
    main()

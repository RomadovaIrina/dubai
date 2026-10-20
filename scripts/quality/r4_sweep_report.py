#!/usr/bin/env python3
"""r4 generator sweep report: per arm the mean over the windows that ALL arms finished, the per-window delta vs A0_base, and the seed spread
(A0 / A5_seed7 / A6_seed99 = the same settings, different noise) as the noise floor: an arm "beats seed noise" only if its mean delta exceeds the
seed spread of that metric AND it wins on >= 2/3 of the windows.
    python scripts/quality/r4_sweep_report.py --dir /workspace/dub/artifacts/r4sweep/win --out reports/experiments/r4-audio/generator_sweep.md"""
import argparse, collections, json, pathlib
import numpy as np
ARMS = ["A0_base", "A5_seed7", "A6_seed99", "A1_g15", "A2_g25", "A3_s30", "A4_nobr"]
DESC = {"A0_base": "v4 / R3: guidance 2.0, 20 steps, bridge 8x4, seed 1247", "A5_seed7": "seed 7", "A6_seed99": "seed 99", "A1_g15": "guidance 1.5",
        "A2_g25": "guidance 2.5", "A3_s30": "30 steps", "A4_nobr": "bridge off"}
# metric, higher-is-better
MET = [("conf_at0_speech_mean", True), ("conf_at0_worst_s", True), ("sync_conf", True), ("lip_r", True), ("flash", False), ("jaw_ratio", False), ("s_per_ls_frame", False)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    R = collections.defaultdict(dict)
    for f in pathlib.Path(a.dir).glob("*.metrics.json"):
        m = json.loads(f.read_text()); arm = m["tag"].rsplit("_", 1)[0]; R[arm][(m["id"], int(m["t0"]))] = m
    arms = [x for x in ARMS if x in R]; wins = sorted(set.intersection(*[set(R[x]) for x in arms])) if arms else []
    seed_arms = [x for x in ("A0_base", "A5_seed7", "A6_seed99") if x in arms]
    def mean(arm, k): v = [R[arm][w][k] for w in wins if R[arm][w].get(k) is not None]; return float(np.mean(v)) if v else None
    spread = {k: (max(mean(s, k) for s in seed_arms) - min(mean(s, k) for s in seed_arms)) for k, _ in MET if all(mean(s, k) is not None for s in seed_arms)}
    L = ["# r4 LatentSync generator sweep on the worst lip-sync windows (audio v1)", "",
         f"Windows ({len(wins)}, 8 s each, worst per-second dub-only SyncNet conf_at0 of a full v4-generator run on audio v1): " + ", ".join(f"{i} {t}-{t + 8} s" for i, t in wins), "",
         "Generator code = v4 (904c5d2: A R3 generator, fp32 AlignRestore, flow:4 anchors, head mask) via `ls_window_ab.py`; one setting changed per arm.",
         "Metrics: conf_at0 = SyncNet confidence at offset 0 per second over seconds where the dub speaks (mean / worst second); sync_conf = window SyncNet;",
         "lip_r = inner-lip gap vs driving audio log-RMS (Pearson); flash = single-frame lip spikes vs the master (r3_flash); jaw = jaw-contour jitter / master",
         "(1.0 = as steady as the source); s/frame = LatentSync seconds per generated frame (GPU shared with other jobs part of the time: indicative).", "",
         "| arm | setting | " + " | ".join(k for k, _ in MET) + " | wins vs base (conf_at0) |", "|---|---|" + "---:|" * len(MET) + "---:|"]
    for arm in arms:
        w = sum(1 for x in wins if (R[arm][x].get("conf_at0_speech_mean") or 0) > (R["A0_base"][x].get("conf_at0_speech_mean") or 0)) if arm != "A0_base" else None
        L.append(f"| {arm} | {DESC[arm]} | " + " | ".join("" if mean(arm, k) is None else f"{mean(arm, k):.3f}" for k, _ in MET) + f" | {'' if w is None else f'{w}/{len(wins)}'} |")
    L += ["", "Seed spread (max - min over the three seeds, same settings) = the noise floor: " + ", ".join(f"{k} {v:.3f}" for k, v in spread.items()), "",
          "## Per window: conf_at0 (speech mean) per arm", "", "| window | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for x in wins: L.append(f"| {x[0]} {x[1]}-{x[1] + 8} s | " + " | ".join(f"{R[arm][x].get('conf_at0_speech_mean') or float('nan'):.2f}" for arm in arms) + " |")
    verdict = []
    for arm in arms:
        if arm in seed_arms: continue
        d = (mean(arm, "conf_at0_speech_mean") or 0) - (mean("A0_base", "conf_at0_speech_mean") or 0)
        w = sum(1 for x in wins if (R[arm][x].get("conf_at0_speech_mean") or 0) > (R["A0_base"][x].get("conf_at0_speech_mean") or 0))
        beats = d > spread.get("conf_at0_speech_mean", 0) and w >= 2 * len(wins) / 3
        verdict.append((d, arm, w, beats))
    L += ["", "## Verdict (conf_at0 speech mean vs base, noise floor = seed spread)", ""]
    for d, arm, w, beats in sorted(verdict, reverse=True):
        L.append(f"- {arm} ({DESC[arm]}): {d:+.3f} vs base, wins {w}/{len(wins)} -> {'BEATS seed noise' if beats else 'within seed noise'}")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()

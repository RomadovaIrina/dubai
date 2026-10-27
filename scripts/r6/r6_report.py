#!/usr/bin/env python3
"""r6 G: window-sweep table (per arm, over the windows every arm has finished) from <win>/<id>_<arm>_<t0>.metrics.json + .chin.json + .log.
    python scripts/r6/r6_report.py [--win DIR] [--base G0_base] [--noise G10_seed7] > table.md"""
import argparse, json, pathlib, re, collections
import numpy as np

COLS = [("conf_at0_speech_mean", "SyncNet dub speech", 1), ("conf_at0_worst_s", "worst s", 1), ("lip_r", "lip_r", 1), ("flash", "flash (sum)", -1),
        ("twitch_sum", "twitch sum", -1), ("twitch_n", "twitch ev", -1), ("chin_rigid", "chin wave rigid", -1), ("mismatch", "mouth/dub s (sum)", -1),
        ("jaw_ratio", "jaw ratio", 0), ("s_per_ls_frame", "s/LS frame", -1), ("peak", "peak MiB", -1)]
SUM = {"flash", "twitch_sum", "twitch_n", "mismatch"}


def load(win):
    rows = collections.defaultdict(dict)
    for m in win.glob("*.metrics.json"):
        b = m.name[:-len(".metrics.json")]
        if b.endswith(".metrics.tmp"): continue
        id_, rest = b.split("_", 1); arm, t0 = rest.rsplit("_", 1); r = json.loads(m.read_text())
        c = win / f"{b}.chin.json"; ch = json.loads(c.read_text()) if c.exists() else {}
        lg = win / f"{b}.log"; pk = re.search(r"peak (\d+) MiB", lg.read_text()) if lg.exists() else None
        mt = win / "mt" / f"{id_}_{t0}.json"
        d = {"conf_at0_speech_mean": r.get("conf_at0_speech_mean"), "conf_at0_worst_s": r.get("conf_at0_worst_s"), "lip_r": r.get("lip_r"), "flash": r.get("flash"),
             "twitch_sum": r.get("twitch_twitch_sum"), "twitch_n": r.get("twitch_n_events"), "chin_rigid": (ch.get("chin_wave_rigid") or {}).get("mean"),
             "mismatch": r.get("mouth_audio_mismatch_s"), "jaw_ratio": r.get("jaw_ratio"), "s_per_ls_frame": r.get("s_per_ls_frame"), "peak": int(pk.group(1)) if pk else None}
        if arm.startswith("MT") and mt.exists():
            j = json.loads(mt.read_text()); d["s_per_ls_frame"] = j["s_per_frame"]; d["peak"] = j["peak_vram_mib"]
        rows[arm][f"{id_} {t0}"] = d
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--win", default="/workspace/dub/artifacts/r6g/win"); ap.add_argument("--base", default="G0_base")
    ap.add_argument("--noise", default="G10_seed7"); a = ap.parse_args(); rows = load(pathlib.Path(a.win))
    common = set.intersection(*[set(v) for v in rows.values()]) if rows else set()
    wins = sorted(common, key=lambda w: (w.split()[0], int(w.split()[1])))
    print(f"Windows ({len(wins)}, 8 s): " + ", ".join(f"{w.split()[0]} {w.split()[1]}-{int(w.split()[1]) + 8} s" for w in wins) + "\n")

    def agg(arm, k):
        v = [rows[arm][w][k] for w in wins if rows[arm][w].get(k) is not None]
        if not v: return None
        return float(np.sum(v)) if k in SUM else float(np.mean(v))
    base = rows.get(a.base)
    print("| arm | " + " | ".join(c[1] for c in COLS) + " | sync wins vs base |"); print("|---|" + "---:|" * (len(COLS) + 1))
    order = sorted(rows, key=lambda r: (r != a.base, r))
    for arm in order:
        cells = []
        for k, _, _ in COLS:
            x = agg(arm, k); cells.append("" if x is None else (f"{x:.3f}" if abs(x) < 10 else f"{x:.0f}"))
        w = ""
        if base and arm != a.base:
            n = sum(1 for ww in wins if (rows[arm][ww]["conf_at0_speech_mean"] or -9) > (base[ww]["conf_at0_speech_mean"] or -9)); w = f"{n}/{len(wins)}"
        print(f"| {arm} | " + " | ".join(cells) + f" | {w} |")
    if a.noise in rows and base:
        print("\nSeed noise floor (|seed 7 - seed 1247|, same settings): " + ", ".join(f"{c[1]} {abs((agg(a.noise, c[0]) or 0) - (agg(a.base, c[0]) or 0)):.3f}" for c in COLS[:8]))
    print("\n## Per window: SyncNet dub speech (conf_at0 mean)\n\n| window | " + " | ".join(order) + " |\n|---|" + "---:|" * len(order))
    for w in wins:
        print(f"| {w} | " + " | ".join(f"{rows[r][w]['conf_at0_speech_mean']:.2f}" if rows[r][w]['conf_at0_speech_mean'] is not None else "nan" for r in order) + " |")


if __name__ == "__main__":
    main()

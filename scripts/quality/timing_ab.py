#!/usr/bin/env python3
"""Offline timing-policy A/B (Agent C): for every policy, recut_offline.py on the SAME raw TTS (no Qwen / TTS / GPU) -> dub_gaps.py (residual
source-speaks / dub-silent gaps, LS frames with the reference run's LatentSync mask) -> pause_rhythm.py (weak-boundary pauses) -> one table row.

    python scripts/quality/timing_ab.py --manifest BASE/03_final.manifest.json --out ROOT/ab --policies ref sp_c sp_p ...
Policies are defined in POLICIES (name -> recut_offline flags on top of the baseline cut rule).
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
BASE = ["--cut", "acoustic", "--long-pause-func", "1.2", "--cut-fade", "adaptive", "--background", "off"]   # cand5 cut rule; no bed (HDEMUCS = NO)
POLICIES = {
    "ref": [],                                                     # cand5: fill floor 0.9, one tail gap
    "f100": ["--fill-slowdown", "1.0"],                            # no slow-down: what layer 2 at 0.9 buys today
    "lead08": ["--max-lead", "0.8"],
    "flow02": ["--flow-gap", "0.2"],                               # source pauses < 0.2 s (VAD splits inside one breath) are not cut points
    "sp_c": ["--spread", "centre"],
    "sp_p": ["--spread", "pauses"],
    "sp_p_f100": ["--spread", "pauses", "--fill-slowdown", "1.0"],
    "flow02_sp_p": ["--flow-gap", "0.2", "--spread", "pauses"],
    "flow02_sp_p_f0875": ["--flow-gap", "0.2", "--spread", "pauses", "--fill-slowdown", "0.875"],
    "flow02_sp_p_f085": ["--flow-gap", "0.2", "--spread", "pauses", "--fill-slowdown", "0.85"],
    "sp_p_f085": ["--spread", "pauses", "--fill-slowdown", "0.85"],
    "f085": ["--fill-slowdown", "0.85"],
    "f0875": ["--fill-slowdown", "0.875"],
    "flow03_sp_p": ["--flow-gap", "0.3", "--spread", "pauses"],
    "flow02_sp_p30": ["--flow-gap", "0.2", "--spread", "pauses", "--spread-pause-max", "0.3", "--spread-head-max", "0.25"],
    "flow02_sp_p20": ["--flow-gap", "0.2", "--spread", "pauses", "--spread-pause-max", "0.2", "--spread-head-max", "0.15"],
}


def row(name: str, gaps: dict, pz: dict, man: dict) -> dict:
    gg = gaps["gaps_gt"]; sp = (man.get("alignment") or {}).get("spread") or {}
    return {"policy": name, "src_wo_dub_s": gaps["source_without_dub_s"], "iou": gaps["vad_iou"], "dub_wo_src_s": gaps["dub_without_source_s"],
            "gt03": f"{gg['0.3']['n']} / {gg['0.3']['s']}", "gt05": f"{gg['0.5']['n']} / {gg['0.5']['s']}", "gt10": f"{gg['1.0']['n']} / {gg['1.0']['s']}",
            "ls_rms": gaps["ls_frames_dub_silent_while_orig_rms"]["fixed"], "ls_vad": gaps["ls_frames_dub_silent_while_orig_vad"]["fixed"],
            "ls_vad_gt03": gg["0.3"]["ls_frames"], "weak_gt03": pz["weak_boundary_pauses_gt_0_3"], "weak_s": pz["weak_boundary_pause_s"],
            "dub_pauses_gt05": pz["dub"].get("gt_0_5"), "atempo_max": gaps["atempo"]["max"], "atempo_min": gaps["atempo"]["min"],
            "lt09": gaps["atempo"]["share_lt_0_9"], "lt1": gaps["atempo"]["share_lt_1"], "gt115": gaps["atempo"]["share_gt_1_15"],
            "spread_parts": sp.get("parts_spread", 0), "spread_pauses": sp.get("pauses", 0), "spread_pause_s": sp.get("pause_s", 0), "spread_head_s": sp.get("head_s", 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); vid = pathlib.Path(man["input"]).stem; root = pathlib.Path(a.out); rows = []
    for name in a.policies:
        o = root / f"{vid}_{name}"; py = sys.executable
        r = subprocess.run([py, str(HERE / "recut_offline.py"), "--manifest", a.manifest, "--out", str(o), *BASE, *POLICIES[name]], capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:]); return 1
        sline = [l for l in r.stdout.splitlines() if l.startswith("spread")]
        rm = o / "recut.manifest.json"
        subprocess.run([py, str(HERE / "dub_gaps.py"), "--manifest", str(rm), "--ref-manifest", a.manifest, "--tag", name, "--out", str(root / "gaps")], check=True, capture_output=True)
        subprocess.run([py, str(HERE / "pause_rhythm.py"), "--manifest", str(rm), "--tag", name, "--out", str(root / "pauses")], check=True, capture_output=True)
        gaps = json.loads((root / "gaps" / f"{vid}_{name}.gaps.json").read_text()); pz = json.loads((root / "pauses" / f"{vid}_{name}.pauses.json").read_text())
        rows.append(row(name, gaps, pz, json.loads(rm.read_text())))
        print(f"{vid} {name:20s} src_wo_dub {gaps['source_without_dub_s']:5.2f} >0.3 {rows[-1]['gt03']:>10s} >0.5 {rows[-1]['gt05']:>10s} >1.0 {rows[-1]['gt10']:>9s} "
              f"LS rms/vad {rows[-1]['ls_rms']}/{rows[-1]['ls_vad']} weak {rows[-1]['weak_gt03']} atempo {rows[-1]['atempo_min']}-{rows[-1]['atempo_max']} {' '.join(sline)}", flush=True)
    (root / f"{vid}_ab.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

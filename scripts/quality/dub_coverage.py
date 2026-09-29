#!/usr/bin/env python3
"""Dubbed-speech coverage of the original speech (Phase A/B audio alignment metrics).

For one run (<run-dir>/<id>_<tag>.manifest.json + its dubbed_16k.wav) compares Silero VAD of the ORIGINAL (manifest
speech_intervals) with Silero VAD of the DUBBED track (same call as the pipeline's gate):
  iou                          |orig ∩ dub| / |orig ∪ dub| over the whole timeline (seconds)
  orig_without_dub_s           original speech seconds where the dub is silent (the "silent while original speaks" cause)
  dub_outside_orig_s           dubbed speech seconds where the original is silent
  per_burst                    for every original VAD burst: covered fraction, and bursts with < 50 % coverage
  atempo                       distribution of the speed-ups actually applied (manifest units[].alignment.atempo, or the
                               per-group E1 placements when an e1_alignment.json is next to the dubbed track)
  ls_frames_*                  copied from quality_manifest's <manifests>/<id>_<tag>.quality.json when present
    python scripts/quality/dub_coverage.py --id 04 --tag baseline --run-dir /tmp/dabai_quality/baseline --dubbed /tmp/dabai_quality/work/work_04/dubbed_16k.wav
    python scripts/quality/dub_coverage.py --id 04 --tag e1 --run-dir /tmp/dabai_quality/candidates --dubbed /tmp/dabai_quality/candidates/e1/work_04/dubbed_16k.wav
"""
from __future__ import annotations
import argparse, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))
Q = pathlib.Path("/tmp/dabai_quality")


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def inter(a, b):
    out = []; i = j = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s: out.append((s, e))
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return out


def total(iv):
    return round(sum(e - s for s, e in iv), 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--run-dir", required=True); ap.add_argument("--dubbed", required=True, help="dubbed_16k.wav")
    ap.add_argument("--manifests", default=str(Q / "manifests")); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from face_aware_latentsync_accel import silero_speech_intervals
    man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_{a.tag}.manifest.json").read_text())
    orig = merge([(v["start"], v["end"]) for v in man["speech_intervals"]]); dub = merge(silero_speech_intervals(pathlib.Path(a.dubbed)))
    both = merge(inter(orig, dub)); t_o, t_d, t_i = total(orig), total(dub), total(both)
    per = []
    for s, e in orig:
        c = total(inter([(s, e)], dub)); per.append({"burst": [s, e], "seconds": round(e - s, 3), "covered_fraction": round(c / max(e - s, 1e-6), 3)})
    e1 = pathlib.Path(a.dubbed).parent / "e1_alignment.json"
    if e1.exists():
        rep = json.loads(e1.read_text()); ats = [p["atempo"] for u in rep["units"] for p in u["placements"] if p.get("placed")]
        src = "e1 placements"; extra = {"overflow_groups": rep["summary"]["overflow_groups"], "cut_groups": rep["summary"]["cut_groups"], "groups": rep["summary"]["groups"]}
    else:
        ats = [u["alignment"]["atempo"] for u in man["units"]]; src = "units"; extra = {}
    q = pathlib.Path(a.manifests) / f"{a.id}_{a.tag}.quality.json"; ls = {}
    if q.exists():
        au = json.loads(q.read_text()).get("audio", {}); ls = {k: au.get(k) for k in ("ls_frames", "ls_frames_dub_silent", "ls_frames_dub_silent_while_original_speaks", "ls_silent_fraction")}
        sn = json.loads(q.read_text()).get("syncnet") or {}; ls["syncnet"] = (sn.get("output") or {}).get("confidence"); ls["av_offset"] = (sn.get("output") or {}).get("av_offset_frames")
    res = {"id": a.id, "tag": a.tag, "dubbed": a.dubbed, "orig_speech_s": t_o, "dub_speech_s": t_d, "overlap_s": t_i,
           "iou": round(t_i / max(t_o + t_d - t_i, 1e-6), 4), "orig_without_dub_s": round(t_o - t_i, 3), "dub_outside_orig_s": round(t_d - t_i, 3),
           "orig_bursts": len(orig), "bursts_covered_lt_50pct": sum(1 for p in per if p["covered_fraction"] < 0.5), "bursts_uncovered": sum(1 for p in per if p["covered_fraction"] < 0.05),
           "atempo": {"source": src, "n": len(ats), "max": max(ats) if ats else None, "gt_1_15": sum(x > 1.15 for x in ats), "gt_1_25": sum(x > 1.25 for x in ats), "gt_1_3": sum(x > 1.3 for x in ats), "values": [round(x, 3) for x in ats], **extra},
           "ls": ls, "per_burst": per}
    print(json.dumps({k: v for k, v in res.items() if k != "per_burst"}, indent=1))
    out = pathlib.Path(a.out or (pathlib.Path(a.manifests) / f"{a.id}_{a.tag}.dub_coverage.json")); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1)); print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

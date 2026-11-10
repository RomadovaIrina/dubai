#!/usr/bin/env python3
"""r16d T: bed A/B for Irina's ear (11.11 "местами не слышно говорящего"). Rebuilds the final mix from the SAME stems (work/mix_inputs.json,
written by run_clean_pipeline_05.py) twice: A = the accepted mix (margin off; must equal the delivered mix), B = DUB_BED_MARGIN_DB-style margin
duck. Muxes each onto the given final video (video stream copied) and prints the voice-over-bed stats per version.
    python bed_margin_ab.py --work <audio work dir> --video FINAL.mp4 --margin 8 --out OUTDIR [--tag old03]"""
import argparse, json, os, pathlib, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import background_bed as bb
ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--video", required=True); ap.add_argument("--margin", type=float, default=8.0)
ap.add_argument("--max-db", type=float, default=-12.0); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="v"); a = ap.parse_args()
w = pathlib.Path(a.work); J = json.load(open(w / "mix_inputs.json")); O = pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True); res = {}
for lab, m in (("A_accepted", None), (f"B_margin{a.margin:g}", a.margin)):
    d = O / f"{a.tag}_{lab}"; d.mkdir(exist_ok=True)
    r = bb.mix_background(J["sep"], pathlib.Path(J["dubbed24k"]), J["speech"], J["dub_vad"], J["duration"], d, margin_db=m, margin_max_db=a.max_db, **J["kw"])
    mp4 = O / f"{a.tag}_{lab}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-i", r["aac"], "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", "-movflags", "+faststart", str(mp4)], check=True)
    res[lab] = {"mp4": str(mp4), "margin_duck": r["margin_duck"]}
    vb = r["margin_duck"]["voice_over_bed"]
    print(f"AB {a.tag} {lab}: voice-over-bed median {vb['median_db']} dB, p5 {vb['p5_db']} dB, seconds < 6 dB {vb['lt6_s']}, < 3 dB {vb['lt3_s']} of {vb['seconds']}; "
          f"margin active {r['margin_duck'].get('seconds_active', 0)} s, mean extra {r['margin_duck'].get('mean_extra_db_when_active', 0)} dB; worst {vb['worst'][:4]}", flush=True)
json.dump(res, open(O / f"{a.tag}_ab.json", "w"), indent=1)

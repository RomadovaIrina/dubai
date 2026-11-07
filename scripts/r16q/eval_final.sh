#!/bin/bash
# r16 Q: ONE eval harness for a FINAL (pasted) dub video (audit r16 "Metrics and training" #1 #8 #9 #13).
#   bash scripts/r16q/eval_final.sh FINAL.mp4 DUB16K.wav OUT_DIR [--route route.json --route-wh W,H | --speaker-xy X,Y] [--device cuda|cpu]
#        [--noise 0|1] [--cpu-check] [--label x]
# speech@0 = the mean per-second SyncNet conf at offset 0 (median d - d(0), +-0.5 s context) over the seconds where the ORIGINAL dub speaks
#   (dub_act >= 0.5), FULL length, on the 25 fps grid, SPEAKER tracks only (sync_perframe.py --route / --speaker-xy; without them: largest box).
#   This is K's speech@0 definition (k_gate.sh) on the fixed multi-face sync_perframe.py. NOT comparable with L's best-offset global conf (l_agg.py).
# face check: a speech second whose speaker crop has < 10 SyncNet frames = face lost; FAIL when lost > 5 % of the speech seconds
#   (k_gate.sh silently dropped those seconds: a video that loses the face looked as good as its remaining seconds).
# noise floor (--noise 1, default): the SAME video scored twice more: decode B = re-encoded x264 crf 23, crop B = the picture shifted 8 px
#   (crop + pad: a different S3FD box / 224 crop); noise = max |speech@0 - variant|. A difference between two candidates below ~2x noise is not real.
# --cpu-check: score once on the CPU as well and report |cpu - gpu| (the harness runs on either).
# Output: OUT_DIR/eval.json (+ sync_*.json / .npz); prints one EVAL line.
set -u
F=$(realpath "$1"); DUB=$(realpath "$2"); O=$(realpath -m "$3"); shift 3
SEL=(); DEV=cuda; NOISE=1; CPUCHK=0; LABEL=$(basename "${F%.*}")
while [ $# -gt 0 ]; do case "$1" in
  --route) SEL+=(--route "$(realpath "$2")"); shift 2;; --route-wh) SEL+=(--route-wh "$2"); shift 2;; --speaker-xy) SEL+=(--speaker-xy "$2"); shift 2;;
  --device) DEV=$2; shift 2;; --noise) NOISE=$2; shift 2;; --cpu-check) CPUCHK=1; shift;; --label) LABEL=$2; shift 2;;
  *) echo "eval_final: unknown arg $1"; exit 2;; esac; done
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$R"; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=$R
PY=${PY:-/venv/dabai/bin/python}; LOCK=${EVAL_GPU_LOCK:-/workspace/dub/gpu.lock}; mkdir -p "$O"
V25=$O/final_25fps.mp4
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$F")
if [ "$FPS" = "25/1" ]; then ln -sfn "$F" "$V25"; else ffmpeg -loglevel error -y -nostdin -i "$F" -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset fast -crf 12 -pix_fmt yuv420p "$V25" || exit 1; fi
WH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$V25")

score() {   # score <video> <tag> <device>  (GPU runs under the lock; -o: the lock fd is NOT inherited by the scorer's children)
  local v=$1 t=$2 d=$3; local cmd=($PY scripts/quality/sync_perframe.py --video "$v" --audio "$DUB" --out "$O/sync_$t.json" --device "$d" "${SEL[@]}")
  if [ "$d" = cuda ]; then flock -o "$LOCK" "${cmd[@]}" > "$O/sync_$t.log" 2>&1; else nice "${cmd[@]}" > "$O/sync_$t.log" 2>&1; fi || { echo "EVAL_FAIL score $t (see $O/sync_$t.log)"; tail -3 "$O/sync_$t.log"; return 1; }
}
score "$V25" main $DEV || exit 1
TAGS=(main)
if [ "$NOISE" = 1 ]; then
  ffmpeg -loglevel error -y -nostdin -i "$V25" -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p "$O/v_dec.mp4" && score "$O/v_dec.mp4" dec $DEV && TAGS+=(dec)
  ffmpeg -loglevel error -y -nostdin -i "$V25" -vf "crop=iw-8:ih-8:8:8,pad=iw+8:ih+8:0:0" -c:v libx264 -preset fast -crf 12 -pix_fmt yuv420p "$O/v_crop.mp4" \
    && score "$O/v_crop.mp4" crop $DEV && TAGS+=(crop)
  rm -f "$O/v_dec.mp4" "$O/v_crop.mp4"
fi
[ "$CPUCHK" = 1 ] && [ "$DEV" != cpu ] && score "$V25" cpu cpu && TAGS+=(cpu)
python3 - "$O" "$LABEL" "$F" "$DUB" "$WH" "${TAGS[@]}" <<'PY'
import json, sys
O, label, final, dub, wh, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
LOST_MAX = 0.05
def m(t):
    s = json.load(open(f"{O}/sync_{t}.json")); rows = [r for r in s["per_second"] if (r.get("dub_act") or 0) >= 0.5]
    have = [r["conf_at0"] for r in rows if r.get("conf_at0") is not None]
    return {"speech_at0": round(sum(have) / len(have), 3) if have else None, "n_speech_sec": len(rows), "n_face_sec": len(have),
            "face_lost_share": round(1 - len(have) / max(len(rows), 1), 3), "global": s["global"], "speaker_select": s.get("speaker_select"),
            "multi_face_frames": s.get("multi_face_frames"), "other_tracks": s.get("other_tracks")}
r = {t: m(t) for t in tags}; M = r["main"]
var = [r[t]["speech_at0"] for t in ("dec", "crop") if t in r and r[t]["speech_at0"] is not None]
noise = round(max(abs(v - M["speech_at0"]) for v in var), 3) if var and M["speech_at0"] is not None else None
cpu = round(abs(r["cpu"]["speech_at0"] - M["speech_at0"]), 3) if "cpu" in r and r["cpu"]["speech_at0"] is not None and M["speech_at0"] is not None else None
status = "FAIL_FACE_LOST" if M["face_lost_share"] > LOST_MAX else ("FAIL_NO_SPEECH" if not M["n_speech_sec"] else "OK")
out = {"label": label, "final": final, "dub": dub, "wh": wh, "status": status, "speech_at0": M["speech_at0"], "noise_pm": noise,
       "n_speech_sec": M["n_speech_sec"], "face_lost_share": M["face_lost_share"], "face_lost_max": LOST_MAX, "global_conf": M["global"]["conf"],
       "av_offset": M["global"]["av_offset"], "speaker_select": M["speaker_select"], "multi_face_frames": M["multi_face_frames"],
       "other_tracks": M["other_tracks"], "cpu_minus_gpu": cpu, "variants": {t: r[t]["speech_at0"] for t in tags}}
json.dump(out, open(f"{O}/eval.json", "w"), indent=1)
print("EVAL " + json.dumps({k: out[k] for k in ("label", "status", "speech_at0", "noise_pm", "n_speech_sec", "face_lost_share", "av_offset", "speaker_select", "cpu_minus_gpu")}))
sys.exit(0 if status == "OK" else 3)
PY

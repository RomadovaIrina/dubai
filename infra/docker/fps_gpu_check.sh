#!/bin/bash
# r18 TZ 9.4 on a GPU VM: the native-fps output (E2E_FPS_NATIVE=on, scripts/keysync/r16/tz/fps_native_merge2.py "B4 retime") vs today's
# 25 fps output (A), on what the laptop could not do: SyncNet, and the REAL 60 / 14.97 fps / vertical sources through the image.
#   [HOST_GPU_LOCK=/workspace/gpu.lock] bash infra/docker/fps_gpu_check.sh <image> <pack dir> <out dir> [code dir = this checkout]
# <pack dir> (scripts/keysync/r16/tz/fps_vm_pack.sh makes it on the laptop):
#   jobs/<name>.<ext>        full jobs (scripts/job dabai_job.run_job, keep_intermediate) in the image with E2E_FPS_NATIVE=on (fps60 = story-1773579769284, q08 14.97, q09 vertical 29.97)
#   pairs/<name>/A.mp4 B.mp4 the laptop's class outputs (A = the 25 fps final, B = native fps, same audio): SyncNet only
# HOST_GPU_LOCK (a shared VM): the host lock file is the image's gpu.lock and R10_HEAVY_LOCK holds it for the whole GPU part of a job.
# Per job: B = the job's output; A = the SAME job's 25 fps picture (work/ks_face.mp4, + the old hires_merge on the 25 fps grid when the
# source was > 1280 px), i.e. exactly what E2E_FPS_NATIVE=off would have shipped, muxed with B's audio -> no second GPU run.
# Then for every pair: SyncNet confidence / offset of A and B (scripts/pilot/syncnet_common.py, LatentSync eval), jobs also
# scripts/keysync/r16/tz/fps_eval_class.py (format, mouth jerk, lip pulse) + 0.5x clips A | B | source. Summary: <out>/fps_gpu_summary.json.
set -uo pipefail
IMG=${1:?image}; PK=$(readlink -f "${2:?pack dir}"); O=$(mkdir -p "${3:?out}" && readlink -f "$3"); C=$(readlink -f "${4:-$(dirname "$0")/../..}"); VOL=${VOL:-dabai-models}
LK=(); [ -n "${HOST_GPU_LOCK:-}" ] && { touch "$HOST_GPU_LOCK"; LK=(-v "$(readlink -f "$HOST_GPU_LOCK")":/workspace/dub/gpu.lock -e R10_HEAVY_LOCK=/workspace/dub/gpu.lock); }   # a shared VM: the image's GPU stages
[ -d $PK/jobs ] || [ -d $PK/pairs ] || { echo "no jobs/ or pairs/ in $PK"; exit 2; }
echo "code $(git -C $C rev-parse --short=12 HEAD 2>/dev/null || echo '?') image $IMG"
docker run --rm --gpus all --shm-size 16g "${LK[@]}" -v $C/scripts:/workspace/dub/dubai/scripts:ro -v $C/infra:/workspace/dub/dubai/infra:ro \
  -v $VOL:/workspace/dub/dubai/models -v $PK:/fps/in:ro -v $O:/fps/out -v $O/artifacts:/workspace/dub/artifacts -e E2E_FPS_NATIVE=on -e DABAI_RESIDENT=0 --entrypoint bash $IMG -c '
set -u; cd /workspace/dub/dubai; export DUB_PROFILE=clean DUB_ALLOW_NC=0 DUB_ALLOW_NC_BED=0; bash infra/docker/entrypoint.sh true || exit 3
PY=/venv/dabai/bin/python; T=scripts/keysync/r16/tz; R=/fps/out; mkdir -p $R/pairs $R/clips
FONT=$(ls /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf 2>/dev/null | head -1)
fr() { ffprobe -v error -select_streams v:0 -show_entries stream=$2 -of csv=p=0 "$1" | head -1 | cut -d, -f1; }
cat > /tmp/job.py <<PYEOF
import sys, json; sys.path.insert(0, "scripts/job"); import dabai_job   # the live backend entry (scripts/job), keep_intermediate for work/
n, src, out = sys.argv[1:4]
st = dabai_job.run_job({"job_id": n, "input": src, "out_dir": out, "target_lang": "en", "face": "keysync", "output": {"keep_intermediate": True}})
print("== job", n, st.get("status"), [f for f in st.get("flags") or [] if "fps" in f or "face" in f], (st.get("error") or {}).get("detail", ""))
PYEOF
for j in /fps/in/jobs/*; do [ -f "$j" ] || continue; n=$(basename "${j%.*}"); J=$R/jobs/$n; P=$R/pairs/$n; mkdir -p $J $P
  [ -f $J/status.json ] || { echo "== job $n start $(date +%T)"; /venv/api/bin/python /tmp/job.py $n "$j" $J > $R/jobs/$n.job.log 2>&1; grep "^== job" $R/jobs/$n.job.log || tail -3 $R/jobs/$n.job.log; }
  B=$(ls $J/e2e/${n}_DabAI_r9_demucs.mp4 $J/e2e/${n}_DabAI_r9_spleeter.mp4 2>/dev/null | head -1); W=$J/e2e/work
  [ -n "$B" ] && [ -f $W/ks_face.mp4 ] || { echo "== job $n: no output / no KeySync picture (not a KeySync job?)"; continue; }
  HR=$(ls $W/hires_src.* 2>/dev/null | head -1); AV=$W/ks_face.mp4
  if [ -n "$HR" ] && [ ! -f $P/A_v.mp4 ]; then   # = the e2e.sh face_hires branch (E2E_FPS_NATIVE=off)
    ffmpeg -loglevel error -y -i $HR -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p $P/hires_25fps.mp4
    $PY scripts/r15/hires_merge.py --hi $P/hires_25fps.mp4 --lo-src $W/ks/master_25fps.mp4 --lo-comp $W/ks_face.mp4 --out $P/A_v.mp4 --threads 16 > $P/A_hires.log 2>&1 || echo "== $n A hires_merge FAILED"
  fi
  [ -n "$HR" ] && AV=$P/A_v.mp4
  ffmpeg -loglevel error -y -i $AV -i $B -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $P/A.mp4; ln -sf $B $P/B.mp4
  S=${HR:-$j}; [ "$(fr $S avg_frame_rate)" != "$(fr $S r_frame_rate)" ] && { [ -f $P/src.mp4 ] || ffmpeg -loglevel error -y -i $S -map 0:v:0 -an -fps_mode cfr -r $(fr $S avg_frame_rate) -c:v libx264 -preset medium -crf 12 -pix_fmt yuv420p $P/src.mp4; S=$P/src.mp4; }   # (= e2e.sh variable_fps -> cfr)
  [ -f $P/eval.json ] || $PY $T/fps_eval_class.py --src $S --a $P/A.mp4 --b $P/B.mp4 --name $n --json $P/eval.json > /dev/null 2> $P/eval.err || echo "== $n eval FAILED: $(tail -1 $P/eval.err)"
  [ -f $P/eval.json ] || continue; RT=$(fr $S avg_frame_rate); F=$(python3 -c "n, d = \"$RT\".split(\"/\"); print(round(float(n) / float(d), 3))")
  python3 -c "
import json; r = json.load(open(\"$P/eval.json\")); W, H = r[\"src\"][\"w\"], r[\"src\"][\"h\"]
for t0, w in r[\"windows\"].items():
    if not isinstance(w, dict): continue
    b = w[\"lips_box\"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(b[2] - b[0], b[3] - b[1]) * 1.6
    print(t0, max(0, int(cx - s / 2)) & ~1, max(0, int(cy - s / 2 * 0.85)) & ~1, min(W, int(cx + s / 2)) & ~1, min(H, int(cy + s / 2 * 0.85)) & ~1)" > $P/windows.txt
  while read t0 x0 y0 x1 y1; do for kind in full mouth; do
    o=$R/clips/${n}_${F}fps_${t0}s_${kind}_A_native_src_half_speed.mp4; [ -f $o ] && continue
    [ $kind = full ] && PV="scale=-2:540" || PV="crop=$((x1-x0)):$((y1-y0)):$x0:$y0,scale=-2:400:flags=lanczos"
    lab() { [ -n "$FONT" ] && echo ",drawtext=fontfile=$FONT:text=$1:x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6"; }
    ffmpeg -nostdin -loglevel error -y -ss $t0 -t 3 -i $P/A.mp4 -ss $t0 -t 3 -i $P/B.mp4 -ss $t0 -t 3 -i $S -filter_complex \
"[0:v]fps=$RT,$PV$(lab A_25fps)[p0];[1:v]fps=$RT,$PV$(lab native_${F}fps)[p1];[2:v]fps=$RT,$PV$(lab source_${F}fps)[p2];[p0][p1][p2]hstack=inputs=3,setpts=2*PTS[v];[0:a]atempo=0.5[au]" \
      -map "[v]" -map "[au]" -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart $o || echo "== $n clip $t0 $kind FAILED"
  done; done < $P/windows.txt
done
for d in /fps/in/pairs/*/; do [ -d "$d" ] || continue; n=$(basename $d); mkdir -p $R/pairs/$n; ln -sf $d/A.mp4 $R/pairs/$n/A.mp4; ln -sf $d/B.mp4 $R/pairs/$n/B.mp4; done
for P in $R/pairs/*/; do n=$(basename $P); [ -f $P/syncnet.json ] && continue; [ -e $P/A.mp4 ] && [ -e $P/B.mp4 ] || continue
  exec 9>> /workspace/dub/gpu.lock; flock 9   # SyncNet is on the GPU
  PYTHONPATH=scripts/pilot $PY -c "
import json, sys; from syncnet_common import eval_syncnet
out = {}
for k in (\"A\", \"B\"):
    try: r = eval_syncnet(__import__(\"pathlib\").Path(\"$P/\" + k + \".mp4\")); out[k] = {\"conf\": r[\"confidence\"], \"offset\": r[\"av_offset_frames\"], \"s\": r[\"seconds\"]}
    except Exception as e: out[k] = {\"error\": str(e)[-300:]}
json.dump(out, open(\"$P/syncnet.json\", \"w\")); print(\"== syncnet $n\", json.dumps(out))"; flock -u 9
done
python3 - <<PYEOF
import json, glob, os
S = {}
for p in sorted(glob.glob("/fps/out/pairs/*/")):
    n = os.path.basename(p.rstrip("/")); e = {}
    for f in ("syncnet", "eval"):
        if os.path.exists(p + f + ".json"): e[f] = json.load(open(p + f + ".json"))
    S[n] = e
json.dump(S, open("/fps/out/fps_gpu_summary.json", "w"), indent=1); print("FPS_GPU_DONE", len(S), "pairs")
PYEOF
' 2>&1 | tee $O/fps_gpu_check.log | grep -E "^(== |FPS_GPU_DONE|code )"

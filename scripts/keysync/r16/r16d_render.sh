#!/bin/bash
# r16d T: the day-4 re-render of the 11-video set, the SAME way on every box (r16-a T, r16-b K2, r16-c K1): same 08.11 job files / inputs, one env.
#   R16D_ENV="KS_MASK=tight KS_OCCL=on KS_TAIL_SRC=on <K1 / K2 extras>" bash scripts/keysync/r16/r16d_render.sh m03 m18 old03 q01 ...
# Run from a checkout of experiment/r16-int at the agreed commit (a worktree is fine: models/ + third_party/ symlinked from the main checkout).
# Out: /workspace/dub/outbox/r16d/<v>.mp4 + <v>.status.json + <v>.eval.json (Q eval_final vs the job's own dub) + run.log; stems kept (keep_intermediate).
set -u
R=$(cd "$(dirname "$0")/../../.." && pwd); cd $R; O=${R16D_OUT:-/workspace/dub/outbox/r16d}; W=${R16D_WORK:-/workspace/dub/r16d}; mkdir -p $O $W/json $W/jobs
ENVS=${R16D_ENV:-KS_OCCL=on KS_TAIL_SRC=on}; M=/workspace/dub/inputs/mamakris
declare -A JF=([m03]=final_0811/f0811_m03 [q01]=final_0811/f0811_q01 [q01_subs]=final_0811/f0811_q01_subs [q02]=final_0811/f0811_q02 [q04]=final_0811/f0811_q04
  [q06]=final_0811/f0811_q06 [m18]=golden/full_m18 [old03]=golden/full_old03 [long5]=golden/long5 [q03]=golden/q03_alpha [q05]=golden/q05_e2e)
echo "R16D $(hostname) commit $(git log --oneline -1 | cut -c1-60) env: $ENVS videos: $*" | tee -a $O/run.log
for v in "$@"; do [ -n "${JF[$v]:-}" ] || { echo "R16D unknown video $v" | tee -a $O/run.log; continue; }
  if [ $v = long5 ] && [ ! -s /workspace/dub/r16t/long/long5_stories.mp4 ]; then   # the 08.11 recipe: 8 mamakris stories -> 720x1280 25 fps, 300 s
    mkdir -p /workspace/dub/r16t/long; ( cd /workspace/dub/r16t/long; rm -f list.txt; i=0
    for s in story-1773579769284-320619146.MP4 story-1774249930391-60765207.MOV story-1774250573471-666527943.MP4 story-1774250795589-441186215.mp4 story-1774250832175-746471562.mp4 story-1774250884789-692894647.mp4 story-1774250904597-834383232.mp4 story-1773579769284-320619146.MP4; do
      ffmpeg -v error -y -i $M/$s -vf "fps=25,scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1" -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p -c:a aac -ar 48000 -ac 2 -b:a 192k p$i.mp4; echo "file p$i.mp4" >> list.txt; i=$((i+1)); done
    ffmpeg -v error -y -f concat -safe 0 -i list.txt -t 300 -c copy long5_stories.mp4 ); fi
  J=$W/json/r16d_$v.json
  python3 - "$R/scripts/keysync/r16/${JF[$v]}.json" "$J" "$W/jobs/r16d_$v" "r16d_$v" "$ENVS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["job_id"] = sys.argv[4]; d["out_dir"] = sys.argv[3]; d.setdefault("output", {})["keep_intermediate"] = True
d["debug_env"] = {**(d.get("debug_env") or {}), **dict(kv.split("=", 1) for kv in sys.argv[5].split())}; json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
  rm -rf $W/jobs/r16d_$v; s=$(date +%s)
  env DUB_ALLOW_NC_BED=0 R10_HEAVY_LOCK=/workspace/dub/gpu.lock /venv/dabai/bin/python scripts/job/dabai_job.py $J > $W/json/r16d_$v.log 2>&1
  ST=$W/jobs/r16d_$v/status.json; cp $ST $O/$v.status.json 2>/dev/null
  OUT=$(python3 -c "import json; print(json.load(open('$ST')).get('output') or '')" 2>/dev/null)
  echo "R16D_JOB $v $(python3 -c "import json; d=json.load(open('$ST')); print(d['status'], sorted({w['code'] for w in d.get('warnings', [])}), d.get('error'))" 2>&1) $(( $(date +%s) - s )) s" | tee -a $O/run.log
  [ -s "$OUT" ] || continue; ffmpeg -v error -y -i "$OUT" -map 0 -c copy -movflags +faststart $O/$v.mp4
  DUB=/workspace/dub/artifacts/r4a/r9_r16d_$v/r16d_$v/dubbed_16k.wav; [ -s $DUB ] || DUB=""   # = e2e AD ($A/r9_$ID/$ID): NOT a find (the r9_<id>src stage folders hold earlier dubs: m03 scored 3.4 instead of 7.7)
  [ -n "$DUB" ] || DUB=$(python3 -c "import json; print((json.load(open('$J')).get('debug_env') or {}).get('DUB_AUDIO_FROM', ''))")/dubbed_16k.wav
  RT=$(ls $W/jobs/r16d_$v/e2e/work/route/route.json 2>/dev/null | head -1); RA=(); [ -n "$RT" ] && RA=(--route $RT --route-wh $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $(python3 -c "import json; print(json.load(open('$J'))['input'])")))
  [ -s "$DUB" ] && bash scripts/r16q/eval_final.sh $O/$v.mp4 $DUB $W/eval_$v "${RA[@]}" --noise 0 --label r16d_$v 2>&1 | grep -E "^EVAL" | cut -c1-260 | tee -a $O/run.log && cp $W/eval_$v/eval.json $O/$v.eval.json
done
echo "R16D_DONE $*" | tee -a $O/run.log

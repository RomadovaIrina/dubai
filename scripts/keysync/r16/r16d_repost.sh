#!/bin/bash
# r16d T: RE-PASTE ONLY a finished r16d_render.sh job (no regeneration, CPU): the face post + paste with a new env, then exactly e2e's tail:
# hi-res merge (if the face work was downscaled), burned subs / watermark (if the job had them), mux with the job's own mix, eval vs its own dub.
#   R16D_ENV="KS_MASK=tight KS_OCCL=on KS_TAIL_SRC=on KS_CAP=1.0" bash scripts/keysync/r16/r16d_repost.sh q03 q05 ...
# Multi-shot jobs: ks_shots.sh with KS_SHOTS_REPOST=1 (keeps every shot's render; same post loop + join as the first run). Old final -> <v>_prev.mp4.
set -u
R=$(cd "$(dirname "$0")/../../.." && pwd); cd $R; . scripts/env.sh > /dev/null 2>&1; O=${R16D_OUT:-/workspace/dub/outbox/r16d}; JW=${R16D_WORK:-/workspace/dub/r16d}; mkdir -p $O
ENVS=${R16D_ENV:?set R16D_ENV}; PY=/venv/dabai/bin/python
for v in "$@"; do ID=r16d_$v; J=$JW/jobs/$ID; W=$J/e2e/work; KW=$W/ks; ST=$J/status.json; s=$(date +%s)
  [ -f $KW/shots.json ] || { echo "REPOST $v: no job work dir ($KW)"; continue; }
  AD=/workspace/dub/artifacts/r4a/r9_$ID/$ID   # = e2e AD exactly (the r9_<id>src stage folders hold earlier dubs / mixes)
  [ -n "$AD" ] && [ -f $AD/final_mix_aac.m4a ] || AD=$($PY -c "import json; print((json.load(open('$ST'))['params'].get('debug_env') or {}).get('DUB_AUDIO_FROM', ''))")
  RT=""; [ -f $W/route/route.json ] && RT=$W/route/route.json
  for K in $KW/kit $W/shots/s*/ks/kit; do [ -d $K ] && cp scripts/keysync/*.py scripts/keysync/r15/*.py scripts/keysync/r16/*.py $K/ 2>/dev/null; done   # the kit dates from the job's prep: refresh it to this checkout (new ks_plain options)
  NSEG=$($PY -c "import json; print(len(json.load(open('$KW/shots.json'))['segments']))"); NEW=$W/ks_face_repost.mp4; rm -f $NEW
  if [ "$NSEG" -gt 1 ]; then env $ENVS KS_ROOT=${KS_ROOT:-/workspace/dub/keysync} KS_SHOTS_REPOST=1 bash scripts/keysync/r16/ks_shots.sh $ID $AD $W $NEW "$RT" $KW > $W/repost_face.log 2>&1
  else env $ENVS KS_ROOT=${KS_ROOT:-/workspace/dub/keysync} KS_PHASE=post bash scripts/keysync/r16/ks_face.sh $ID x $AD $W $NEW $RT > $W/repost_face.log 2>&1; fi
  rc=$?; grep -E "KS_SHOTS_REPOST|KS_PASTE|K2_PAUSE|KS_TAILSRC|KS_SHOTS_CONCAT" $W/repost_face.log | cut -c1-200 | sed "s/^/REPOST $v /"
  [ $rc = 0 ] && [ -s $NEW ] || { echo "REPOST $v: face post failed rc=$rc (see $W/repost_face.log)"; continue; }
  VID=$NEW
  if ls $W/hires_src.* > /dev/null 2>&1; then HI=$W/hires_25fps.mp4
    [ -s $HI ] || ffmpeg -loglevel error -y -i $(ls $W/hires_src.* | head -1) -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p $HI
    $PY scripts/r15/hires_merge.py --hi $HI --lo-src $KW/master_25fps.mp4 --lo-comp $VID --out $W/face_hires_repost.mp4 --json $W/face_hires_repost.json --threads 16 > $W/repost_hires.log 2>&1 \
      && VID=$W/face_hires_repost.mp4 || { echo "REPOST $v: hires merge failed"; continue; }; fi
  SA=$($PY - "$ST" "$AD" "$ID" <<'PY'
import json, sys, glob
p = json.load(open(sys.argv[1]))["params"]; s, w = p.get("subs") or {}, p.get("watermark") or {}; a = []
if s.get("enabled"):
    m = (glob.glob(f"{sys.argv[2]}/{sys.argv[3]}_*.manifest.json") or [""])[0]
    a += ["--manifest", m, "--lang", s.get("lang", "dub"), "--style", s.get("style", "yellow"), "--karaoke", "on" if s.get("karaoke", True) else "off"]
elif w.get("enabled"): a += ["--no-subs"]
if w.get("enabled"): a += ["--watermark", w.get("png") or "text:" + w.get("text", "DabAI"), "--wm-corner", w.get("corner", "br"), "--wm-opacity", str(w.get("opacity", 0.6))]
print(" ".join(a))
PY
)
  if [ -n "$SA" ]; then SUBS_THREADS=8 $PY scripts/r7/r7_subs.py burn --video $VID $SA --out $W/final_subs_repost.mp4 > $W/repost_subs.log 2>&1 && VID=$W/final_subs_repost.mp4 || { echo "REPOST $v: subs burn failed"; continue; }; fi
  [ -f $O/$v.mp4 ] && mv -f $O/$v.mp4 $O/${v}_prev.mp4
  ffmpeg -loglevel error -y -i $VID -i $AD/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $O/$v.mp4 || { echo "REPOST $v: mux failed"; continue; }
  RA=(); [ -n "$RT" ] && RA=(--route $RT --route-wh $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $($PY -c "import json; print(json.load(open('$J/../../json/$ID.json'))['input'])" 2>/dev/null || echo x)))
  bash scripts/r16q/eval_final.sh $O/$v.mp4 $AD/dubbed_16k.wav $JW/eval_repost_$v "${RA[@]}" --noise 0 --device ${R16D_EVAL_DEVICE:-cuda} --label repost_$v 2>&1 | grep -E "^EVAL" | cut -c1-220 | sed "s/^/REPOST $v /"
  cp $JW/eval_repost_$v/eval.json $O/$v.eval.json 2>/dev/null; echo "REPOST_DONE $v env [$ENVS] $(( $(date +%s) - s )) s" | tee -a $O/run.log
done

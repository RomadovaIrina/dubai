#!/bin/bash
# score4.sh <video> "<t0:t1> ...": base CP vs adapt CP (p1): speech@0 (Q eval_final, locked GPU), K1 cheek_colour (edge / inner / lower jaw Lab vs
# source, + now_1011), the 0.5x 4-panels "now_1011 | base CP | adapt CP | source" at the spots, the GPU cost -> sc_<v>/ + outbox/r16_l/d4/
v=$1; SPOTS=$2; D=/workspace/dub/r16l/d4; O=$D/sc_$v; X=/workspace/dub/outbox/r16_l/d4; mkdir -p $O $X; FT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
DUB=/workspace/dub/r16in/HANDOFF_K/paste_inputs/$v/dubbed_16k.wav; NOW=/workspace/dub/k1/now1011/now_$v.mp4
B=$D/base_$v/l4_base_$v.mp4; P=$D/p1_$v/l4_p1_$v.mp4; KW=$(dirname $(find $D/base_$v -path "*ks/master_25fps.mp4" | head -1))
ARMS=${ARMS:-base p1}; declare -A LAB=( [base]=base_CP [p1]=adapt_CP_p1 [p1h]=adapt_CP_half ); TAG=${TAG:-adapt}
spd() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["speech_at0"], d["status"])' "$1"; }
for a in $ARMS; do F=$D/${a}_$v/l4_${a}_$v.mp4; if [ -s $O/ev_$a/eval.json ]; then echo "$v $a speech@0 $(spd $O/ev_$a/eval.json) (cached)"; continue; fi
  bash /workspace/dub/r16l/int/scripts/r16q/eval_final.sh $F $DUB $O/ev_$a --noise 0 --label ${v}_$a > $O/ev_$a.log 2>&1
  echo "$v $a speech@0 $(python3 -c "import json; d=json.load(open(\"$O/ev_$a/eval.json\")); print(d[\"speech_at0\"], d[\"status\"])")"; done
/venv/dabai/bin/python3 /workspace/dub/k1/cheek_colour.py $KW/lm_f0.npz $KW/master_25fps.mp4 $NOW $(for a in $ARMS; do echo -n "$D/${a}_$v/l4_${a}_$v.mp4 "; done) 2>&1 | grep -v -i "broken\|muxing\|trailer\|closing" > $O/colour.txt; cat $O/colour.txt
read BX BY BS < <(python3 -c "import json;d=json.load(open(\"$KW/data/box_f0.json\"));print(d[\"x\"],d[\"y\"],d[\"side\"])")
for sp in $SPOTS; do t0=${sp%:*}; t1=${sp#*:}; i=0; IN=(); FC=""; L=""
  PL=("now_1011=$NOW"); for a in $ARMS; do PL+=("${LAB[$a]}=$D/${a}_$v/l4_${a}_$v.mp4"); done; PL+=("source=$KW/master_25fps.mp4")
  for p in "${PL[@]}"; do lab=${p%%=*}; f=${p#*=}; IN+=(-i "$f")
    FC+="[$i]fps=25,trim=$t0:$t1,setpts=PTS-STARTPTS,crop=$BS:$BS:$BX:$BY,scale=320:320,drawtext=fontfile=$FT:text=$lab:x=5:y=h-24:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6[v$i];"; L+="[v$i]"; i=$((i+1)); done
  FC+="${L}hstack=$i,drawtext=fontfile=$FT:text=$v from ${t0}s (0.5x):x=5:y=5:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6,setpts=2*PTS"; echo "$FC" > $O/fc_$t0.txt
  ffmpeg -v error -y "${IN[@]}" -filter_complex_script $O/fc_$t0.txt -an -r 25 -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart $X/${v}_${TAG}_${t0}-${t1}s_half_speed.mp4 && echo "clip $X/${v}_${TAG}_${t0}-${t1}s_half_speed.mp4"; done
python3 - $v <<"PY"
import json, sys, subprocess
v = sys.argv[1]; D = "/workspace/dub/r16l/d4"; A = json.load(open(f"/workspace/dub/r16in/L_adapt/{v}_p1/adapt_time.json"))
src = {"m03": "/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV", "old03": "/workspace/dub/inputs/test_videos_raw/03.mp4"}[v]
vmin = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src], capture_output=True, text=True).stdout) / 60
import os
g = {a: json.load(open(f"{D}/{a}_{v}/status.json")).get("gpu_busy_s") for a in ("base", "p1", "p1h") if os.path.exists(f"{D}/{a}_{v}/status.json")}
ad = A["load_s"] + A["dump_s"] + A["train_s"]
r = {"video": v, "vmin": round(vmin, 3), "job_gpu_s": g, "job_gpu_s_per_vmin": {k: round(x / vmin, 1) for k, x in g.items() if x},
     "adapt_gpu_s": round(ad, 1), "adapt_gpu_s_per_vmin": round(ad / vmin, 1), "adapt_time": A,
     "total_p1_per_vmin": round((g["p1"] + ad) / vmin, 1) if g["p1"] else None}
json.dump(r, open(f"{D}/sc_{v}/cost.json", "w")); print("COST " + json.dumps(r))
PY

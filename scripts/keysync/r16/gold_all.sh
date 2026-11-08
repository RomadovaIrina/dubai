#!/bin/bash
# r16 T golden test, per video: gold_all.sh <v> <src> <accepted.mp4> <mix dir: dubbed_16k.wav + final_mix_aac.m4a> [route.json] [t1]
# -> golden/cmp/<v>.json (face-box PSNR per second e2e vs accepted + the 2 worst seconds' 0.5x clips), frames, audio md5 vs the mix,
#    Q's harness speech@0 (scripts/r16q/eval_final.sh: full length, speaker track) for e2e AND the accepted final.
v=$1; SRC=$2; ACC=$3; MD=$4; RT=${5:-}; T1=${6:-0}; G=/workspace/dub/r16t/golden; R=/workspace/dub/dubai; J=$G/jobs/gold_$v; C=$G/cmp; mkdir -p $C
OUT=$(ls $J/*.mp4 | grep -v -E "_(spleeter|demucs)" | head -1); [ -n "$OUT" ] || OUT=$(ls $J/*.mp4 | head -1); W=$J/e2e/work/ks
WS=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x $W/master_25fps.mp4); BOX=$(ls $W/data/box_f0.json $J/e2e/work/shots/s*/ks/data/box_f0.json 2>/dev/null | head -1)
/venv/dabai/bin/python $R/scripts/keysync/r16/gold_cmp.py --acc $ACC --e2e $OUT --src $SRC --box $BOX --work-size $WS --out $C/$v --t1 $T1 2>/dev/null | grep -E "^GOLD|^clip"
nf() { ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$1"; }
am() { ffmpeg -v error -i "$1" -map 0:a:0 -c copy -f md5 - 2>/dev/null; }
echo "FRAMES $v e2e $(nf $OUT) accepted $(nf $ACC) master $(nf $W/master_25fps.mp4)"
echo "AUDIO $v e2e==mix $([ "$(am $OUT)" = "$(am $MD/final_mix_aac.m4a)" ] && echo yes || echo NO) accepted==mix $([ "$(am $ACC)" = "$(am $MD/final_mix_aac.m4a)" ] && echo yes || echo NO)"
RA=(); [ -n "$RT" ] && RA=(--route $RT --route-wh $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $SRC))   # (Q made the route on the source)
for x in "e2e $OUT" "acc $ACC"; do set -- $x
  bash $R/scripts/r16q/eval_final.sh $2 $MD/dubbed_16k.wav $C/eval_${v}_$1 "${RA[@]}" --noise 0 --label ${v}_$1 2>&1 | grep -E "^EVAL" | cut -c1-300; done

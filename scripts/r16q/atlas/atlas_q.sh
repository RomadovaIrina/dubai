#!/bin/bash
# r16 Q atlas on r16-q (CPU, capped: 8 processes x 4 threads; ffmpeg readers 2 threads). atlas_q.sh SETNAME E2E_DIR [ids...]
#   per id: r16q_defects (all checks incl. seam / mouth) -> r16q_detect (all detectors, same cache) -> spring (single-shot ids)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
SET=$1; E=$2; shift 2; Q=/workspace/dub/r16q; O=$Q/out_$SET; mkdir -p $O; C=/workspace/dub/r16in/day1/HANDOFF_Q/clips; M=/workspace/dub/inputs/mamakris
PY=/venv/mediapipe/bin/python; S=$Q/scripts
declare -A SRC=([f0811_m03]=$M/story-1774249919482-367803614.MOV [full_m18]=$M/story-1774250817106-616905094.mp4
 [full_old03]=/workspace/dub/inputs/test_videos_raw/03.mp4 [f0811_q01]=$C/q01_amir.mp4 [f0811_q01_subs]=$C/q01_amir.mp4
 [f0811_q02]=$C/q02_vera.mp4 [q03_e2e_alpha_full]=$C/q03_priest.mp4 [f0811_q04]=$C/q04_farikh.mp4 [q05_e2e_full]=$C/q05_rozhkov.mp4
 [f0811_q06]=$C/q06_suvorov.mp4 [long5]=$Q/src/long5_stories.mp4)
SINGLE=" f0811_m03 full_m18 full_old03 f0811_q01 f0811_q01_subs f0811_q04 "
one() { id=$1; F=$E/$id.mp4; Sx=${SRC[$id]}; for x in $F $Sx; do [ -s $x ] || { echo "MISSING $id $x"; return; }; done
  [ -s $Q/defects/$id.det.npz ] && [ ! -s $O/$id.det.npz ] && cp $Q/defects/$id.det.npz $O/
  nice -n 10 $PY $S/r16q_defects.py --src $Sx --ours $F --out $O/$id.json --npz $O/$id.det.npz --label $id > $O/$id.log 2>&1; r1=$?
  nice -n 10 $PY $S/r16q_detect.py --src $Sx --ours $F --out $O/$id.detall.json --npz $O/$id.det.npz --label $id > $O/$id.detall.log 2>&1; r2=$?
  r3=-; if [[ "$SINGLE" == *" $id "* ]]; then WH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $F | tr , :); MS=$Q/masters/${SET}_$id.25.mp4
    [ -s $MS ] || nice ffmpeg -v error -threads 4 -y -i $Sx -an -vf "fps=25,scale=$WH" -c:v libx264 -threads 4 -crf 12 -preset fast $MS
    nice -n 10 $PY $S/r16q_spring.py --final $F --master $MS --out $O/$id.spring.json > $O/$id.spring.log 2>&1; r3=$?; fi
  echo "$id defects=$r1 detect=$r2 spring=$r3"; }
export -f one; export E O Q S PY C M SINGLE SET
ids=("$@"); [ ${#ids[@]} = 0 ] && ids=(long5 full_old03 f0811_q02 f0811_q06 q05_e2e_full q03_e2e_alpha_full f0811_q04 f0811_q01 f0811_q01_subs f0811_m03 full_m18)
for id in "${ids[@]}"; do echo $id; done | xargs -P 8 -I{} bash -c "$(declare -p SRC); one {}"
echo ATLAS_Q_DONE $SET

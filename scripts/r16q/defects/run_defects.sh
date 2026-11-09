#!/bin/bash
# r16 Q day 3 on r16-k2 (CPU only): defects check per e2e video once its mp4 has landed. run_defects.sh [ids...]
Q=/workspace/dub/q_tmp; E=$Q/e2e; O=$Q/defects; C=/workspace/dub/r16in/day1/HANDOFF_Q/clips; M=/workspace/dub/inputs/mamakris
declare -A SRC=([f0811_m03]=$M/story-1774249919482-367803614.MOV [full_m18]=$M/story-1774250817106-616905094.mp4
 [full_old03]=/workspace/dub/inputs/test_videos_raw/03.mp4 [f0811_q01]=$C/q01_amir.mp4 [f0811_q01_subs]=$C/q01_amir.mp4
 [f0811_q02]=$C/q02_vera.mp4 [q03_e2e_alpha_full]=$C/q03_priest.mp4 [f0811_q04]=$C/q04_farikh.mp4 [q05_e2e_full]=$C/q05_rozhkov.mp4
 [f0811_q06]=$C/q06_suvorov.mp4 [long5]=/workspace/dub/q_tmp/src/long5_stories.mp4)
declare -A OURS=([long5]=$E/long5_full.mp4)
for id in "$@"; do (
  F=${OURS[$id]:-$E/$id.mp4}; S=${SRC[$id]}
  for i in $(seq 1 240); do [ -s "$F" ] && break; sleep 30; done
  for x in "$F" "$S"; do [ -s "$x" ] || { echo "MISSING $id $x"; exit 2; }; done
  nice -n 10 /venv/mediapipe/bin/python $Q/scripts/r16q_defects.py --src "$S" --ours "$F" --out $O/$id.json --npz $O/$id.det.npz --label $id > $O/$id.log 2>&1
  echo "$id rc=$? $(tail -1 $O/$id.log | cut -c1-400)" ) &
done; wait; echo RUN_DONE

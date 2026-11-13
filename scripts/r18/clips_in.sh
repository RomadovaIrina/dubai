#!/bin/bash
# r18, INSIDE the image: the half-speed clips of one video for Irina's eye -> /r18/out/<id>/clips/
#   labelled grid (face crop): 08.11 | restored | source / seam a | seam b | seam c, at the moments she flagged
#   blind grid: restored / a / b / c shuffled as A-D (the key in <clip>.key.json, not on the picture)
#   clips_in.sh <id> <08.11 file in /r18/ref> <input in /r18/in> ["t0-t1:name ..." labelled windows] ["t0-t1:name ..." blind windows]
set -u
ID=$1; REF=/r18/ref/$2; SRC=/r18/in/$3; O=/r18/out/$ID; W=/data/jobs/$ID/e2e/work; P=/venv/dabai/bin/python; S=/workspace/dub/dubai/scripts/r18/r18_sbs.py; mkdir -p $O/clips
declare -A LW=([m03]="8-12:cheek_hair 19-23:cheek_hair 36-40.4:tail" [old03]="0-6:glued_lips 62-67:black_hole_right_cheek 85-90.2:tail"
               [long5]="0-6:jaw_hood 46-52:chunk_jump_0-48 106-112:1-48 124-130:2-06_hair_edge 260-267:4-23_cut")
declare -A BW=([m03]="0-40.4:full" [old03]="0-20:start 55-75:mid" [long5]="0-12:start 120-132:2-00")
LWS=${4:-${LW[$ID]:-}}; BWS=${5:-${BW[$ID]:-}}; seed=318
for w in $LWS; do t=${w%%:*}; n=${w#*:}; f=$O/clips/${ID}_${t%%-*}-${t#*-}s_$n.mp4
  $P $S --in 08.11=$REF --in restored=$O/restored.mp4 --in source=$SRC --in seam_a=$O/a.mp4 --in seam_b=$O/b.mp4 --in seam_c=$O/c.mp4 --cols 3 --crop auto --work $W --height 720 --audio 1 --t0 ${t%%-*} --t1 ${t#*-} --out $f 2>&1 | tail -1; done
for w in $BWS; do t=${w%%:*}; n=${w#*:}; f=$O/clips/${ID}_BLIND_seam_${t%%-*}-${t#*-}s_$n.mp4; seed=$((seed + 7))
  $P $S --in restored=$O/restored.mp4 --in a=$O/a.mp4 --in b=$O/b.mp4 --in c=$O/c.mp4 --blind restored,a,b,c --seed $seed --cols 2 --crop auto --work $W --height 720 --audio 0 --t0 ${t%%-*} --t1 ${t#*-} --out $f 2>&1 | tail -1; done
echo "CLIPS_DONE $ID $(ls $O/clips/*.mp4 | wc -l) clips"

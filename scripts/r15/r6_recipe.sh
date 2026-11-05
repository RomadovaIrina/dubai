#!/bin/bash
# r6 eval with agent F's clean lip recipe (reports/experiments/r15-f3/REPORT.md, 3db4167): CodeFormer off (the pre-CF layer) + layer SR
# + the adaptive lip-contrast flag (LCA 0.45) + lip colour lock and flow median (lip_temporal --mode both --radius 5 --mradius 2),
# on the generator runs of q2u / the r14 student s4p1500c8u / r6c8u (old03, m03); then sync_speech, mouth_open (vs q2u + recipe),
# F's lip_flicker, and the eye clips. GPU only for the layer SR and SyncNet (gated, gpu.lock); the composites are CPU.
#   bash scripts/r15/r6_recipe.sh      (after the r6 E2E; called by scripts/r15/r6_chain.sh)
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R; export DUB_ROOT=$R; . scripts/env.sh >/dev/null
X=${R5_ROOT:-/workspace/dub/r15x}; E=$X/eval; N=$X/R6_NOTES.txt; OB=/workspace/dub/outbox/r15x; B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a
J=$X/lipflick_r6.jsonl; mkdir -p $OB
note(){ echo "$(date +%T) $*" | tee -a $N; }
gpu_idle(){ for i in $(seq 1 240); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return 0; sleep 15; done; }
ARMS=${R6_ARMS:-q2u s4p1500c8u r6c8u}; IDS=${R6_IDS:-old03 m03}
note "recipe start: arms $ARMS on $IDS"
for v in $IDS; do for a in $ARMS; do id=${a}_$v; ST=r9g_$id; W=$B/$ST/work_${id}_$ST; P=$W/${id}_${ST}_25fps
  [ -s ${P}_ls_pre.mp4 ] || { note "recipe $id: no generator run ($P)"; continue; }
  # 1. the CF-off layer = the pre-CodeFormer dump (as scripts/r15/run_f2.sh), 2. its layer SR (GPU)
  [ -e ${P}_ls_cfd_off.mp4 ] || ln -s $(basename ${P}_ls_pre.mp4) ${P}_ls_cfd_off.mp4; cp ${P}_ls_raw.mediapipe.npz ${P}_ls_cfd_off.mediapipe.npz
  S=${P}_ls_cfd_off_srt1.mp4
  if [ ! -s $S ]; then gpu_idle
    flock /workspace/dub/gpu.lock python scripts/r7/r7_layer_sr.py --ls-raw ${P}_ls_cfd_off.mp4 --lm ${P}_ls_cfd_off.mediapipe.npz --out $S --temporal 1 --batch 8 --mask down2 | grep "^==" | cut -c1-200
    cp ${P}_ls_cfd_off.mediapipe.npz ${S%.mp4}.mediapipe.npz; fi
done; done
# 3. the recipe composites (CPU, two at a time)
for v in $IDS; do for a in $ARMS; do id=${a}_$v
  [ -s $B/r9f_${id}rec/${id}_r9f_${id}rec.mp4 ] && continue
  ( GT=r9g_$id LCA=0.45 LT=both LT_EXTRA="--radius 5 --mradius 2" LT_TAG=r5 bash scripts/r15/comp_f2.sh $id rec off ) &
  [ $(jobs -r | wc -l) -ge 2 ] && wait -n
done; done; wait
# 4. metrics
for v in $IDS; do
  SRC=$(ls $E/q2u_$v/e2e/work/in/q2u_$v.* | head -1); lm=$B/r9g_q2u_$v/timing/q2u_${v}_source_lm_F.npz; VV=""
  for a in $ARMS; do id=${a}_$v; V=$B/r9f_${id}rec/${id}_r9f_${id}rec.mp4; [ -s $V ] || { note "recipe $id: no composite"; continue; }
    w=$(ls $A/r9_$id/$id/dubbed_16k.wav 2>/dev/null | head -1)
    [ -s $E/${id}rec.sync.txt ] || flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $V --audio $w --label ${id}rec 2>&1 | grep "^== sync" > $E/${id}rec.sync.txt
    note "$(cat $E/${id}rec.sync.txt)"
    SRC_OVR=$SRC bash scripts/r15/lf.sh ${id}rec $J > $X/lf_${id}rec.log 2>&1 || note "lip_flicker $id failed: $(tail -1 $X/lf_${id}rec.log | cut -c1-150)"
    VV="$VV ${a}rec=$V"
  done
  python scripts/r14/mouth_open.py --wav $A/r9_q2u_$v/q2u_$v/dubbed_16k.wav --video $VV --json $X/recipe_$v.mouth.json > $X/recipe_$v.mouth.log 2>&1
  # 5. eye clips: the r6 student vs the r14 student and vs q2u, all with the recipe (mouth 2-panel, 0.5x; m03 also the 17-21 s span)
  for L in s4p1500c8u q2u; do
    l=$B/r9f_${L}_${v}rec/${L}_${v}_r9f_${L}_${v}rec.mp4; r=$B/r9f_r6c8u_${v}rec/r6c8u_${v}_r9f_r6c8u_${v}rec.mp4; { [ -s $l ] && [ -s $r ]; } || continue
    o=$OB/${v}_mouth_${L}rec_vs_r6c8urec.mp4
    python scripts/r13/distill/mouth_2panel.py --left $l --right $r --lm $lm --out $o --labels "$L + lip recipe" "r6 student + lip recipe"
    ffmpeg -v error -y -i $o -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast ${o%.mp4}_half_speed.mp4
    [ $v = m03 ] && ffmpeg -v error -y -ss 17 -to 21 -i $o -filter_complex "[0:v]setpts=2*PTS,drawtext=text='t=17-21 s (0.5x)':x=(w-tw)/2:y=h-30:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v];[0:a]atempo=0.5[a]" \
      -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac ${o%.mp4}_t17-21_half_speed.mp4
  done
done
python scripts/r15/r6_recipe_table.py --root $X --arms $ARMS --ids $IDS > $X/R6_RECIPE_TABLE.md 2>&1; cat $X/R6_RECIPE_TABLE.md | tee -a $N
note "recipe done"

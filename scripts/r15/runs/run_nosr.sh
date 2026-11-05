cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m
for x in q2_m03:r9g_q2_m03fu q2_old03:r9g_q2_old03; do IFS=: read id ST <<< "$x"; P=$B/$ST/work_${id}_$ST/${id}_${ST}_25fps
  for lab in ia04 off; do L=${P}_ls_cfd_$lab; [ -s ${L}_srnl.mp4 ] || python scripts/r15/lips_nosr.py --pre $L.mp4 --sr ${L}_srt1.mp4 --lm $L.mediapipe.npz --out ${L}_srnl.mp4 | sed "s/^/$id $lab /"; cp $L.mediapipe.npz ${L}_srnl.mediapipe.npz; done & done; wait
until grep -q LT_DONE /workspace/dub/r15f3/run_lt.out; do sleep 30; done
( SRL=srnl GT=r9g_q2_m03fu bash scripts/r15/comp_f3.sh q2_m03 funl ia04 ) & ( SRL=srnl LCA=0.45 GT=r9g_q2_m03fu bash scripts/r15/comp_f3.sh q2_m03 fublanl off ) & wait
( SRL=srnl GT=r9g_q2_old03 bash scripts/r15/comp_f3.sh q2_old03 nl ia04 ) & ( SRL=srnl LCA=0.45 GT=r9g_q2_old03 bash scripts/r15/comp_f3.sh q2_old03 blanl off ) & wait
echo NOSR_DONE $(date +%T)

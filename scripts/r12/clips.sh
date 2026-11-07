cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/outbox/r14_f
for a in "old03 f1" "old03 f2" "m03 f1" "m03 f2"; do set -- $a; bash scripts/r12/clip_ls.sh $1 $2 "q2 + $2" $O 4 | tail -2 & done
for t in f1 f2; do bash scripts/r12/clip_at.sh m03 $B/r9f_m03/m03_r9f_m03.mp4 "q2 (now)" $B/r9f_m03$t/m03_r9f_m03$t.mp4 "$t" "8.3-8.8,11.3-11.7,24.2-24.8" $O/m03_q2_vs_${t}_mouth05x.mp4 & done
wait; echo CLIPS_DONE

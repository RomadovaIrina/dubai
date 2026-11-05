cd /workspace/dub/dubai-f
AUD="--retime-hf nearest --cm-mouth-mode luma --skin-texture off --boundary off --seg-joint 3"
( FCX="$AUD" GT=r9g_q2_m03fu bash scripts/r15/comp_f4.sh q2_m03 fuaud ia04 ) & ( FCX="$AUD" GT=r9g_q2_old03 bash scripts/r15/comp_f4.sh q2_old03 aud ia04 ) & wait
until grep -q LT3_DONE /workspace/dub/r15f3/run_lt3.out; do sleep 20; done
( FCX="$AUD" LCA=0.45 LT=color GT=r9g_q2_m03fu bash scripts/r15/comp_f4.sh q2_m03 fublacaud off ) & ( FCX="$AUD" LCA=0.45 LT=color GT=r9g_q2_old03 bash scripts/r15/comp_f4.sh q2_old03 blacaud off ) & wait
J=/workspace/dub/r15f3/lipflick_aud.jsonl; : > $J; for t in q2_m03fuaud q2_old03aud q2_m03fublacaud q2_old03blacaud; do bash scripts/r15/lf.sh $t $J > /dev/null 2>&1 & done; wait; echo AUD_DONE $(date +%T)

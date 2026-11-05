cd /workspace/dub/dubai-f
st(){ GT=r9g_s4p1500c8u_m03 LT=color bash scripts/r15/comp_f2.sh s4p1500c8u_m03 ltc ia04; LCA=0.45 GT=r9g_s4p1500c8u_m03 LT=color bash scripts/r15/comp_f2.sh s4p1500c8u_m03 lac ia04; }
st &
until grep -q LT2_DONE /workspace/dub/r15f3/run_lt2.out; do sleep 20; done
( LCA=0.45 GT=r9g_q2_m03fu LT=both bash scripts/r15/comp_f2.sh q2_m03 fublab off ) & ( LCA=0.45 GT=r9g_q2_old03 LT=both bash scripts/r15/comp_f2.sh q2_old03 blab off;
LCA=0.45 GT=r9g_q2_old03 LT=both LT_EXTRA="--radius 5" LT_TAG=r5 bash scripts/r15/comp_f2.sh q2_old03 blab5 off ) & wait
J=/workspace/dub/r15f3/lipflick_w.jsonl; for t in s4p1500c8u_m03 s4p1500c8u_m03ltc s4p1500c8u_m03lac q2_m03fublab q2_old03blab q2_old03blab5; do GT_OVR=r9g_s4p1500c8u_m03 bash scripts/r15/lf.sh $t $J > /dev/null 2>&1 & done; wait
echo LT3_DONE $(date +%T)

cd /workspace/dub/dubai-f
GT=r9g_s4p1500c8u_m03 LT=color bash scripts/r15/comp_f2.sh s4p1500c8u_m03 ltc ia04
LCA=0.45 GT=r9g_s4p1500c8u_m03 LT=color bash scripts/r15/comp_f2.sh s4p1500c8u_m03 lac ia04
J=/workspace/dub/r15f3/lipflick_w.jsonl; for t in s4p1500c8u_m03 s4p1500c8u_m03ltc s4p1500c8u_m03lac; do bash scripts/r15/lf.sh $t $J > /dev/null 2>&1 & done; wait; echo ST_DONE $(date +%T)

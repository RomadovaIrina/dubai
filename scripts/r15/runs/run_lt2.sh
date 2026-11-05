cd /workspace/dub/dubai-f
m18(){ GT=r9g_q2_m18 LT=color bash scripts/r15/comp_f2.sh q2_m18 ltc ia04; LCA=0.45 GT=r9g_q2_m18 LT=color bash scripts/r15/comp_f2.sh q2_m18 blac off; LCA=0.45 GT=r9g_q2_m18 LT=color LT_EXTRA="--radius 5" LT_TAG=r5 bash scripts/r15/comp_f2.sh q2_m18 blac5 off; }
m03(){ LCA=0.45 GT=r9g_q2_m03fu LT=color LT_EXTRA="--radius 5" LT_TAG=r5 bash scripts/r15/comp_f2.sh q2_m03 fublac5 off; }
o03(){ LCA=0.45 GT=r9g_q2_old03 LT=color LT_EXTRA="--radius 5" LT_TAG=r5 bash scripts/r15/comp_f2.sh q2_old03 blac5 off; }
m18 & m03 & o03 & wait; echo LT2_DONE $(date +%T)

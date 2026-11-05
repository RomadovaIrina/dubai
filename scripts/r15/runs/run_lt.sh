cd /workspace/dub/dubai-f
run(){ id=$1; GT=$2; sx=$3; lab=$4; lt=$5; shift 5; env "$@" LT=$lt GT=$GT bash scripts/r15/comp_f2.sh $id $sx $lab; }
run q2_m03 r9g_q2_m03fu fultc ia04 color & run q2_m03 r9g_q2_m03fu fultm ia04 median & run q2_m03 r9g_q2_m03fu fublac off color LCA=0.45 & run q2_m03 r9g_q2_m03fu fublam off median LCA=0.45 & wait
run q2_old03 r9g_q2_old03 ltc ia04 color & run q2_old03 r9g_q2_old03 ltm ia04 median & wait
run q2_old03 r9g_q2_old03 blac off color LCA=0.45 & run q2_old03 r9g_q2_old03 blam off median LCA=0.45 & wait
echo LT_DONE $(date +%T)

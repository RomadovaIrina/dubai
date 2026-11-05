cd /workspace/dub/dubai-f; J=/workspace/dub/r15f3/lipflick_w.jsonl; : > $J
for t in q2_old03c0 q2_old03ltc q2_old03blac q2_old03blam q2_m03fu q2_m03fultc q2_m03fublac q2_m03fublam q2_m18c0 q2_m18bla; do bash scripts/r15/lf.sh $t $J > /dev/null 2>&1 & done; wait
until grep -q LT2_DONE /workspace/dub/r15f3/run_lt2.out; do sleep 30; done
for t in q2_m18ltc q2_m18blac q2_m18blac5 q2_m03fublac5 q2_old03blac5; do bash scripts/r15/lf.sh $t $J > /dev/null 2>&1 & done; wait; echo LFW_DONE

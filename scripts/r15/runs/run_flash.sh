cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/r15f3/flash; mkdir -p $O
run(){ id=$1; t=$2; ST=$3; SRC=$(ls /workspace/dub/r14d/eval/$id/e2e/work/in/$id.* | head -1); V=$(ls $B/r9f_$t/*_r9f_$t.mp4 | head -1)
  CUDA_VISIBLE_DEVICES= nice -n 19 python scripts/quality/r3_flash.py --video $V --src $SRC --lm $B/$ST/timing/${id}_source_lm_F.npz --out $O/$t.flash.json > /dev/null 2>&1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); s=sum(1 for e in d[\"flash_list\"] if (e.get(\"mc_ratio\") or 0)>=2 and (e.get(\"mc_vs_src\") or 0)>=2); print(sys.argv[2].ljust(22), \"flash_frames\", d[\"flash_frames\"], \"strong\", s)" $O/$t.flash.json $t; }
for t in q2_m03fu q2_m03fub q2_m03fubla q2_m03fublac q2_m03fublam q2_m03fultc q2_m03fublab5 q2_m03fublab5o3 q2_m03fublab5sp; do run q2_m03 $t r9g_q2_m03fu; done
run q2_m03 s4p1500c8u_m03 r9g_s4p1500c8u_m03; run q2_m03 s4p1500c8u_m03lac r9g_s4p1500c8u_m03; run q2_m03 s4p1500c8u_m03ltc r9g_s4p1500c8u_m03
for t in q2_old03c0 q2_old03bla q2_old03blac q2_old03blab5; do run q2_old03 $t r9g_q2_old03; done
for t in q2_m18c0 q2_m18bla q2_m18blac5 q2_m18blab5; do run q2_m18 $t r9g_q2_m18; done
echo FLASH_DONE

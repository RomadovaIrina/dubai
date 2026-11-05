cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/r15f3/tf; mkdir -p $O
for x in q2_m03:fu q2_m03:fultc q2_m03:fublac q2_m03:fublam q2_old03:c0 q2_old03:ltc q2_old03:blac q2_old03:blam; do IFS=: read id t <<< "$x"; ST=r9g_$id; [ $id = q2_m03 ] && ST=r9g_q2_m03fu
  SRC=$(ls /workspace/dub/r14d/eval/$id/e2e/work/in/$id.* | head -1); SL=$B/$ST/timing/${id}_source_lm_F.npz; V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4
  ( python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $SL --label $id$t > $O/$id$t.txt 2>&1; python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label $id$t >> $O/$id$t.txt 2>&1 ) & done; wait
grep -h "^==" $O/*.txt > /workspace/dub/r15f3/tone_pop.txt; echo TF_DONE

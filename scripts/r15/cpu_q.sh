# r15 F: CPU queue: old03 + m18 control / fix composites, tone_flicker + mouth_pop on all 6, the m03 clip and clips at every faded run
cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/outbox/r15_f; L=/workspace/dub/r15f/logs
until grep -q COMP_M03_DONE $L/m03.out; do sleep 15; done
for r in q2_old03 q2_m18; do
  bash scripts/r15/comp_f.sh $r c0 rep2s0 "" & bash scripts/r15/comp_f.sh $r cf rep2sc "--interp-fb-err 0.03" & wait; echo "== comp $r done $(date +%T)"
done
for r in q2_m03 q2_old03 q2_m18; do SRC=$(ls /workspace/dub/r14d/eval/$r/e2e/work/in/$r.* | head -1); SL=$B/r9g_$r/timing/${r}_source_lm_F.npz
  for t in c0 cf; do V=$B/r9f_$r$t/${r}_r9f_$r$t.mp4
    python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $SL --label "$r $t" | tee -a /workspace/dub/r15f/flicker.txt
    python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label "$r $t" | tee -a /workspace/dub/r15f/flicker.txt; done
  W=$(python - $B/r9g_$r/work_${r}_r9g_$r/${r}_r9g_${r}_25fps_ls_cfd_ia04_srt1_rep2sc.repair.json <<'PY'
import json, sys
f = json.load(open(sys.argv[1]))["motion_faded"]; ts = sorted({round(b / 25, 2) for b, *_ in f})
print(",".join(f"{max(0, t - 0.6):.2f}-{t + 0.7:.2f}" for t in ts))
PY
); echo "== $r faded windows $W"
  [ "$r" = q2_m03 ] && W="17-21${W:+,$W}"
  [ -n "$W" ] && bash scripts/r15/clip_f.sh $r $B/r9f_${r}c0/${r}_r9f_${r}c0.mp4 "before" $B/r9f_${r}cf/${r}_r9f_${r}cf.mp4 "after" "$W" $O/${r#q2_}_before_vs_after_mouth05x.mp4
done
echo CPU_Q_DONE

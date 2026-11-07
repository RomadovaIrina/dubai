#!/bin/bash
# r14 end of night on r14-d (the server is destroyed after NIGHT_DONE): tables, report, commit, git bundle, NIGHT_DONE last.
# The session's own NIGHT_REPORT.md (analysis + Russian summary) is used when it arrives within WAIT_MIN; otherwise the auto tables + notes.
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r14d; D=reports/experiments/r14-distill; mkdir -p $D
python scripts/r14/night_table.py --root $X > $X/NIGHT_TABLE.md 2>&1
for r in r2_s4 r3_s4 r3b_s4; do [ -s $X/ckpt/$r/curve.png ] && cp $X/ckpt/$r/curve.png $D/curve_$r.png; [ -s $X/ckpt/$r/ckpt_val.json ] && cp $X/ckpt/$r/ckpt_val.json $D/ckpt_val_$r.json; done
w=0; until [ -s $X/REPORT_READY ] || [ $w -ge ${WAIT_MIN:-60} ]; do sleep 60; w=$((w + 1)); done
if [ ! -s $X/REPORT_READY ]; then
  { echo "# r14 student — night report (auto: the session's write-up did not arrive)"; echo; cat $X/NIGHT_TABLE.md; echo; echo "## Notes"; echo '```'; cat $X/NIGHT_NOTES.txt 2>/dev/null; echo '```'; } > $X/NIGHT_REPORT.md
fi
cp $X/NIGHT_REPORT.md $D/NIGHT_REPORT.md; cp $X/NIGHT_TABLE.md $D/NIGHT_TABLE.md; mkdir -p scripts/r14/chains; cp $X/after_s4b.sh $X/r3_chain.sh $X/night2.sh scripts/r14/chains/ 2>/dev/null
for p in scripts/r13/distill/distill2.py scripts/r14/ckpt_val.py scripts/r14/viz_ckpts.py scripts/r14/eval_r14.sh scripts/r14/mouth_open.py scripts/r14/lipbox.py scripts/r14/night_table.py scripts/r14/night_finalize.sh \
  scripts/quality/candidate_video_stage.py scripts/quality/ls_stable.py scripts/r12/probe_seg.py scripts/r12/ref_smart.py scripts/r12/event_diff.py scripts/r12/top_windows.py \
  scripts/r14/chains $D/NIGHT_REPORT.md $D/NIGHT_TABLE.md $D/curve_*.png $D/ckpt_val_*.json $D/*.md; do [ -e "$p" ] && git add "$p"; done   # one by one: a missing path must not unstage the rest
git commit -q -m "r14 student round 3: temporal term off the lips, mouth-opening hinge, mouth_open metric, night report; agent F night smart reference (0008)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "commit $(git rev-parse --short HEAD)"
# incremental (coordinator: --all is ~2.2 GB): only what experiment/r14-distill adds on top of origin r13-spec + r13 (f547003 / d21b51c)
git bundle create $X/r14-distill.bundle experiment/r14-distill ^origin/experiment/r13-spec ^origin/experiment/r13 \
  && git bundle verify $X/r14-distill.bundle && echo "bundle ok $(du -h $X/r14-distill.bundle | cut -f1)" || echo "BUNDLE FAILED" | tee -a $X/NIGHT_NOTES.txt
cp $D/DATA_SOURCES.md $X/DATA_SOURCES.md 2>/dev/null
touch $X/NIGHT_DONE; echo "NIGHT_DONE $(date +%T)"

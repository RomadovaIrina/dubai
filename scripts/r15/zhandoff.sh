#!/bin/bash
# r15 W wrap-up: /workspace/dub/HANDOFF_W = FINAL_REPORT.md + bundles + configs / evidence + videos + ckpt + MANIFEST.txt
set -u
R=/workspace/dub/dubai; X=/workspace/dub/r15w/eval; H=/workspace/dub/HANDOFF_W; RD=$R/reports/experiments/r15-wild
mkdir -p $H/{bundles,configs/env,configs/timing,configs/eval,configs/zstage,configs/audit,tables,videos/panels,videos/finals,ckpt} $RD/zstage $RD/audit
cd $R
python3 scripts/r15/zreport_table.py > $H/tables/ZTABLE_FINAL.md
cat /workspace/dub/r15w/report_head.md $H/tables/ZTABLE_FINAL.md /workspace/dub/r15w/report_tail.md > $H/FINAL_REPORT.md
# ---- repo: report + tables + evidence, commit
cp $H/FINAL_REPORT.md $RD/Z_FINAL_REPORT.md; cp $H/tables/ZTABLE_FINAL.md $RD/ZTABLE_FINAL.md
cp /workspace/dub/r15w/zstage/*.json /workspace/dub/r15w/zstage/*.txt $RD/zstage/ 2>/dev/null; cp /workspace/dub/r15w/audit/*.json $RD/audit/ 2>/dev/null
cp $X/lipflick.jsonl $X/flash.jsonl $X/explode.jsonl $RD/ 2>/dev/null
git add $RD && git commit -q -F /workspace/dub/r15w/zmsg4.txt; HEAD=$(git rev-parse --short HEAD); echo "head $HEAD"
sed -i "s/__HEAD__/$HEAD/" $H/FINAL_REPORT.md
# ---- bundles (verified)
git bundle create $H/bundles/r15-wild-z.full.bundle experiment/r15-wild-z 2>&1 | tail -1
if [ $(stat -c %s $H/bundles/r15-wild-z.full.bundle) -gt 629145600 ]; then echo "full bundle $(du -h $H/bundles/r15-wild-z.full.bundle | cut -f1) > 600 MB: removed (the base is on GitHub; use the incremental bundle)"; rm -f $H/bundles/r15-wild-z.full.bundle; echo "full bundle skipped: > 600 MB (repo .git 2.1 GB); base commits c1740ce / 54b8509 / f74f9b4 are on GitHub" > $H/bundles/FULL_BUNDLE_SKIPPED.txt; fi
git bundle create $H/bundles/r15-wild-z.incr.bundle c1740ce..experiment/r15-wild-z 2>&1 | tail -1
for b in $H/bundles/*.bundle; do git bundle verify $b > /dev/null 2>&1 && echo "verified $(basename $b)" || echo "VERIFY FAILED $b"; done
git log --oneline c1740ce..HEAD > $H/bundles/commits_since_c1740ce.txt; echo "experiment/r15-wild-z $(git rev-parse HEAD)" > $H/bundles/HEAD.txt
# ---- configs / evidence
cp /workspace/dub/setup_z.sh scripts/pilot/clean_face_stack_calib.json $H/configs/ 2>/dev/null
for d in $X/*_*/; do n=$(basename $d); [ -f $d/logs/env.txt ] && cp $d/logs/env.txt $H/configs/env/$n.env.txt; [ -f $d/timing.json ] && cp $d/timing.json $H/configs/timing/$n.timing.json; done
cp $X/*.jsonl $X/*.sync.txt $X/*.flash.json $X/*.explode.json $X/ZTABLE.md $H/configs/eval/ 2>/dev/null
cp /workspace/dub/r15w/zstage/* $H/configs/zstage/ 2>/dev/null; cp /workspace/dub/r15w/audit/*.json $H/configs/audit/ 2>/dev/null
# ---- videos: every eye panel of the round + the key m03 finals (full length)
cp /workspace/dub/outbox/r15_w/*.mp4 $H/videos/panels/ 2>/dev/null
for a in q2d q2g15 q2g15lk q2ts25 q2ts25lk q2g125 q2g125lk; do [ -s $X/${a}_m03/${a}_m03.mp4 ] && cp $X/${a}_m03/${a}_m03.mp4 $H/videos/finals/; done
for a in q2d q2v2 q2cfa q2nb; do [ -s $X/${a}_m18/${a}_m18.mp4 ] && cp $X/${a}_m18/${a}_m18.mp4 $H/videos/finals/; done
# ---- checkpoint (student LoRA used for s4c8)
cp /workspace/dub/r15w/ckpt/*.pt $H/ckpt/ 2>/dev/null
cd $H && find . -type f -printf "%s %p\n" | sort -k2 > MANIFEST.txt
echo "HANDOFF $H $(du -sh $H | cut -f1) files $(wc -l < MANIFEST.txt)"

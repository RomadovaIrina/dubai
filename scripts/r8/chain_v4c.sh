#!/bin/bash
# r8v4c = r8v4b takes (r8v4bsrc) + the median-ambience mix floor + the final soften; same chain flags as scripts/r8/r8_audio.sh
cd /workspace/dub/dubai; A=/workspace/dub/artifacts/r4a; T=r8v4c
FINAL_SOFTEN=on RETAKE=off REUSE=r8v4bsrc R7_PREV=r8repro2 LIP_ARGS="--max-shift 2.0 --grow-unit 1.5 --grow 0.5 --pace on --cover-all on --dsplit off" \
  REMIX_ARGS="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on --gap-clean-floor 3" \
  bash scripts/r7/r7_audio.sh $T 04 05 02 01 03
. scripts/env.sh >/dev/null; mkdir -p $A/$T/cov
for id in 01 02 03 04 05; do
  python scripts/r8/coverage.py --run $A/$T/$id --id $id --label $T --face $A/sep/$id/source_mediapipe.npz --out $A/$T/cov/cov_$id 2>/dev/null | tail -1
  python scripts/r8/pacing.py --run $A/$T/$id --id $id --label $T --face $A/sep/$id/source_mediapipe.npz --out $A/$T/cov/pace_$id 2>/dev/null | tail -1
  python scripts/r8/cuts.py --run $A/$T/$id --id $id --label $T --out $A/$T/cov/cuts_$id 2>/dev/null | tail -1
  python scripts/r8/mixdrop.py --run $A/$T/$id --id $id --label $T --out $A/$T/cov/mix_$id 2>/dev/null | tail -1
done
echo CHAIN_V4C_DONE

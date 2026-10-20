#!/bin/bash
# Round 4 V queue 1: as each v4-repro video lands -> one-layer-off ablation composites (CPU) -> gate scores of v4-repro, the v4 reference
# (manifest borrowed from v4-repro) and every arm.
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4v
for id in 04 03 05 01 02; do
  until [ -f $B/v4repro/${id}_v4repro.manifest.json ] && grep -q e2e_wall_s $B/v4repro/${id}_v4repro.manifest.json; do sleep 30; done
  bash scripts/r4/r4_ablate.sh $id
  python scripts/r4/r4_gate.py score --label v4repro --dir $B/v4repro --ids $id
  python scripts/r4/r4_gate.py score --label v4ref --dir $B/ref --manifest-from $B/v4repro:v4repro --ids $id
  for arm in no_retime no_sanitize no_sharpen no_colour no_skin no_faceclip no_seamlf no_lipguard; do python scripts/r4/r4_gate.py score --label $arm --dir $B/abl --ids $id; done
  echo "== Q1 $id done"
done
echo Q1_DONE

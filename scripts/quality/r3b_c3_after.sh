#!/bin/bash
# r3 B: after each audio-r3 render (gen/<id>_c3F) -> R3 composite comp/<id>_c3_R3 + eval + SyncNet -> outbox <id>_full_R3face_audio-C-r3.mp4
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; O=/workspace/dub/outbox; PY=/venv/dabai/bin/python
R3FLAGS="--colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on"
for id in "$@"; do
  until grep -q "^== $id c3F\|^FAIL $id c3F" $R/gen/${id}_c3F.log 2>/dev/null; do sleep 20; done
  if [ $id = 02 ]; then   # no face: source frames + C's mix at the source frame rate
    $PY scripts/quality/fps_composite.py --manifest $R/gen/02_c3F.manifest.json --ls-raw $R/gen/02_c3F.mp4 --out $R/comp/02_c3_R3.mp4 --threads 12 | tail -1
  else
    rm -f $R/timing/${id}_master_timing_F.json.c3; GEN=c3F scripts/quality/r3_compF.sh $id c3_R3 $R3FLAGS
    scripts/quality/r3_sync.sh $id:c3_R3 $id:c3F
  fi
  cp $R/comp/${id}_c3_R3.mp4 $O/${id}_full_R3face_audio-C-r3.mp4; echo "$id c3 done $(date +%H:%M)"
done

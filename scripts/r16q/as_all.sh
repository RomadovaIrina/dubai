#!/bin/bash
# r16 Q: active-speaker check for every clip, in parallel on the CPU
cd /workspace/dub/dubai; A=/workspace/dub/artifacts/r4a; O=/workspace/dub/r16q/as
one(){ AS_THREADS=24 nice /venv/dabai/bin/python scripts/quality/r16q_active_speaker.py --video $2 --manifest $3 --out $O/$1.json --device cpu > $O/$1.log 2>&1; echo "## $1 $(tail -1 $O/$1.log | cut -c1-400)"; }
one m03 $O/m03_25.mp4 $A/r9_t16_m03/t16_m03/t16_m03_r4b.manifest.json &
one q07_altapress $O/q07_25.mp4 $(ls $A/r9_q07_altapress/q07_altapress/*manifest.json) &
for id in q01_amir q02_vera q03_priest q04_farikh q05_rozhkov q06_suvorov q08_albats; do one $id /workspace/dub/r16q/ks/${id}_src/master_25fps.mp4 $(ls $A/r9_$id/$id/*manifest.json) & done
wait; echo AS_ALL_DONE

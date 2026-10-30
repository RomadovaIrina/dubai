#!/bin/bash
# r3_flash on each face layer of one video on the 25 fps master (raw LatentSync / +CodeFormer / +detail+SR / +fx)
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export OMP_NUM_THREADS=6; id=$1; W=/workspace/dub/artifacts/r4m/r9g_$id/work_${id}_r9g_$id; S=$W/video/master_25fps.mp4; LM=$W/video/master_mediapipe.npz
for L in ls_pre ls_raw ls_cfd_cfd_srt1 ls_cfd_ia04_srt1_rep2; do [ -f $W/${id}_r9g_${id}_25fps_$L.mp4 ] || continue
  python scripts/quality/r3_flash.py --video $W/${id}_r9g_${id}_25fps_$L.mp4 --src $S --lm $LM --out /workspace/dub/r9/diag/flash_${id}_$L.json > /dev/null 2>&1
  python3 - /workspace/dub/r9/diag/flash_${id}_$L.json $L <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); L = d["flash_list"]
k = lambda e: max((abs(e.get("z_hf", 0)), "texture"), (abs(e.get("z_lum", 0)), "brightness"), (abs(e.get("z_col", 0)), "colour"))[1]
print(sys.argv[2], "flash", len(L), {n: sum(1 for e in L if k(e) == n) for n in ("texture", "brightness", "colour")})
PY
done; echo FL_DONE

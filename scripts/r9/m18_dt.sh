#!/bin/bash
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off DUB_INPUTS=/workspace/dub/r9/runs/m18/work/in
B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag
SKIP_SCORE=1 TAG=r9f_m18fxdt SRC_TAG=r9g_m18 CFD=self CFD_LABEL=a05t2 CFD_ARGS="--inner-alpha 0.4 --alpha 0.5 --temporal 2" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" REP=1 bash scripts/r7/r7_m7_comp.sh m18 > $D/m18_dt.log 2>&1
python scripts/quality/r3_flash.py --video $B/r9f_m18fxdt/m18_r9f_m18fxdt.mp4 --src /workspace/dub/r9/runs/m18/work/in/m18.mp4 --lm $B/r9g_m18/timing/m18_source_lm_F.npz --out $D/flash_r9f_m18fxdt.json > /dev/null 2>&1
python3 - <<'PY'
import json
k = lambda e: max((abs(e.get("z_hf", 0)), "texture"), (abs(e.get("z_lum", 0)), "brightness"), (abs(e.get("z_col", 0)), "colour"))[1]
for T in ["r9f_m18", "r9f_m18fx", "r9f_m18fxdt"]:
    d = json.load(open(f"/workspace/dub/r9/diag/flash_{T}.json")); L = d["flash_list"]; s = [e for e in L if (e.get("mc_ratio") or 0) >= 2 and (e.get("mc_vs_src") or 0) >= 2]
    print(T, "flash", len(L), "strong", len(s), {n: sum(1 for e in L if k(e) == n) for n in ("texture", "brightness", "colour")})
PY
echo M18DT_DONE

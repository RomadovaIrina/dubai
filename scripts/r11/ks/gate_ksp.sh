cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; R=/workspace/dub/r10/gate; W=/workspace/dub/artifacts/r4m/r9g_old03/work_old03_r9g_old03
[ -s $R/old03base/face_gate.json ] || bash scripts/r10/gate.sh old03 "" > /tmp/g_old03base.log 2>&1
[ -s $R/old03Zk5/face_gate.json ] || bash scripts/r10/gate.sh old03 Zk5 $W/old03_r9g_old03_25fps_k5.mp4 > /tmp/g_old03k5.log 2>&1
bash scripts/r10/gate.sh old03 Zkssph /workspace/dub/r11/ksp/qo3_sp_fixh.mp4 > /tmp/g_old03ks.log 2>&1
python scripts/r10/gate_cmp.py --base $R/old03base --cand $R/old03Zk5 --md /workspace/dub/r11/gate_ks.md | head -3
python scripts/r10/gate_cmp.py --base $R/old03base --cand $R/old03Zkssph --md /workspace/dub/r11/gate_ks.md | head -3
python scripts/r10/gate_cmp.py --base $R/old03Zk5 --cand $R/old03Zkssph --md /workspace/dub/r11/gate_ks.md | head -3
python scripts/r10/face_metrics.py --video $W/old03_r9g_old03_25fps_k5.mp4 --src /workspace/dub/r10/runs/old03/work/in/old03.mp4 --lm-src /workspace/dub/artifacts/r4m/r9g_old03/timing/old03_source_lm_F.npz --out /workspace/dub/r11/ksp/old03_k5.json > /dev/null 2>&1
python3 -c "
import json; j=json.load(open(\"/workspace/dub/r11/ksp/old03_k5.json\")); r=j[\"flicker\"]; c=j.get(\"chin_flow\") or {}
print(\"old03 k5 ring\", r[\"ring_bright_ratio\"], r[\"ring_bright_strong\"], \"swim\", c.get(\"swim_rms\"), \"ev\", c.get(\"events\"), c.get(\"strong\"))"
echo GATE_KS_DONE

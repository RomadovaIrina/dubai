#!/bin/bash
# r16 K2: full-length finals = ref_teeth (k2_refpick) + b8 gate KeySync + K1's chain (k1/r16-paste 5c07cf7: pad60 without the duplicate seam
# frame, 40 ms delayed dub, ECC align, tsmooth gate:0.5, snap, guard, plain paste --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2, tex 2)
# + optionally the pause arm (b) (k2_pause.py warp on the guarded layer). Run dir /workspace/dub/k1run (K1's tools copied there).
#   k2_k1chain.sh <v> <ref_idx>
v=$1; REF=$2; D=/workspace/dub/k1run; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; R=/workspace/dub/r16k2; MP=/venv/mediapipe/bin/python; DP=/venv/dabai/bin/python
cd $D; mkdir -p $v; W=$D/$v
N=$(python3 -c "import numpy as np; print(len(np.load('$P/$v/source_lm_F.npz')['j']))")
# 1. pad60 crop (the last 60 frames mirrored, the last frame NOT repeated) + the 40 ms delayed dub padded with silence
$MP - <<PY
import cv2, subprocess
c = cv2.VideoCapture("$R/src/${v}_crop512.mp4"); F = []
while True:
    ok, f = c.read()
    if not ok: break
    F.append(f)
F = F + F[-2:-62:-1]
w = cv2.VideoWriter("$W/pad60.avi", cv2.VideoWriter_fourcc(*"FFV1"), 25, (512, 512)); [w.write(f) for f in F]; w.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "$W/pad60.avi", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "$W/pad60.mp4"], check=True); subprocess.run(["rm", "$W/pad60.avi"])
print("pad60", len(F), "frames")
PY
DUR=$(python3 -c "print(($N + 60) / 25)")
ffmpeg -v error -y -i $P/$v/dubbed_16k.wav -af "adelay=40:all=1,apad=whole_dur=$DUR" -t $DUR -ar 16000 -ac 1 $W/dub_d40_pad.wav
# 2. KeySync: ref_teeth + b8 gate
echo "$v $W/pad60.mp4 $W/dub_d40_pad.wav" > $W/jobs.txt
( cd /workspace/dub/keysync/code; . ../venv/bin/activate; flock /workspace/dub/gpu.lock env KS_REF_IDX=$REF KS_GATE="thr=0.12,cap=0.25,hi=8,lo=2" KS_DC=2:0 KS_LMK_CACHE=/workspace/dub/keysync/lmk \
  python ../ks_fast.py --jobs $W/jobs.txt --out $W/ks --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s --interp-lora /workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt --interp-steps 2 --vae-batch 8 --vae-compile > $W/ks.log 2>&1 )
grep -E "KS_JOB|KS_GATE hi" $W/ks.log | cut -c1-110 | sed "s/^/$v /"
RAW=$(ls $W/ks/$v/*.mp4 | head -1)
# 3. K1 chain
$MP /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $RAW --out $W/raw.mp.npz > /dev/null 2>&1
$DP k_ecc.py --ks $RAW --src $W/pad60.mp4 --lm $P/$v/source_crop512.mediapipe.npz --out $W/ecc.npz > /dev/null
$DP ks_tsmooth.py --layer $RAW --lm $W/raw.mp.npz --src-lm $P/$v/source_crop512.mediapipe.npz --out $W/ts.mp4 --alpha gate:0.5 2>&1 | grep gate: | sed "s/^/$v /"
$DP ks_snap.py --layer $W/ts.mp4 --lm $W/raw.mp.npz --out $W/snap.mp4 | grep ks_snap | sed "s/^/$v /"
$DP ks_guard.py --layer $W/snap.mp4 --src $W/pad60.mp4 --n $N --out $W/guard.mp4 | tail -1 | cut -c1-200 | sed "s/^/$v guard /"
# 4. pause arm (b) on the guarded layer (the layer is already on the dub timeline: shift 0 vs the 40 ms delayed dub)
$MP $R/k2_pause.py --mode warp --ks $W/guard.mp4 --ks-lm $W/raw.mp.npz --wav $P/$v/dubbed_16k.wav --shift 0 --out $W/guard_pb.mp4 | sed "s/^/$v /"
OTH=""; [ $v = m03 ] && OTH="--others $P/m03/faces2.npz --others-grow 1.3"
for arm in guard guard_pb; do
  KS_CMS=1 $DP ks_plain.py --ks $W/$arm.mp4 --ks-lm $W/raw.mp.npz --master $R/src/${v}_master_25fps.mp4 --box $P/$v/box.json --lm $P/$v/source_lm_F.npz --audio $P/$v/final_mix_aac.m4a --shift 0 \
    --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $W/ecc.npz --align-scale 0 --tex 2 $OTH --out $W/final_$arm.mp4 2>&1 | grep -E "WARN|Trace|Error" | sed "s/^/$v $arm /"
  bash k_gate.sh $W/final_$arm.mp4 $P/$v/dubbed_16k.wav $W/final_$arm.sync.json | grep -o "speech_at0[^,]*" | sed "s/^/$v $arm /"
done
# 5. last-frame check: lower-face change of each of the last 30 frames vs the median change
$MP - <<PY
import cv2, numpy as np, json
b = json.load(open("$P/$v/box.json")); x, y, s = b["x"], b["y"], b["side"]
for arm in ("guard", "guard_pb"):
    c = cv2.VideoCapture("$W/final_" + arm + ".mp4"); n = int(c.get(7)); G = []
    for k in range(n):
        ok, f = c.read()
        if not ok: break
        G.append(cv2.cvtColor(f[y + s // 2:y + s, x:x + s], cv2.COLOR_BGR2GRAY).astype(np.float32))
    d = np.array([np.abs(G[k] - G[k - 1]).mean() for k in range(1, len(G))]); med = float(np.median(d))
    print("$v", arm, "frames", len(G), "median lower-face change %.2f, last-30 max %.2f (x%.1f)" % (med, d[-30:].max(), d[-30:].max() / max(med, 1e-6)))
PY
echo "CHAIN_DONE $v"

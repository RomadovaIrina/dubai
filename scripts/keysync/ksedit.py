p = "/workspace/dub/keysync/ks_final.sh"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep("""# the face agent's ks_fix.py (r10-face scripts/r11) --guard 3 --chin tex; raw square paste back to the source frames; the demucs final mix.
#   ks_final.sh <id> <source video>  -> /workspace/dub/outbox/r12_ks/final/<name>_KeySync.mp4""",
"""# the face agent's ks_fix.py (r10-face scripts/r11) --guard 3 --chin tex; the lag fix ks_tsmooth.py (alpha KS_TSMOOTH, default 0.8, 0 = off);
# raw square paste back to the source frames; the demucs final mix.
#   ks_final.sh <id> <source video> [name]  -> $KS_OUTDIR/<name>_KeySync.mp4 (default /workspace/dub/outbox/r12_ks/final_v2)
# the stages already on disk (KeySync output, raw sync, layer, layer_fix) are reused; KS_REDO=1 rebuilds the layer + fix.""")
rep("O=$K/final/$id; mkdir -p $O /workspace/dub/outbox/r12_ks/final",
    "O=$K/final/$id\nTS=${KS_TSMOOTH:-0.8}; OUT=${KS_OUTDIR:-/workspace/dub/outbox/r12_ks/final_v2}; mkdir -p $O $OUT")
rep("python $K/ks_layer.py", "[ -s $O/layer_fix.mp4 ] && [ -z \"$KS_REDO\" ] || {\npython $K/ks_layer.py")
rep("--guard-report $O/guard.json\n", """--guard-report $O/guard.json; rm -f $O/layerfix.mediapipe.npz; }
# 3b) the lag fix: flow-aligned temporal smoothing of the mouth region (accepted by eye on m18 at alpha 0.8)
L=$O/layer_fix.mp4
if [ "$TS" != 0 ]; then
  [ -s $O/layerfix.mediapipe.npz ] || /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $O/layer_fix.mp4 --out $O/layerfix.mediapipe.npz | tail -1
  python $K/ks_tsmooth.py --layer $O/layer_fix.mp4 --lm $O/layerfix.mediapipe.npz --out $O/layer_ts.mp4 --alpha $TS; L=$O/layer_ts.mp4
fi
""")
rep("ffmpeg -v error -y -i $O/layer_fix.mp4 -vf", "ffmpeg -v error -y -i $L -vf")
rep("/workspace/dub/outbox/r12_ks/final/${NAME}_KeySync.mp4", "$OUT/${NAME}_KeySync.mp4")
open(p, "w").write(s)
print("EDITED")

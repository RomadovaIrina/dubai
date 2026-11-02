p = "/workspace/dub/keysync/ks_final.sh"
s = open(p).read()
a = "  python $K/ks_tsmooth.py --layer $O/layer_fix.mp4 --lm $O/layerfix.mediapipe.npz --out $O/layer_ts.mp4 --alpha $TS; L=$O/layer_ts.mp4"
b = """  SRCLM=""; if [ "$TS" = auto ]; then [ -s $O/master.mediapipe.npz ] || /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $MASTER --out $O/master.mediapipe.npz | tail -1; SRCLM="--src-lm $O/master.mediapipe.npz"; fi
  python $K/ks_tsmooth.py --layer $O/layer_fix.mp4 --lm $O/layerfix.mediapipe.npz --out $O/layer_ts.mp4 --alpha $TS $SRCLM; L=$O/layer_ts.mp4"""
assert a in s; s = s.replace(a, b)
s = s.replace("the lag fix ks_tsmooth.py (alpha KS_TSMOOTH, default 0.8, 0 = off);", "the lag fix ks_tsmooth.py (alpha KS_TSMOOTH, default 0.8, 0 = off, auto = adaptive: only where KeySync jitters more than the source);")
open(p, "w").write(s); print("EDITED2")

cd /workspace/dub/r16k2; R=/workspace/dub/r16k2; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; MP=/venv/mediapipe/bin/python
for v in m03 old03; do K=$R/gen/rg8/$v/${v}_crop512_$v.mp4; SH=$(python3 -c "import json; print(-int(json.load(open(\"$P/$v/raw_sync.json\"))[\"global\"][\"av_offset\"]))"); n=$(python3 -c "import numpy as np;print(len(np.load(\"$P/$v/source_lm_F.npz\")[\"j\"]))")
 $MP k2_pause.py --mode weights --src-lm $P/$v/source_lm_F.npz --wav $P/$v/dubbed_16k.wav --n $n --out $R/$v/keep.npy
 $MP k2_pause.py --mode warp --ks $K --ks-lm $K.mp.npz --wav $P/$v/dubbed_16k.wav --shift $SH --out $R/$v/ks_closed.mp4
 KSLM=$K.mp.npz bash k2_var.sh $v rt $K & KSLM=$K.mp.npz PLAINARGS="--lip-keep $R/$v/keep.npy" bash k2_var.sh $v pa $K & KSLM=$K.mp.npz bash k2_var.sh $v pb $R/$v/ks_closed.mp4 & wait
 b=$(python3 -c "import json;d=json.load(open(\"$P/$v/box.json\"));print(f\"crop={d[\"side\"]}:{d[\"side\"]}:{d[\"x\"]}:{d[\"y\"]},scale=512:512:flags=lanczos\")")
 for m in rt pa pb; do ffmpeg -v error -y -i $v/$m/final.mp4 -vf $b -c:v libx264 -crf 12 $v/$m/fc.mp4; $MP /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $v/$m/fc.mp4 --out $v/$m/fc.npz > /dev/null 2>&1; done
 $MP - <<PY
import numpy as np, wave
def op(f):
    L=np.load(f)["lm"][...,:2]; return np.array([np.linalg.norm(p[13]-p[14])/max(np.linalg.norm(p[61]-p[291]),1) if np.isfinite(p).all() else np.nan for p in L])
w=wave.open("$P/$v/dubbed_16k.wav"); a=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768
X={"source":op("$P/$v/source_crop512.mediapipe.npz") if False else op("refpick/$v.mp.npz")}
for m in ("rt","pa","pb"): X[m]=op("$v/"+m+"/fc.npz")
n=min(len(x) for x in X.values()); db=np.array([20*np.log10(np.sqrt((a[i*640:(i+1)*640]**2).mean())+1e-9) for i in range(n)]); sil=db<db.max()-35
for k,x in X.items(): x=x[:n]; print("CLOSE $v",k,"closed-in-silence %.0f%%  closed overall %.0f%%  mean open %.3f"%(100*np.nanmean(x[sil]<0.03),100*np.nanmean(x<0.03),np.nanmean(x)))
PY
done; echo PAUSE_DONE

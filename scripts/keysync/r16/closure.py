import numpy as np, wave, sys
P="/workspace/dub/r16in/HANDOFF_K/paste_inputs"
def op(f):
    L=np.load(f)["lm"][...,:2]; return np.array([np.linalg.norm(p[13]-p[14])/max(np.linalg.norm(p[61]-p[291]),1) if np.isfinite(p).all() else np.nan for p in L])
for v in ("m03","old03","m18"):
    w=wave.open(f"{P}/{v}/dubbed_16k.wav"); a=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768
    X={"source":op(f"refpick/{v}.mp.npz")}
    for g in sys.argv[1:]: X[g]=op(f"gen/{g}/{v}/{v}_crop512_{v}.mp4.mp.npz")
    n=min(len(x) for x in X.values()); db=np.array([20*np.log10(np.sqrt((a[i*640:(i+1)*640]**2).mean())+1e-9) for i in range(n)]); sil=db<db.max()-35
    for lb,x in X.items(): x=x[:n]; print(v,lb,"mean open %.3f closed %.0f%% closed-in-silence %.0f%% wide %.0f%%"%(np.nanmean(x),100*np.nanmean(x<0.03),100*np.nanmean(x[sil]<0.03),100*np.nanmean(x>0.25)))

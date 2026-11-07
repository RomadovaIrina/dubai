import sys, numpy as np, cv2
# intflick.py lm.npz t0 t1 label=ks.mp4 ... : inner-mouth temporal change on open frames: mean |dL| and the 1-frame "zigzag" |L[t]-(L[t-1]+L[t+1])/2|
INNER=[78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
L=np.load(sys.argv[1])["lm"][...,:2]; t0,t1=float(sys.argv[2]),float(sys.argv[3]); k0,k1=int(t0*25),int(t1*25)
for lv in sys.argv[4:]:
    lb,v=lv.split("="); c=cv2.VideoCapture(v); c.set(1,k0); F=[cv2.cvtColor(c.read()[1],cv2.COLOR_BGR2LAB)[...,0].astype(np.float32) for _ in range(k1-k0)]
    d=[];z=[]
    for i in range(1,len(F)-1):
        p=L[k0+i]
        if np.linalg.norm(p[13]-p[14])/np.linalg.norm(p[61]-p[291])<0.15: continue
        m=np.zeros(F[0].shape,np.uint8); cv2.fillPoly(m,[np.round(cv2.convexHull(p[INNER].astype(np.float32))[:,0]).astype(np.int32)],1); m=cv2.erode(m,np.ones((5,5),np.uint8))>0
        d.append(np.abs(F[i]-F[i-1])[m].mean()); z.append(np.abs(F[i]-(F[i-1]+F[i+1])/2)[m].mean())
    print(lb, "open frames", len(d), "mean|dL|", round(float(np.mean(d)),2), "zigzag", round(float(np.mean(z)),2), "zigzag p95", round(float(np.percentile(z,95)),2))

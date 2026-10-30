#!/usr/bin/env python3
"""r9: the "mouth slides off" the user saw in the ref shift16 / fixed panels: the gate's slide / lower-face shear / chin-wave rows per panel."""
import json
B = "/workspace/dub/artifacts/r4m"
for i, fx in [("m03", "fx3"), ("m19", "fx"), ("m15", "fx")]:
    for lab, T in [("r9", f"r9f_{i}"), ("r9+fix", f"r9f_{i}{fx}"), ("shift16+fix", f"r9f_{i}shift16L05{fx}"), ("fixed+fix", f"r9f_{i}fixedL05{fx}")]:
        try: s = json.load(open(f"{B}/{T}/{i}_{T}.score.json"))
        except Exception: print(i, lab, "no score"); continue
        sl = {k: v for k, v in (s.get("slide") or {}).items() if not isinstance(v, (list, dict))}
        fv = s.get("face_v4") or {}; ch = s.get("chinwave") or {}
        print(f"{i} {lab:12s} slide {json.dumps(sl)[:260]} | shear95_lowerface {fv.get('shear95_lowerface_ls_core')} | chinwave mean {ch.get('mean')} worst2s {ch.get('worst_2s')}")

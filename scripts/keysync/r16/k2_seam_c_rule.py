"""Where does variant (c) (tight mask without the generated chin) lose sync? Per second: delta conf_at0 (c - now) from sync_main.json, vs the
generated jaw excess (raw KeySync lip gap - source lip gap, crop space, / mouth width) and the dub level. Then a per-second rule: keep the chin
(no (c)) where the jaw excess is high; use (c) elsewhere. Reports the share of seconds (c) would cover and the sync loss it would keep."""
import json, sys
import numpy as np

S = "/workspace/dub/k1/seam"; R = "/workspace/dub/k2/seamc"
def per_sec(path):
    s = json.load(open(path)); out = {}
    for p in s["per_second"]:
        if p.get("conf_at0") is not None: out[int(p.get("sec", p.get("t", 0)))] = (p["conf_at0"], p.get("dub_act") or 0)
    return out
def gap(lmf):
    L = np.load(lmf)["lm"][..., :2]; w = np.linalg.norm(L[:, 61] - L[:, 291], axis=1); return np.linalg.norm(L[:, 13] - L[:, 14], axis=1) / np.maximum(w, 1)
rows = []
for vid in ("q03a", "old03", "m03", "q04", "q01"):
    KW = f"{S}/w_{vid}/ks"; a = per_sec(f"{R}/ev/{vid}_now/sync_main.json"); c = per_sec(f"{R}/ev/{vid}_c/sync_main.json")
    gs = gap(f"{KW}/crop_lm_f0.npz"); gr = gap(f"{KW}/post_f0/raw_lm.npz")[:len(gs)]
    ex = np.nan_to_num(np.clip(gr - gs, 0, None)); secs = sorted(set(a) & set(c))
    d = np.array([c[t][0] - a[t][0] for t in secs]); act = np.array([a[t][1] for t in secs]); e = np.array([ex[t * 25:(t + 1) * 25].mean() if t * 25 < len(ex) else 0 for t in secs])
    sp = act >= 0.5
    rho = float(np.corrcoef(np.argsort(np.argsort(e[sp])), np.argsort(np.argsort(d[sp])))[0, 1]) if sp.sum() > 3 else float("nan")
    for thr in (0.03, 0.05):
        keep = e > thr   # chin kept (no (c)) where the generated jaw works
        loss_c_all = d[sp].mean(); loss_rule = np.where(keep, 0, d)[sp].mean()
        rows.append((vid, thr, round(float(rho), 2), round(float(loss_c_all), 3), round(float(loss_rule), 3), round(float((~keep).mean()), 2)))
    print(f"{vid}: speech secs {sp.sum()}, delta(c-now) mean {d[sp].mean():+.3f}; spearman(delta, jaw excess) {rho:+.2f}; delta on high-excess secs (>0.05) {d[sp & (e > 0.05)].mean() if (sp & (e > 0.05)).any() else float('nan'):+.3f}, on low {d[sp & (e <= 0.05)].mean() if (sp & (e <= 0.05)).any() else float('nan'):+.3f}")
print("video thr rho loss_c_everywhere loss_with_rule share_seconds_(c)")
for r in rows: print(*r)

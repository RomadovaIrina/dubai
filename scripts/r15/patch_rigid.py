# r15 F: motion-aware rigid track: anchor_filters.validate_motion + LS_RIGID_VALIDATE=<thr IOD> in clean_face_stack (default off). Run from the worktree root.
import pathlib
p = pathlib.Path("scripts/pilot/anchor_filters.py"); s = p.read_text()
old = '''def _apply(H, p):'''
new = '''RIGID_PTS = [1, 4, 5, 6, 168, 197, 195, 33, 133, 263, 362, 10, 151, 9]   # nose bridge / eye corners / forehead: rigid with the head


def validate_motion(H: np.ndarray, lm: np.ndarray, thr: float = 0.02, return_info: bool = False):
    """r15 F (m03 19 s, the doubled mouth): the DIS-flow rigid motion H[t] is checked against the landmarks' own motion on rigid points
    (nose bridge, eye corners, forehead). Where H moves those points away from where MediaPipe puts them by more than thr IOD (a second face,
    hair or a hand moving inside the upper-face hull; m03: 479-503, up to 0.10 IOD while the head is still), H[t] is blended towards the
    landmark similarity: weight 0 at thr, 1 at 2 thr (feathered). flow:S then no longer accumulates a spurious jump, which its Gaussian spreads
    over +-2 S frames of LatentSync's crop / paste-back (the generated face pasted at a shifted place = a second mouth on the cheek)."""
    import cv2
    H = np.asarray(H, np.float64).copy(); T = min(len(H), len(lm)); n_fix = 0; wsum = 0.0; res_max = 0.0
    for t in range(1, T):
        a, b = lm[t - 1, RIGID_PTS, :2].astype(np.float64), lm[t, RIGID_PTS, :2].astype(np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()): continue
        iod = float(np.linalg.norm(lm[t, 33, :2] - lm[t, 263, :2]))
        res = float(np.linalg.norm(_apply(H[t], a) - b, axis=1).mean()) / max(iod, 1e-6); res_max = max(res_max, res)
        if res <= thr: continue
        M, _ = cv2.estimateAffinePartial2D(a.astype(np.float32), b.astype(np.float32), method=cv2.LMEDS)
        if M is None: continue
        w = float(np.clip(res / thr - 1.0, 0.0, 1.0)); H[t] = (1 - w) * H[t] + w * M; n_fix += 1; wsum += w
    info = {"thr": thr, "frames_blended": n_fix, "weight_sum": round(wsum, 2), "residual_max": round(res_max, 4)}
    return (H, info) if return_info else H


def _apply(H, p):'''
assert old in s; s = s.replace(old, new, 1); p.write_text(s)

p = pathlib.Path("scripts/pilot/clean_face_stack.py"); s = p.read_text()
old = '''            if anchor_filter: lm_ls = af.filter_landmarks(lm, Hrig, anchor_filter).astype(np.float32)'''
new = '''            rv_ = float(os.environ.get("LS_RIGID_VALIDATE", "0") or 0)   # r15 F (default off): the rigid track checked against the landmarks (m03 19 s doubled mouth)
            if rv_ > 0: Hrig, self.anchor_info["rigid_validate"] = af.validate_motion(Hrig, lm, rv_, return_info=True); print(f"  [rigid] validate {self.anchor_info['rigid_validate']}", flush=True)
            if anchor_filter: lm_ls = af.filter_landmarks(lm, Hrig, anchor_filter).astype(np.float32)'''
assert old in s; s = s.replace(old, new, 1); p.write_text(s)
print("patched anchor_filters + clean_face_stack")

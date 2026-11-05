# r15 F (2): interpolation-confidence guard for the layer repair (--interp-fb-err, default 0 = off), on top of patch_motion.py
import pathlib
p = pathlib.Path("scripts/r9/r9_layer_repair.py"); s = p.read_text()
s = s.replace('''    ap.add_argument("--max-run", type=int, default=3);''', '''    ap.add_argument("--interp-fb-err", type=float, default=0.0, help="r15 F interpolation confidence (0 = off): the forward-backward error (p90, IOD) of the p<->q flow in the "
                    "mouth box; above it the redraw fades out (weight 1 at the threshold, 0 at 2x it) and the generated frames are kept. m03 19 s (the doubled mouth): 0.168; "
                    "the good repairs of old03 / m03 / m18: <= 0.01 mostly")
    ap.add_argument("--max-run", type=int, default=3);''', 1)
old = '''            if wm <= 0.0: continue   # fast move: keep the generated frames (the generator followed the move; the flags are stale-landmark false alarms)
            gp, gq = cv2.cvtColor(F[p], cv2.COLOR_BGR2GRAY), cv2.cvtColor(F[q], cv2.COLOR_BGR2GRAY)
            fpq, fqp = dis.calc(gp, gq, None), dis.calc(gq, gp, None)'''
new = '''            if wm <= 0.0: continue   # fast move: keep the generated frames (the generator followed the move; the flags are stale-landmark false alarms)
            gp, gq = cv2.cvtColor(F[p], cv2.COLOR_BGR2GRAY), cv2.cvtColor(F[q], cv2.COLOR_BGR2GRAY)
            fpq, fqp = dis.calc(gp, gq, None), dis.calc(gq, gp, None)
            if a.interp_fb_err > 0:   # r15 F: the two anchors do not map onto each other in the mouth (head move, mouth reshaped): interpolation would ghost
                f_ = src[(b + e) // 2] if np.isfinite(src[(b + e) // 2][[33, 263, 61, 291, 13, 14]]).all() else lm[(b + e) // 2]
                if np.isfinite(f_[[33, 263, 61, 291, 13, 14]]).all():
                    iod_ = np.linalg.norm(f_[33, :2] - f_[263, :2]); cx_, cy_ = f_[[61, 291, 13, 14], :2].mean(0); r_ = int(0.9 * iod_)
                    ys, xs = slice(max(0, int(cy_ - r_)), max(1, int(cy_ + r_))), slice(max(0, int(cx_ - r_)), max(1, int(cx_ + r_)))
                    back = cv2.remap(fqp, gx + fpq[..., 0], gy + fpq[..., 1], cv2.INTER_LINEAR)
                    fbe = float(np.percentile(np.linalg.norm((fpq + back)[ys, xs], axis=2), 90) / max(iod_, 1e-6))
                    wc = float(np.clip(2.0 - fbe / a.interp_fb_err, 0.0, 1.0))
                    if wc < 1.0: moved.append([b, e, "fb_err", round(fbe, 4), round(wc, 3)])
                    wm = min(wm, wc)
                if wm <= 0.0: continue'''
assert old in s; s = s.replace(old, new, 1)
old = '''    if a.motion_thr > 0: rp["motion"] = {"thr": a.motion_thr, "faded_runs": len(moved), "p95": round(float(np.percentile(mot, 95)), 4), "p99": round(float(np.percentile(mot, 99)), 4)}'''
new = '''    if a.motion_thr > 0: rp["motion"] = {"thr": a.motion_thr, "p95": round(float(np.percentile(mot, 95)), 4), "p99": round(float(np.percentile(mot, 99)), 4)}
    if a.motion_thr > 0 or a.interp_fb_err > 0: rp["faded_runs"] = len(moved); rp["interp_fb_err"] = a.interp_fb_err'''
assert old in s; s = s.replace(old, new, 1)
p.write_text(s); print("patched (2)")

# r15 F: motion-aware layer repair (--motion-thr, default 0 = off = the r13 behaviour). Run from the worktree root.
import pathlib
p = pathlib.Path("scripts/r9/r9_layer_repair.py"); s = p.read_text()

s = s.replace('''def main():''', '''def face_motion(frames, lm):
    """r15 F: per master frame k, the face's pixel motion from k-1 to k: p90 of the DIS flow magnitude inside the source face box (landmark bbox
    +20 %, half resolution), in IOD. From PIXELS on purpose: MediaPipe VIDEO-mode landmarks lag on a fast head move (m03 19 s: the nose x moves
    2 px while the face moves ~15 px), so the slide / lip-colour detectors fire on a correct generated mouth and the 2-neighbour flow
    interpolation then blends frames across the move (the doubled / lopsided mouth)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); m = np.zeros(len(frames)); prev = None
    for k, f in enumerate(frames):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
        if prev is not None and k < len(lm) and np.isfinite(lm[k][[33, 263]]).all():
            q = lm[k]; iod = np.linalg.norm(q[33, :2] - q[263, :2]); x0, y0 = np.nanmin(q[:, :2], 0); x1, y1 = np.nanmax(q[:, :2], 0); px, py = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
            H, W = g.shape; a0, b0 = max(0, int((y0 - py) / 2)), max(0, int((x0 - px) / 2)); a1, b1 = min(H, int((y1 + py) / 2) + 1), min(W, int((x1 + px) / 2) + 1)
            if a1 - a0 > 8 and b1 - b0 > 8:
                fl = dis.calc(prev[a0:a1, b0:b1], g[a0:a1, b0:b1], None); m[k] = 2 * np.percentile(np.linalg.norm(fl, axis=2), 90) / max(iod, 1e-6)
        prev = g
    return m


def main():''', 1)

s = s.replace('''    ap.add_argument("--max-run", type=int, default=3);''', '''    ap.add_argument("--motion-thr", type=float, default=0.0, help="r15 F motion-aware repair (needs --src-video; 0 = off): where the face moves faster than this "
                    "(face_motion, IOD per frame, max over the run's span +-2 frames) the flow interpolation is faded out (weight 1 at the threshold, 0 at 2x it) and the generated frames kept")
    ap.add_argument("--max-run", type=int, default=3);''', 1)

old = '''        Sr, Lr = red_stats(SF, src), red_stats(F[:len(SF)], src); m = min(len(Sr), n)'''
new = '''        if a.motion_thr > 0:
            mot = face_motion(SF, src); mot = np.concatenate([mot, np.zeros(max(0, n - len(mot)))])[:n]
        Sr, Lr = red_stats(SF, src), red_stats(F[:len(SF)], src); m = min(len(Sr), n)'''
assert old in s; s = s.replace(old, new, 1)

old = '''    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); fb_out = np.zeros(n, np.float32); rep = []'''
new = '''    if "mot" not in locals(): mot = np.zeros(n)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); fb_out = np.zeros(n, np.float32); rep = []; moved = []'''
assert old in s; s = s.replace(old, new, 1)

old = '''            gp, gq = cv2.cvtColor(F[p], cv2.COLOR_BGR2GRAY), cv2.cvtColor(F[q], cv2.COLOR_BGR2GRAY)'''
new = '''            mv = float(mot[max(0, p - 2):min(n, q + 3)].max()) if a.motion_thr > 0 else 0.0
            wm = 1.0 if a.motion_thr <= 0 else float(np.clip(2.0 - mv / a.motion_thr, 0.0, 1.0))   # r15 F: 1 below the threshold, 0 at 2x it
            if wm < 1.0: moved.append([b, e, round(mv, 4), round(wm, 3)])
            if wm <= 0.0: continue   # fast move: keep the generated frames (the generator followed the move; the flags are stale-landmark false alarms)
            gp, gq = cv2.cvtColor(F[p], cv2.COLOR_BGR2GRAY), cv2.cvtColor(F[q], cv2.COLOR_BGR2GRAY)'''
assert old in s; s = s.replace(old, new, 1)

old = '''                F[k] = np.clip((1 - al) * wp.astype(np.float32) + al * wq.astype(np.float32), 0, 255).astype(np.uint8)
                lm[k] = (1 - al) * lm[p] + al * lm[q]'''
new = '''                F[k] = np.clip(wm * ((1 - al) * wp.astype(np.float32) + al * wq.astype(np.float32)) + (1 - wm) * F[k].astype(np.float32), 0, 255).astype(np.uint8)
                lm[k] = wm * ((1 - al) * lm[p] + al * lm[q]) + (1 - wm) * lm[k]'''
assert old in s; s = s.replace(old, new, 1)

old = '''    pathlib.Path(a.out).with_suffix(".repair.json").write_text(json.dumps({**rp, "repaired": rep}, indent=1))'''
new = '''    if a.motion_thr > 0: rp["motion"] = {"thr": a.motion_thr, "faded_runs": len(moved), "p95": round(float(np.percentile(mot, 95)), 4), "p99": round(float(np.percentile(mot, 99)), 4)}
    pathlib.Path(a.out).with_suffix(".repair.json").write_text(json.dumps({**rp, "repaired": rep, "motion_faded": moved}, indent=1))'''
assert old in s; s = s.replace(old, new, 1)
p.write_text(s); print("patched r9_layer_repair.py")

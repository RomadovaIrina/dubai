#!/usr/bin/env python3
"""r15 Z: apply the GENERATOR-side fixes of the face-path code audit (coordination/audit_r15_face.md). Idempotent; every anchor must match once.
Clear bugs fixed by default:
  - codeformer_accel: the eye centroids of the 5-point CodeFormer alignment no longer include the MediaPipe iris points that MAP106 put
    into the replayed 106 eye contours (34/38/40 -> 468/468/470, 88/92/94 -> 473/473/475; two of them twice). CF_EYE_IRIS=1 restores.
  - anchor_filters.rigid_motion: a failed RANSAC fit, too few flow samples or an inlier share < AF_MIN_INLIER (0.3) no longer leaves an
    identity step in the cumulative head track; the frame pair's landmark similarity (rigid points, least squares) is used instead.
    The generator's rigid cache is versioned (master_rigid_v2*.npz).
Behaviour changes behind flags (default off):
  CF_ALIGN_FIT=lsq    least-squares similarity for the CodeFormer alignment (instead of LMEDS on 5 points)
  CF_MOUTH=rigid      CodeFormer mouth corners = the segment's median position in the eyes + nose frame (not the SOURCE mouth)
  AF_UPPER=nobrow     the rigid head motion measured without the brows / forehead
  LS_NOSE_RIGID=1     LatentSync's nose anchor without the nostril wings (rigid nose + the video's median wing offset)
  LS_SIL_HYST=1       steps-silent window class: majority of 3 consecutive windows (no isolated 10 / 20-step window)
  LS_VAE_MEAN=1       the VAE posterior MEAN for the masked / reference latents (LatentSync samples it)
    python scripts/r15/patch_audit.py"""
import pathlib, re, sys
R = pathlib.Path(__file__).resolve().parents[2]


def sub(path, old, new, count=1, regex=False, marker=None):
    p = R / path; s = p.read_text()
    if (marker or new) in s:
        print(f"  = {path}: already applied"); return
    n = len(re.findall(old, s)) if regex else s.count(old)
    if n != count: sys.exit(f"{path}: anchor found {n}x (expected {count}): {old[:80]!r}")
    s = re.sub(old, new, s) if regex else s.replace(old, new); p.write_text(s); print(f"  + {path}")


# ---------------------------------------------------------------- codeformer_accel.py
sub("scripts/optim/codeformer_accel.py", "import pathlib\nimport sys\nimport time\n", "import os\nimport pathlib\nimport sys\nimport time\n")
sub("scripts/optim/codeformer_accel.py", "LM106_TO_5: list = [list(range(33, 43)), list(range(87, 97)), list(range(77, 85)), 52, 61]\n",
    "LM106_TO_5: list = [list(range(33, 43)), list(range(87, 97)), list(range(77, 85)), 52, 61]\n"
    "# r15 audit (default): the REPLAYED 106 (clean_face_stack.MAP106) put MediaPipe iris points into the eye contours (106 34/38/40 -> 468/468/470,\n"
    "# 88/92/94 -> 473/473/475: two of them twice), so the eye centroids moved with gaze / blinks. Lid contour only; CF_EYE_IRIS=1 restores.\n"
    "_IRIS106 = {34, 38, 40, 88, 92, 94}\n"
    "if os.environ.get(\"CF_EYE_IRIS\", \"0\") != \"1\":\n"
    "    LM106_TO_5 = [[i for i in LM106_TO_5[0] if i not in _IRIS106], [i for i in LM106_TO_5[1] if i not in _IRIS106]] + LM106_TO_5[2:]\n\n\n"
    "def lsq_similarity(src, dst) -> np.ndarray:\n"
    "    \"\"\"least-squares similarity (rotation + uniform scale + translation) src -> dst, closed form, 2x3 float64 (CF_ALIGN_FIT=lsq)\"\"\"\n"
    "    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64); A = np.zeros((2 * len(src), 4)); b = dst.reshape(-1)\n"
    "    A[0::2] = np.stack([src[:, 0], -src[:, 1], np.ones(len(src)), np.zeros(len(src))], 1)\n"
    "    A[1::2] = np.stack([src[:, 1], src[:, 0], np.zeros(len(src)), np.ones(len(src))], 1)\n"
    "    a_, b_, tx, ty = np.linalg.lstsq(A, b, rcond=None)[0]; return np.array([[a_, -b_, tx], [b_, a_, ty]])\n\n\n"
    "def rigid_mouth_lm5(lms: list) -> list:\n"
    "    \"\"\"CF_MOUTH=rigid (r15 audit): the 5-point mouth corners are the SOURCE (RU) mouth while CodeFormer restores the GENERATED mouth.\n"
    "    They are replaced by the segment's median mouth-corner position in the eyes + nose frame: they follow the head, not the speech.\"\"\"\n"
    "    idx = [i for i, l in enumerate(lms) if l is not None]\n"
    "    if len(idx) < 3: return lms\n"
    "    S = {i: lsq_similarity(lms[i][:3], FFHQ_TEMPLATE_512[:3]) for i in idx}\n"
    "    med = np.median(np.stack([np.asarray(lms[i][3:], np.float64) @ S[i][:, :2].T + S[i][:, 2] for i in idx]), 0); out = list(lms)\n"
    "    for i in idx:\n"
    "        l = np.array(lms[i], np.float32, copy=True); l[3:] = (med - S[i][:, 2]) @ np.linalg.inv(S[i][:, :2]).T; out[i] = l\n"
    "    return out\n")
sub("scripts/optim/codeformer_accel.py",
    "    def affine_for(lm5: np.ndarray) -> np.ndarray:\n        return cv2.estimateAffinePartial2D(",
    "    def affine_for(lm5: np.ndarray) -> np.ndarray:\n"
    "        if os.environ.get(\"CF_ALIGN_FIT\", \"lmeds\") == \"lsq\": return lsq_similarity(lm5, FFHQ_TEMPLATE_512)   # r15 audit flag: no LMEDS fit jumps\n"
    "        return cv2.estimateAffinePartial2D(")

# ---------------------------------------------------------------- face_aware_latentsync_accel.py (the CodeFormer call site)
sub("scripts/pilot/face_aware_latentsync_accel.py", "        from codeformer_accel import lm5_from_insightface106\n",
    "        from codeformer_accel import lm5_from_insightface106, rigid_mouth_lm5\n")
sub("scripts/pilot/face_aware_latentsync_accel.py",
    r"(\n( +))(lms = \[lm5_from_insightface106\(dets\[k\]\[1\]\) if k < len\(dets\) else None for k in range\(n_use\)\]; lm_src = \"insightface106->5 \(verify pass\)\"\n)",
    r'\1\3\2if os.environ.get("CF_MOUTH", "source") == "rigid": lms = rigid_mouth_lm5(lms); lm_src += " + rigid mouth (CF_MOUTH, r15 audit)"\n', regex=True,
    marker='os.environ.get("CF_MOUTH", "source")')

# ---------------------------------------------------------------- anchor_filters.py
sub("scripts/pilot/anchor_filters.py", "from __future__ import annotations\nimport numpy as np\n", "from __future__ import annotations\nimport os\nimport numpy as np\n")
sub("scripts/pilot/anchor_filters.py", "EYE_IN = [159, 145, 386, 374, 468, 473]  # pupils / lid centres (excluded radius: blinks)\n",
    "EYE_IN = [159, 145, 386, 374, 468, 473]  # pupils / lid centres (excluded radius: blinks)\n"
    "# r15 audit: brows / forehead move with expressions (brow raises) and are not rigid; AF_UPPER=nobrow measures the head motion without them\n"
    "BROW_FOREHEAD = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276, 9, 8, 151, 108, 337, 69, 299}\n"
    "if os.environ.get(\"AF_UPPER\", \"prod\") == \"nobrow\": UPPER = [k for k in UPPER if k not in BROW_FOREHEAD]\n"
    "MIN_INLIER = float(os.environ.get(\"AF_MIN_INLIER\", \"0\") or 0)   # r15 audit flag: below this RANSAC inlier SHARE the fit is replaced (0 = off: at\n"
    "# the 0.5 px threshold the share measures flow noise, m18 median 0.53, 23 % of pairs < 0.3); always replaced: no fit / < 12 samples / < 8 inliers\n"
    "CACHE_TAG = \"_v2\" + (\"_nobrow\" if os.environ.get(\"AF_UPPER\", \"prod\") == \"nobrow\" else \"\")   # rigid-track cache files: the code version\n"
    "RIGID_LM = [1, 4, 5, 6, 168, 197, 195, 33, 133, 263, 362]   # landmark fallback for a failed flow fit (nose / eye corners; no forehead)\n\n\n"
    "def _lm_similarity(a, b):\n"
    "    \"\"\"least-squares similarity of the rigid landmarks a (frame t-1) -> b (frame t); identity when they are missing\"\"\"\n"
    "    a = np.asarray(a, np.float64)[RIGID_LM, :2]; b = np.asarray(b, np.float64)[RIGID_LM, :2]\n"
    "    if not (np.isfinite(a).all() and np.isfinite(b).all()): return np.array([[1, 0, 0], [0, 1, 0]], np.float64)\n"
    "    A = np.zeros((2 * len(a), 4)); A[0::2] = np.stack([a[:, 0], -a[:, 1], np.ones(len(a)), np.zeros(len(a))], 1)\n"
    "    A[1::2] = np.stack([a[:, 1], a[:, 0], np.zeros(len(a)), np.ones(len(a))], 1)\n"
    "    p, q, tx, ty = np.linalg.lstsq(A, b.reshape(-1), rcond=None)[0]; return np.array([[p, -q, tx], [q, p, ty]])\n")
sub("scripts/pilot/anchor_filters.py",
    "            if len(xs) >= 12:\n"
    "                p0 = np.stack([xs + X0, ys + Y0], 1).astype(np.float32); p1 = p0 + F[ys, xs]\n"
    "                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=0.5, maxIters=2000, confidence=0.995)\n"
    "                if M is not None:\n"
    "                    Hs[t] = M; info[t] = (len(xs), float(inl.mean()))\n",
    "            M, share = None, 0.0\n"
    "            if len(xs) >= 12:\n"
    "                p0 = np.stack([xs + X0, ys + Y0], 1).astype(np.float32); p1 = p0 + F[ys, xs]\n"
    "                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=0.5, maxIters=2000, confidence=0.995)\n"
    "                share = float(inl.mean()) if M is not None and inl is not None else 0.0\n"
    "            if M is not None and share >= MIN_INLIER and share * len(xs) >= 8:\n"
    "                Hs[t] = M; info[t] = (len(xs), share)\n"
    "            else:   # r15 audit (default): a failed / unreliable fit used to stay IDENTITY = a step in the cumulative head track that flow:4\n"
    "                Hs[t] = _lm_similarity(lm[t - 1], lm[t]); info[t] = (len(xs), -1.0 - share)   # spreads over +-12 frames; info < 0 = replaced\n")

# ---------------------------------------------------------------- clean_face_stack.py (the generator's rigid cache + the nose anchor)
sub("scripts/pilot/clean_face_stack.py", 't1 = time.perf_counter(); rp = pathlib.Path(work) / "master_rigid.npz"',
    't1 = time.perf_counter(); rp = pathlib.Path(work) / f"master_rigid{af.CACHE_TAG}.npz"')
sub("scripts/pilot/clean_face_stack.py", "        self.ls_dets = ls_detections(lm_ls, W, H, calib)\n",
    "        if os.environ.get(\"LS_NOSE_RIGID\", \"0\") == \"1\": lm_ls = nose_rigid(lm_ls); self.anchor_info[\"nose_rigid\"] = True   # r15 audit flag\n"
    "        self.ls_dets = ls_detections(lm_ls, W, H, calib)\n")
sub("scripts/pilot/clean_face_stack.py", "\n\ndef _seed_master_mp(",
    "\n\ndef nose_rigid(lm):\n"
    "    \"\"\"LS_NOSE_RIGID=1 (r15 audit): LatentSync's nose anchor (106 74/77/83/86 = MediaPipe 5/64/294/4) holds the two nostril WINGS, which\n"
    "    flare with smiles / speech and shift the crop. The wing points are moved so that the anchor centroid = the rigid nose centroid (5/4/1/195)\n"
    "    plus the video's median wing offset in the eye frame (the mean alignment and the wing spread are kept).\"\"\"\n"
    "    lm = np.array(lm, np.float64, copy=True); ok = np.isfinite(lm[:, 0, 0]) & np.isfinite(lm[:, [4, 5, 1, 195, 64, 294, 33, 263]]).all((1, 2))\n"
    "    if ok.sum() < 3: return lm.astype(np.float32)\n"
    "    e = lm[:, 263] - lm[:, 33]; d = np.linalg.norm(e, axis=1, keepdims=True); ux = e / np.maximum(d, 1e-6); uy = np.stack([-ux[:, 1], ux[:, 0]], 1)\n"
    "    rig = lm[:, [5, 4, 1, 195]].mean(1); off = lm[:, [5, 64, 294, 4]].mean(1) - rig\n"
    "    med = np.median(np.stack([(off * ux).sum(1), (off * uy).sum(1)], 1)[ok] / d[ok], 0)\n"
    "    T = rig + (med[0] * ux + med[1] * uy) * d; m = (4 * T - lm[:, 5] - lm[:, 4]) / 2; half = (lm[:, 64] - lm[:, 294]) / 2\n"
    "    lm[ok, 64] = (m + half)[ok]; lm[ok, 294] = (m - half)[ok]; return lm.astype(np.float32)\n\n\n"
    "def _seed_master_mp(")

# ---------------------------------------------------------------- ls_stable.py (VAE mean, steps-silent hysteresis)
sub("scripts/quality/ls_stable.py", "        t0 = time.perf_counter(); pipe = pipe or load_latentsync()\n",
    "        t0 = time.perf_counter(); pipe = pipe or load_latentsync()\n"
    "        if os.environ.get(\"LS_VAE_MEAN\", \"0\") == \"1\": _vae_mean(pipe)   # r15 audit flag\n")
sub("scripts/quality/ls_stable.py", "\n\nclass StableRunner:\n",
    "\n\ndef _vae_mean(pipe):\n"
    "    \"\"\"LS_VAE_MEAN=1 (r15 audit): LatentSync's prepare_mask_latents / prepare_image_latents SAMPLE the VAE posterior (latent_dist.sample),\n"
    "    a fresh per-frame perturbation of the masked / reference latents; the posterior mean is used instead.\"\"\"\n"
    "    if getattr(pipe.vae, \"_r15_vae_mean\", False): return\n"
    "    enc = pipe.vae.encode\n"
    "    def encode(x, *a, **k):\n"
    "        out = enc(x, *a, **k); d = out.latent_dist; d.sample = lambda generator=None, _d=d: _d.mean; return out\n"
    "    pipe.vae.encode = encode; pipe.vae._r15_vae_mean = True; print(\"  LS_VAE_MEAN: VAE posterior mean for the masked / reference latents\", flush=True)\n\n\n"
    "class StableRunner:\n")
sub("scripts/quality/ls_stable.py",
    "                    is_sil = {w: bool(sil[w[0]:w[1]].mean() >= self.silent_frac) for w in windows}\n",
    "                    is_sil = {w: bool(sil[w[0]:w[1]].mean() >= self.silent_frac) for w in windows}\n"
    "                    if os.environ.get(\"LS_SIL_HYST\", \"0\") == \"1\" and len(full) >= 3:   # r15 audit flag: majority of 3 consecutive windows\n"
    "                        seq_ = [is_sil[w] for w in full]\n"
    "                        for i_ in range(1, len(full) - 1): is_sil[full[i_]] = (seq_[i_ - 1] + seq_[i_] + seq_[i_ + 1]) >= 2\n")
# ---------------------------------------------------------------- frame_map.py (exact master -> source map; default)
sub("scripts/pilot/frame_map.py", "import pathlib, subprocess\nimport numpy as np\n", "import os, pathlib, subprocess\nimport numpy as np\n")
sub("scripts/pilot/frame_map.py", "\n\ndef frame_map(src: pathlib.Path, master_frames, max_step: int = 4) -> dict:\n",
    "\n\ndef exact_master_to_source(src) -> np.ndarray:\n"
    "    \"\"\"r15 audit (default; FRAME_MAP=thumb restores the greedy thumbnail match): which source frame each master frame holds, decided EXACTLY\n"
    "    by replaying the master command (face_aware_latentsync.make_master: `-r 25`, or DUB_MASTER_TIMING=fps `-vf fps=25`) on a tiny index video\n"
    "    with the source's own timestamps (each frame's index as 14 bit blocks), as scripts/quality/r3_master_timing.py does. The greedy match\n"
    "    disagreed with that map on 249 of 2258 old03 master frames (up to 3 source frames: a landmark spike when FAST_MPREUSE seeds the master pass).\"\"\"\n"
    "    import av, tempfile\n"
    "    c = av.open(str(src)); s = c.streams.video[0]; tb = s.time_base; rate = s.average_rate\n"
    "    pts = np.array(sorted(p.pts for p in c.demux(s) if p.pts is not None), np.int64); c.close()\n"
    "    NB, BS = 14, 16; W, H = 8 + NB * (BS + 4), BS + 16\n"
    "    tm = [\"-vf\", f\"fps={FPS:g}\"] if os.environ.get(\"DUB_MASTER_TIMING\", \"r25\") == \"fps\" else [\"-r\", f\"{FPS:g}\"]\n"
    "    with tempfile.TemporaryDirectory() as td:\n"
    "        idx = pathlib.Path(td) / \"idx.mp4\"; out = pathlib.Path(td) / \"idx_25.mp4\"\n"
    "        o = av.open(str(idx), \"w\"); st = o.add_stream(\"libx264rgb\", rate=rate); st.width, st.height = W, H; st.pix_fmt = \"rgb24\"; st.time_base = tb\n"
    "        st.codec_context.time_base = tb; st.options = {\"crf\": \"0\", \"preset\": \"ultrafast\"}\n"
    "        for k, p in enumerate(pts):\n"
    "            img = np.zeros((H, W, 3), np.uint8)\n"
    "            for b in range(NB):\n"
    "                if (k >> b) & 1: img[8:8 + BS, 8 + b * (BS + 4):8 + b * (BS + 4) + BS] = 255\n"
    "            f = av.VideoFrame.from_ndarray(img, format=\"rgb24\"); f.pts = int(p); f.time_base = tb\n"
    "            for q in st.encode(f): o.mux(q)\n"
    "        for q in st.encode(): o.mux(q)\n"
    "        o.close()\n"
    "        subprocess.run([\"ffmpeg\", \"-loglevel\", \"error\", \"-y\", \"-nostdin\", \"-i\", str(idx)] + tm + [\"-crf\", \"18\", \"-an\", str(out)], check=True)\n"
    "        c = av.open(str(out)); j = []\n"
    "        for fr in c.decode(video=0):\n"
    "            g = fr.to_ndarray(format=\"gray\"); j.append(sum(1 << b for b in range(NB) if g[8 + BS // 2, 8 + b * (BS + 4) + BS // 2] > 127))\n"
    "        c.close()\n"
    "    return np.array(j, np.int64)\n\n\n"
    "def frame_map(src: pathlib.Path, master_frames, max_step: int = 4) -> dict:\n")
sub("scripts/pilot/frame_map.py",
    "    used = np.bincount(jm, minlength=len(S)); lag = sp[jm] - np.arange(len(M)) / FPS\n"
    "    return {\"method\": \"grey 58x106 thumbnail match, monotone search (step 0..%d source frames)\" % max_step, \"master_fps\": FPS,\n",
    "    jm_thumb = jm.copy(); method = \"grey 58x106 thumbnail match, monotone search (step 0..%d source frames)\" % max_step\n"
    "    if os.environ.get(\"FRAME_MAP\", \"exact\") != \"thumb\":   # r15 audit (default): the exact replayed map (exact_master_to_source)\n"
    "        try:\n"
    "            ex = exact_master_to_source(src)\n"
    "            if len(ex) >= len(M) and ex[:len(M)].max() < len(S):\n"
    "                jm = ex[:len(M)].astype(int); mad = np.abs(S[jm] - M).mean(axis=(1, 2)); method = \"exact: master command replayed on an index video (thumbnail match kept as a check)\"\n"
    "            else: print(f\"  [frame_map] exact map unusable ({len(ex)} frames, max {ex.max() if len(ex) else -1}); thumbnail match kept\", flush=True)\n"
    "        except Exception as e_:\n"
    "            print(f\"  [frame_map] exact map failed ({e_!r}); thumbnail match kept\", flush=True)\n"
    "    used = np.bincount(jm, minlength=len(S)); lag = sp[jm] - np.arange(len(M)) / FPS\n"
    "    return {\"method\": method, \"master_fps\": FPS, \"thumb_disagree\": int((jm_thumb != jm).sum()),\n")
print("PATCH_AUDIT_OK")

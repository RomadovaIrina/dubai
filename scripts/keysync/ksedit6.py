K = "/workspace/dub/keysync/"
def ed(p, pairs):
    s = open(K + p).read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a); s = s.replace(a, b)
    open(K + p, "w").write(s); print("EDITED", p)

ed("r11face/ks_fix.py", [
 ("ap.add_argument(\"--cm-robust\"",
  "ap.add_argument(\"--flash-fix\", type=float, default=0, help=\"r13: single-frame mouth flash repair: a layer frame whose mouth-box hf / L / chroma spikes > T robust sigmas \"\n"
  "                \"over the median of +-2 frames (while the master does not, < T/2) gets its lips region from the neighbours' mean (r3_flash logic at 25 fps)\")\n"
  "ap.add_argument(\"--flash-report\", default=\"\")\n"
  "ap.add_argument(\"--cm-robust\""),
 ("# ---- explosion guard\n",
  "# ---- r13 flash repair (before everything else: the repaired frames go through the guard / colour match like any other)\n"
  "if a.flash_fix > 0:\n"
  "    Lr = Lr.copy()\n"
  "    def mstats(fr, i):\n"
  "        p = SL[i, LIPS_O]; c = p.mean(0); r = 0.75 * mw[i]\n"
  "        xa, xb, ya, yb = int(max(0, c[0] - r)), int(min(W, c[0] + r)), int(max(0, c[1] - 0.6 * r)), int(min(H, c[1] + 0.6 * r))\n"
  "        g = cv2.cvtColor(fr[ya:yb, xa:xb], cv2.COLOR_BGR2LAB).astype(np.float32)\n"
  "        return [np.abs(cv2.Laplacian(g[..., 0], cv2.CV_32F)).mean(), g[..., 0].mean(), (np.abs(g[..., 1] - 128) + np.abs(g[..., 2] - 128)).mean()]\n"
  "    sk_ = np.array([mstats(Lr[i], i) for i in range(n)]); sm_ = np.array([mstats(M[i], i) for i in range(n)])\n"
  "    def spike(x):\n"
  "        z = np.zeros_like(x)\n"
  "        for i in range(n):\n"
  "            nb = [x[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < n]; z[i] = x[i] - np.median(nb, 0)\n"
  "        return z\n"
  "    sc = 1.4826 * np.median(np.abs(np.diff(sm_, axis=0)), 0) + 1e-3                   # the master's robust frame-to-frame scale\n"
  "    zk, zm = np.abs(spike(sk_)) / sc, np.abs(spike(sm_)) / sc\n"
  "    fl = np.where((zk.max(1) > a.flash_fix) & (zm.max(1) < a.flash_fix / 2))[0]\n"
  "    good = np.ones(n, bool); good[fl] = False; gi = np.where(good)[0]\n"
  "    for i in fl:\n"
  "        lo_, hi_ = gi[gi < i], gi[gi > i]\n"
  "        if not len(lo_) or not len(hi_): continue\n"
  "        p, q = lo_[-1], hi_[0]; t = (i - p) / (q - p)\n"
  "        nb = Lr[p].astype(np.float32) * (1 - t) + Lr[q].astype(np.float32) * t\n"
  "        mk = poly(np.concatenate([KL[i, LIPS_O], SL[i, LIPS_O]]), int(0.25 * mw[i])).astype(np.uint8)\n"
  "        mk = cv2.dilate(mk, np.ones((3, 3), np.uint8)) | poly(SL[i, LIPS_O], int(0.25 * mw[i])).astype(np.uint8)\n"
  "        mk = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), max(1.0, 0.08 * mw[i]))[..., None]\n"
  "        bx = Lr[i][y0:y0 + s, x0:x0 + s].astype(np.float32)\n"
  "        Lr[i][y0:y0 + s, x0:x0 + s] = np.clip(bx * (1 - mk) + nb[y0:y0 + s, x0:x0 + s] * mk + 0.5, 0, 255).astype(np.uint8)\n"
  "    if a.flash_report: json.dump({\"frames\": fl.tolist(), \"zk\": zk.round(2).tolist()}, open(a.flash_report, \"w\"))\n"
  "    print(f\"== flash-fix {a.flash_fix}: {len(fl)} frames repaired {fl.tolist()[:40]}\", flush=True)\n\n"
  "# ---- explosion guard\n"),
])

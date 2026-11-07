import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/workspace/dub/keysync/r16/breath/m18_"
def hp(x): return x - np.convolve(np.pad(x, (12, 12), mode="edge"), np.ones(25) / 25, "valid")
arms = [("d40", "d40 plain", "#9aa0a6"), ("smart", "smart (what Irina saw)", "#d93025"), ("b_eccS0", "candidate: ECC align, scale locked", "#1a73e8")]
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for i, reg in enumerate(("ring", "lower")):
    for k, lab, col in arms:
        x = np.load(R + k + ".npz")[reg]; t = np.arange(len(x)) / 25
        ax[i].plot(t, hp(x) * 100, color=col, lw=1.2 if k != "smart" else 1.6, label=f"{lab}: std {hp(x).std() * 100:.2f} %")
    ax[i].axhline(0, color="k", lw=0.5); ax[i].set_ylabel(f"{'jaw outline' if reg == 'ring' else 'cheeks + jaw'}\nsize vs source, %"); ax[i].legend(loc="upper right", fontsize=8)
ax[1].set_xlabel("time, s"); fig.suptitle("m18: face size pulsing vs the source (fast part, 1 s trend removed); 0 = same size as the source")
plt.tight_layout(); plt.savefig("/workspace/dub/outbox/r16_k1/m18_size_breathing_plot.png", dpi=110)
s = hp(np.load(R + "smart.npz")["ring"]); c = np.convolve(np.abs(s), np.ones(50) / 50, "valid"); picks = []
for i in np.argsort(-c):
    if all(abs(i - p) > 50 for p in picks): picks.append(i)
    if len(picks) == 3: break
print(" ".join(f"{p / 25:.1f}:{(p + 50) / 25:.1f}" for p in sorted(picks)))

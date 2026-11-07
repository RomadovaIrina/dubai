#!/usr/bin/env python3
"""r13 distillation pilot, step 1: capture the LatentSync CONDITIONING of every inference window from the PRODUCTION generator, without
editing it (r4_video.sh runs ${R10_GEN_PY}, as scripts/r13/fault_gen.py): the UNet's forward is wrapped; every call's conditional rows
(audio != 0) give mask (1 ch), masked-image latents (4), reference latents (4) and the whisper audio embeddings of a 16-frame window, which do
not depend on the timestep, so a cheap 2-step run is enough (--steps 2 --steps-silent 2 appended). Duplicates (the same window at the next
timestep) are skipped by a hash.
For the SyncNet loss the window's mel (80 x 52, LatentSync's own melspectrogram + crop, unet_dataset.py) is cut from the segment's audio:
Audio2Feature.audio2feat / feature2chunks are wrapped to keep each segment's audio and its per-frame chunks; a window's audio rows are found
in them (exact fp16 match of its first rows) -> its start frame a -> mel[:, int(80 * a / 25) : + 52].
-> $R13_CAPTURE_DIR/<job>_<k>.pt  {"cond": (9, F, 64, 64) fp16, "audio": (F, 50, 384) fp16, "mel": (1, 80, 52) | None, "start": a}"""
import hashlib, math, os, pathlib, runpy, sys
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[2]; LS = ROOT / "third_party" / "latentsync"
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality")); sys.path.insert(0, str(LS))
import numpy as np, torch
from latentsync.models.unet import UNet3DConditionModel
from latentsync.whisper.audio2feature import Audio2Feature

OUT = pathlib.Path(os.environ["R13_CAPTURE_DIR"]); OUT.mkdir(parents=True, exist_ok=True); JOB = os.environ.get("R10_JOB", "job")
seen, n, miss, DBG = set(), [0], [0], [0]; ST = {"path": None, "rows": {}, "mel": None}
MEL_LEN = math.ceil(16 / 5 * 16)   # 52, unet_dataset.mel_window_length


def mel_of(path):
    import librosa
    cwd = os.getcwd(); os.chdir(LS)   # latentsync.utils.audio reads configs/audio.yaml relative to the repo
    try:
        from latentsync.utils.audio import melspectrogram
        y, _ = librosa.load(str(path), sr=16000); return torch.from_numpy(melspectrogram(y)).float()
    finally:
        os.chdir(cwd)


_a2f, _f2c = Audio2Feature.audio2feat, Audio2Feature.feature2chunks


def audio2feat(self, audio_path, *a, **k):
    ST["path"] = audio_path
    try: ST["mel"] = mel_of(audio_path)
    except Exception as e: ST["mel"] = None; print(f"[capture] mel failed {e}", flush=True)
    return _a2f(self, audio_path, *a, **k)


def feature2chunks(self, *a, **k):
    chunks = _f2c(self, *a, **k)
    ST["rows"] = {}
    for j, c in enumerate(chunks):
        t = (c if torch.is_tensor(c) else torch.as_tensor(np.asarray(c))).to(torch.float16).cpu().contiguous()
        ST["rows"].setdefault(hashlib.md5(t.numpy().tobytes()).hexdigest(), j)
    return chunks


Audio2Feature.audio2feat, Audio2Feature.feature2chunks = audio2feat, feature2chunks
orig = UNet3DConditionModel.forward


def forward(self, sample, timestep, encoder_hidden_states=None, *a, **k):
    with torch.no_grad():
        au = encoder_hidden_states
        if os.environ.get("R13_CAPTURE_DEBUG") and DBG[0] < 8:
            DBG[0] += 1; print(f"[capture-dbg] sample {tuple(sample.shape)} t {timestep} au {None if au is None else (tuple(au.shape), [round(float(au[r].abs().max()), 3) for r in range(au.shape[0])])}", flush=True)
        if au is not None and sample.shape[1] == 13:
            aub = au.reshape(sample.shape[0], -1, *au.shape[1:])   # the audio rows come flattened per frame: (B * F, 50, 384) -> (B, F, 50, 384)
            for r in range(sample.shape[0]):
                if float(aub[r].abs().max()) == 0: continue          # the CFG unconditional row
                c = sample[r, 4:].detach().to(torch.float16).cpu().contiguous()
                h = hashlib.md5(c.numpy().tobytes()).hexdigest()
                if h in seen: continue
                seen.add(h); ar = aub[r].detach().to(torch.float16).cpu().contiguous()
                start = ST["rows"].get(hashlib.md5(ar[0].numpy().tobytes()).hexdigest()); mel = None
                if start is not None and ST["mel"] is not None:
                    i0 = int(80.0 * start / 25); m = ST["mel"][:, i0:i0 + MEL_LEN]
                    if m.shape[-1] == MEL_LEN: mel = m.unsqueeze(0).to(torch.float16)
                if mel is None: miss[0] += 1
                torch.save({"cond": c, "audio": ar, "mel": mel, "start": start, "job": JOB}, OUT / f"{JOB}_{n[0]:05d}.pt"); n[0] += 1
    return orig(self, sample, timestep, encoder_hidden_states, *a, **k)


UNet3DConditionModel.forward = forward
target = ROOT / "scripts" / "quality" / "candidate_video_stage.py"
sys.argv[0] = str(target); sys.argv += ["--steps", "2", "--steps-silent", "2"]
try:
    runpy.run_path(str(target), run_name="__main__")
finally:
    print(f"[capture] {JOB}: {n[0]} windows ({miss[0]} without a mel) -> {OUT}", flush=True)

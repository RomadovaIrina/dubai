#!/usr/bin/env python3
"""Clean background-bed A/B (Agent C, 2026-10-16): replace torchaudio HDEMUCS_HIGH_MUSDB_PLUS (weights without a licence -> NO) in
background_bed.py by a separator whose code AND weights are permissive, CPU only. Every candidate produces bg_44k.wav + vocals_44k.wav and goes
through the UNCHANGED background_bed.mix_background (balance, duck, room-tone fill) with the same dub track, so only the bed differs.

  spleeter   Deezer Spleeter 2stems (MIT code, weights = release asset of the MIT repo), own venv (TensorFlow): bg = accompaniment
  umxhq      Open-Unmix UMX-HQ (MIT code, MIT weights on Zenodo): bg = drums + bass + other
  umxse      Open-Unmix UMX-SE speech enhancement (MIT / MIT, 16 kHz): bg = noise (upsampled, both channels)
  roomtone   no model: the source itself outside source speech (+-margin, 30 ms crossfades), silence under speech -> mix_background's room-tone
             fill puts the video's own steady noise floor there (no music under speech)
  hdemucs    the cand5 bed of the reference run (NC / no licence: comparison only, never shipped)

Metrics on the bed as mixed (mix / peak_norm - dub): in dub pauses where the SOURCE speaks (the risk zone: the Russian voice leaks) vs dub
pauses where the source is silent (pure ambience): Silero speech probability of the bed (voice residue), bed level (dB), level difference
(continuity: 0 = the ambience does not change when the source starts talking), 100 ms level jumps inside dub pauses (p95), mix level in dub
pauses (dead air = very low).

    python scripts/quality/bed_ab.py --manifest BASE/03_final.manifest.json --out ROOT/bed --beds spleeter umxhq umxse roomtone hdemucs
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality"): sys.path.insert(0, str(HERE.parent / d))
SR = 44100
BM = pathlib.Path("/workspace/dub/artifacts/audio-timing/bed_models"); PYLIB = "/workspace/dub/artifacts/audio-timing/pylib"; SPLEETER_PY = "/workspace/dub/venv-spleeter/bin/python"


def ff(args): subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", *args], check=True)


SPLEETER_CODE = r"""
import os, sys, numpy as np, soundfile as sf, tensorflow as tf
n = int(os.environ.get("DUB_THREADS", "2")); tf.config.threading.set_intra_op_parallelism_threads(n); tf.config.threading.set_inter_op_parallelism_threads(1)
from spleeter.separator import Separator
y, sr = sf.read(sys.argv[1], dtype="float32"); assert sr == 44100
s = Separator("spleeter:2stems", multiprocess=False).separate(y)
sf.write(sys.argv[2], s["accompaniment"], sr, subtype="FLOAT"); sf.write(sys.argv[3], s["vocals"], sr, subtype="FLOAT")
"""


def sep_spleeter(src, d):   # r2: the pipeline's own separator (background_bed.separate_spleeter, scripts/setup_spleeter.sh)
    import background_bed as bb, shutil
    tmp = d / "source_44k.wav"; shutil.copy(src, tmp); info = bb.separate_spleeter(tmp, d, time.perf_counter())
    return {"model": info["model"], "separate_s": info["separate_s"]}


def _umx(kind, y, sr):
    import torch
    os.environ["TORCH_HOME"] = str(BM / "torch_home"); sys.path.insert(0, PYLIB); import openunmix
    torch.set_num_threads(int(os.environ.get("DUB_THREADS", "2")))
    if kind == "umxhq":
        m = openunmix.umxhq(targets=["vocals", "drums", "bass", "other"], device="cpu", niter=1)
        with torch.no_grad(): est = m(torch.from_numpy(np.ascontiguousarray(y.T))[None])[0].numpy()   # (4, 2, L)
        return (est[1] + est[2] + est[3]).T, est[0].T
    m = openunmix.umxse(targets=["speech", "noise"], device="cpu", niter=1)
    y16 = y.mean(1); import scipy.signal as ss
    x = ss.resample_poly(y16, 160, 441).astype(np.float32)
    with torch.no_grad(): est = m(torch.from_numpy(x)[None, None])[0].numpy()   # (2, 1, L16)
    up = lambda z: np.repeat(ss.resample_poly(z[0], 441, 160)[:, None].astype(np.float32), 2, axis=1)
    return up(est[1]), up(est[0])


def sep_umx(kind, src, d):
    y, sr = sf.read(str(src), dtype="float32"); bg, voc = _umx(kind, y, sr); n = len(y)
    fit = lambda z: np.pad(z, ((0, max(0, n - len(z))), (0, 0)))[:n]
    sf.write(str(d / "bg_44k.wav"), fit(bg), SR, subtype="FLOAT"); sf.write(str(d / "vocals_44k.wav"), fit(voc), SR, subtype="FLOAT")
    return {"model": "Open-Unmix UMX-HQ (bg = drums + bass + other)" if kind == "umxhq" else "Open-Unmix UMX-SE (16 kHz, bg = noise)"}


def sep_roomtone(src, d, speech, margin=0.15, xf=0.03):
    y, sr = sf.read(str(src), dtype="float32"); n = len(y); g = np.ones(n, np.float32); k = int(xf * sr)
    for s, e in speech:
        a, b = max(0, int((s - margin) * sr)), min(n, int((e + margin) * sr)); g[a:b] = 0.0
    g = np.convolve(g, np.ones(k) / k, mode="same").astype(np.float32)     # 30 ms crossfades at every gate edge
    import background_bed as bb   # its room_tone() skips floors above -42 dBFS (music / loud ambience); here the shuffled steady frames ARE the bed under speech
    tone, floor = bb.room_tone(y, speech, SR, n, max_floor_db=0.0)
    if tone is None: tone = np.zeros_like(y)
    bg = y * g[:, None] + tone * (1 - g)[:, None]
    sf.write(str(d / "bg_44k.wav"), bg, SR, subtype="FLOAT"); sf.write(str(d / "vocals_44k.wav"), y * (1 - g)[:, None], SR, subtype="FLOAT")
    return {"model": "none: source outside source speech (+-0.15 s); under speech the video's own steady non-speech frames, shuffled + crossfaded",
            "tone_floor_db": round(float(20 * np.log10(floor + 1e-12)), 2) if floor else None}


def mask(ivs, n, sr):
    m = np.zeros(n, bool)
    for s, e in ivs: m[max(0, int(s * sr)):min(n, int(e * sr))] = True
    return m


def speech_prob(y16):
    import torch
    from silero_vad import load_silero_vad
    m = load_silero_vad(); w = 512; p = []
    with torch.no_grad():
        for i in range(0, len(y16) - w, w): p.append(float(m(torch.from_numpy(y16[i:i + w].copy()), 16000)))
    return np.array(p), w / 16000


def metrics(mix_wav, dub24, norm, speech, dubsp, dur):
    import scipy.signal as ss
    mix, _ = sf.read(mix_wav, dtype="float32"); n = len(mix)
    d44 = ss.resample_poly(sf.read(str(dub24), dtype="float32")[0], 147, 80).astype(np.float32); d44 = np.pad(d44, (0, max(0, n - len(d44))))[:n]
    bed = mix.mean(1) / norm - d44
    S, D = mask(speech, n, SR), mask(dubsp, n, SR); risk, amb = (~D) & S, (~D) & (~S)
    db = lambda z: round(float(20 * np.log10(np.sqrt(np.mean(z ** 2)) + 1e-9)), 2) if len(z) else None
    p, hop = speech_prob(ss.resample_poly(bed, 160, 441).astype(np.float32)); idx = (np.arange(len(p)) * hop * SR).astype(int)
    pr, pa = p[risk[idx]], p[amb[idx]]
    h = int(0.1 * SR); fr = 20 * np.log10(np.sqrt(np.mean(bed[: n // h * h].reshape(-1, h) ** 2, axis=1)) + 1e-9); pz = (~D)[: n // h * h].reshape(-1, h).mean(1) > 0.99
    jumps = np.abs(np.diff(fr))[pz[1:] & pz[:-1]]
    return {"bed_db_risk": db(bed[risk]), "bed_db_ambience": db(bed[amb]), "continuity_db": round(db(bed[risk]) - db(bed[amb]), 2) if risk.any() and amb.any() else None,
            "residue_speech_prob_risk_mean": round(float(pr.mean()), 3) if len(pr) else None, "residue_speech_s_risk": round(float((pr > 0.5).sum() * hop), 2),
            "residue_speech_prob_ambience_mean": round(float(pa.mean()), 3) if len(pa) else None, "residue_speech_s_ambience": round(float((pa > 0.5).sum() * hop), 2),
            "risk_s": round(float(risk.sum() / SR), 2), "ambience_s": round(float(amb.sum() / SR), 2), "level_jump_p95_db": round(float(np.percentile(jumps, 95)), 2) if len(jumps) else None,
            "mix_db_dub_pauses": db(mix.mean(1)[~D])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--beds", nargs="+", default=["spleeter", "umxhq", "umxse", "roomtone", "hdemucs"])
    ap.add_argument("--residue-duck-db", type=float, default=0.0, help="background_bed.mix_background residue_duck_db (runner default -15)")
    ap.add_argument("--dub", default=None, help="dub track (24 kHz); default: the manifest work dir's dubbed_24k.wav")
    a = ap.parse_args()
    import background_bed as bb
    from dub_gaps import dub_vad
    man = json.loads(pathlib.Path(a.manifest).read_text()); W = pathlib.Path(man["work_dir"]); dur = man["source"]["duration_s"]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); src = out / "source_44k.wav"
    if not src.exists(): ff(["-i", man["input"], "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(src)])
    dub24 = pathlib.Path(a.dub) if a.dub else W / "dubbed_24k.wav"; d16 = out / "dub16.wav"; ff(["-i", str(dub24), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(d16)])
    speech = [(v["start"], v["end"]) for v in man["speech_intervals"]]; dubsp = dub_vad(d16, 60); rows = {}
    ref_bg = pathlib.Path(str(man.get("background", {}).get("separation", {}).get("bg", "")))
    for b in a.beds:
        d = out / b; d.mkdir(exist_ok=True); t0 = time.perf_counter()
        if b == "spleeter": info = sep_spleeter(src, d)
        elif b in ("umxhq", "umxse"): info = sep_umx(b, src, d)
        elif b == "roomtone": info = sep_roomtone(src, d, speech)
        elif b == "hdemucs":   # comparison only: the reference run's own HDEMUCS stems (read-only, never shipped)
            orig = pathlib.Path("/workspace/dub/artifacts/baseline") / pathlib.Path(man["input"]).stem / "work" / f"work_{pathlib.Path(man['input']).stem}"
            if (orig / "bg_44k.wav").exists():
                for f in ("bg_44k.wav", "vocals_44k.wav"):
                    if not (d / f).exists(): (d / f).symlink_to(orig / f)
            elif not (d / "bg_44k.wav").exists():   # r2: stems not uploaded -> separate again, comparison only (the runner refuses HDEMUCS without this variable)
                os.environ["DUB_ALLOW_NC_BED"] = "1"; bb.separate_background(pathlib.Path(man["input"]), d, separator="hdemucs"); os.environ.pop("DUB_ALLOW_NC_BED")
            info = {"model": "torchaudio HDEMUCS_HIGH_MUSDB_PLUS (cand5; NO licence: reference only)"}
        sep_s = round(time.perf_counter() - t0, 1)
        mx = bb.mix_background({"bg": str(d / "bg_44k.wav"), "vocals": str(d / "vocals_44k.wav")}, dub24, speech, dubsp, dur, d, residue_duck_db=a.residue_duck_db)
        m = metrics(mx["wav"], dub24, mx["peak_norm"], speech, dubsp, dur)
        rows[b] = {**info, "separate_s_cpu": sep_s, "mix": {k: mx[k] for k in ("room_fill", "residue_duck", "bg_gain_db", "bed_db_in_dub_pauses", "mix_db_in_dub_pauses", "peak_norm")}, **m}
        print(f"{b:9s} sep {sep_s:6.1f}s | residue risk p {m['residue_speech_prob_risk_mean']} ({m['residue_speech_s_risk']} s of {m['risk_s']}) amb p {m['residue_speech_prob_ambience_mean']} "
              f"| bed dB risk {m['bed_db_risk']} amb {m['bed_db_ambience']} cont {m['continuity_db']} | jump p95 {m['level_jump_p95_db']} | mix dB in pauses {m['mix_db_dub_pauses']} | fill {mx['room_fill'].get('applied')}", flush=True)
    (out / "bed_ab.json").write_text(json.dumps(rows, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

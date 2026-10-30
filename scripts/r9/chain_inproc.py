"""r9 speed (audio lever C, R9_CHAIN_INPROC=on with R9_ONEPROC=on): the CPU chain after the TTS stage inside the audio process.

Without it: close_inner + soften (r8_audio.sh), then lip_place + hush + soften + remix (r7 -> r6 -> r5 -> r4_audio.sh), six Python processes, each
importing torch / librosa / transformers again, lip_place loading Whisper large-v3 again and hush wav2vec2 again (m01: 29 s after the stage).
Here the same mains run with the same argv, output dirs, links and env, in the process that already holds the TTS QA's Whisper (int8_float16 =
lip_place's own WhisperModel arguments) and wav2vec2 (= CtcAligner.standalone's model): those two are reused, nothing else is shared.
Thread pools are set to the chain processes' (OMP / MKL / torch = DUB_THREADS of r4_audio.sh, 24 at PAR=1).
    run(job, qa)    job = scripts/r8/r8_audio.sh's chain_<id>.json"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@contextlib.contextmanager
def _logged(path, argv):
    """the step's stdout / stderr to its log (as the bash redirect), its argv as sys.argv"""
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    out0, err0, argv0 = sys.stdout, sys.stderr, sys.argv
    with open(path, "w") as fh:
        sys.stdout = sys.stderr = fh; sys.argv = argv
        try: yield
        finally: sys.stdout, sys.stderr, sys.argv = out0, err0, argv0


def _step(main, name, argv, log):
    """True when the step's main ran without an exception / non-zero exit (the bash `&&`)"""
    with _logged(log, [name] + [str(x) for x in argv]):
        try:
            r = main(); return r in (None, 0)
        except SystemExit as e:
            return e.code in (None, 0)
        except Exception:
            import traceback; traceback.print_exc(); return False


def _link(target, link):
    link = pathlib.Path(link); link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists(): link.unlink()
    link.symlink_to(target)


def _reuse(qa):
    """lip_place's WhisperModel(large-v3, cuda, int8_float16) = the QA's Whisper; hush's CtcAligner.standalone() = the QA's wav2vec2"""
    import faster_whisper, tts_align
    W0 = faster_whisper.WhisperModel
    qpath = str(ROOT / "models" / "whisper-large-v3")
    def WM(model_size_or_path, device="auto", compute_type="default", *a, **k):
        if qa is not None and str(model_size_or_path) == qpath and device == "cuda" and compute_type == "int8_float16" and not a and not k \
                and getattr(qa, "_r9_compute", None) == "int8_float16":
            return qa.wh
        return W0(model_size_or_path, device, compute_type, *a, **k)
    faster_whisper.WhisperModel = WM
    if qa is not None:
        tts_align.CtcAligner.standalone = classmethod(lambda cls, device="cuda": cls.from_qa(qa) if device == qa.device else cls(*_standalone(device)))
    return W0


def _standalone(device):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    p = str(ROOT / "models" / "wav2vec2-base-960h")
    return Wav2Vec2Processor.from_pretrained(p), Wav2Vec2ForCTC.from_pretrained(p).to(device).eval(), device


def run(job: dict, qa) -> bool:
    for p in (ROOT / "scripts" / "pilot", ROOT / "scripts" / "quality", ROOT / "scripts" / "r7", ROOT / "scripts" / "r8"): sys.path.insert(0, str(p))
    import close_inner, soften_edges, lip_place, hush_placed, r3_remix
    S, A, id_, TAG = pathlib.Path(job["S"]), pathlib.Path(job["A"]), job["id"], job["tag"]
    # 1. r8_audio.sh, the one-process branch: the furthest stage output, then close_inner / soften (the stage's env)
    t0 = time.time(); W = S / "rt" / id_
    if not (W / "dubbed_24k.wav").exists(): print(f"[chain] {id_}: no retake output", file=sys.stderr); return False
    if (S / "ct" / id_ / "dubbed_24k.wav").exists(): W = S / "ct" / id_
    if (S / "ba" / id_ / "dubbed_24k.wav").exists(): W = S / "ba" / id_
    if job["close"] and _step(close_inner.main, "close_inner", ["--work", W, "--out", S / "ci" / id_, "--tmin", job["tmin"]], S / f"ci_{id_}.log"): W = S / "ci" / id_
    if job["soften"] and _step(soften_edges.main, "soften_edges", ["--work", W, "--out", S / "se" / id_], S / f"se_{id_}.log"): W = S / "se" / id_
    t3 = time.time(); _link(W, S / "mix_src" / id_)
    json.dump({"retake": 0.0, "burst_align": 0.0, "close_inner": t3 - t0}, open(S / f"{id_}.steps.json", "w"))
    # 2. r4_audio.sh one() on REUSE=<tag>src: lip_place, hush, final soften, remix, mux (the env the r7 -> r6 -> r5 -> r4 wrappers give it)
    os.environ.update(job["env"])
    import torch
    th = int(job["env"].get("DUB_THREADS", "24")); torch.set_num_threads(th)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(th)
    except Exception:
        pass
    W0 = _reuse(qa)
    O = A / TAG; (O / "runs").mkdir(parents=True, exist_ok=True); (O / "mix_src").mkdir(parents=True, exist_ok=True)
    with open(O / "README.txt", "a") as fh: fh.write(f"r4_audio {TAG} (in-process chain, scripts/r9/chain_inproc.py) REMIX_ARGS='{job['remix_args']}' REUSE='{S.name}'\n")
    t_a = time.time(); W = pathlib.Path(os.path.realpath(S / "mix_src" / id_)); fc = A / "sep" / id_ / "source_mediapipe.npz"
    man = sorted(glob.glob(str(W / f"{id_}_*.manifest.json")))[0]
    if not _step(lip_place.main, "lip_place", ["--work", W, "--out", O / "placed" / id_, "--manifest", man, "--face", fc] + shlex.split(job["lip_args"]), O / "runs" / f"{id_}.place.log"):
        print(f"FAIL place {id_}"); return False
    W = O / "placed" / id_; t_p = time.time()
    if not _step(hush_placed.main, "hush_placed", ["--placed", W, "--out", O / "hushed" / id_], O / "runs" / f"{id_}.hush.log"):
        print(f"FAIL hush {id_}"); return False
    W = O / "hushed" / id_
    if _step(soften_edges.main, "soften_edges", ["--work", W, "--out", O / "soft" / id_], O / "runs" / f"{id_}.soft.log"): W = O / "soft" / id_
    else: print(f"WARN final soften {id_}")
    t_h = time.time(); _link(W, O / "mix_src" / id_)
    if os.environ.get("R9_ON_MIXSRC"):   # r9 speed O: the demucs remix starts now (as r4_audio.sh does)
        subprocess.Popen(["bash", "-c", os.environ["R9_ON_MIXSRC"]], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    if not _step(r3_remix.main, "r3_remix", ["--src", O / "mix_src", "--out", O] + shlex.split(job["remix_args"]) + [id_], O / "runs" / f"{id_}.remix.log"):
        print(f"FAIL remix {id_}"); return False
    t_r = time.time()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", job["src"], "-i", str(O / id_ / "final_mix_aac.m4a"), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", str(O / f"{id_}_over-original.mp4")])
    json.dump({"audio_or_reuse": 0.0, "lip_place": t_p - t_a, "hush": t_h - t_p, "remix": t_r - t_h, "mux+metrics(not production)": time.time() - t_r}, open(O / "runs" / f"{id_}.steps.json", "w"))
    import faster_whisper; faster_whisper.WhisperModel = W0
    (S / f"chain_done_{id_}").write_text("ok\n")
    return True

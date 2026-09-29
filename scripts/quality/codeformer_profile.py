#!/usr/bin/env python3
"""Legacy CodeFormer profile (week 2, step 1): where does `inference_codeformer.py -i video.mp4` spend its time?

Two measurements on the same input video (normally the LatentSync/pipeline output of 04):
  --cli           the untouched upstream CLI as scripts/pilot/benchmark_06_codeformer.py runs it (own process, PNGs on disk,
                  model load inside), wall + peak VRAM via scripts/pilot/measure_vram.GpuSampler. This is the "A. Legacy" number.
  --instrumented  the SAME per-frame loop as inference_codeformer.py re-executed in-process with a timer around every step
                  (decode, read_image, RetinaFace detection, align/warp, img2tensor+H2D, CodeFormer net, empty_cache, ParseNet,
                  paste-back, PNG writes, PNG re-read + mp4 assembly). Every upstream function is called unchanged; only timers
                  and the ParseNet call are wrapped. The sum of the step timers is compared with the loop wall time -> "other".
Writes <out>/codeformer_profile_<stem>.json and (with --md) reports/quality/codeformer_baseline_profile.md.
    source scripts/env.sh
    python scripts/quality/codeformer_profile.py --input /tmp/dabai_quality/baseline/04_baseline.mp4 --cli --instrumented --md
"""
from __future__ import annotations
import argparse, json, os, pathlib, shutil, subprocess, sys, tempfile, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
from pilot_common import ROOT, THIRD_PARTY, gpu_info, write_json  # noqa: E402

CF = THIRD_PARTY / "CodeFormer"
Q = pathlib.Path("/tmp/dabai_quality")


def probe(p: pathlib.Path) -> dict:
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)], check=True, capture_output=True, text=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video"); num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    return {"duration_s": round(float(j["format"]["duration"]), 3), "width": v["width"], "height": v["height"], "fps": round(int(num) / max(int(den), 1), 4), "frames": int(v.get("nb_frames") or 0)}


def run_cli(src: pathlib.Path, w: float, keep: bool) -> dict:
    """A. Legacy: upstream CLI in its own process, exactly as benchmark_06_codeformer.py invokes it."""
    from measure_vram import GpuSampler
    info = probe(src); work = pathlib.Path(tempfile.mkdtemp(prefix="dabai-cf-legacy-"))
    cmd = ["/venv/dabai/bin/python", "inference_codeformer.py", "-i", str(src), "-o", str(work), "-w", str(w), "-s", "1",
           "--detection_model", "retinaface_resnet50", "--save_video_fps", f"{info['fps'] or 25:g}"]
    t0 = time.perf_counter(); proc = subprocess.Popen(cmd, cwd=str(CF), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    s = GpuSampler(proc.pid, 0.2); s.start(); rc = proc.wait(); dur = time.perf_counter() - t0; s.stop()
    outs = sorted(work.glob("*.mp4")); op = probe(outs[0]) if outs else None
    n_png = sum(1 for _ in work.rglob("*.png")); size = sum(p.stat().st_size for p in work.rglob("*") if p.is_file())
    row = {"mode": "legacy_cli", "cmd": " ".join(cmd), "returncode": rc, "seconds": round(dur, 3), "s_per_video_min": round(dur / info["duration_s"] * 60, 2),
           "peak_device_mib": s.peak_device_mib, "peak_process_mib": s.peak_proc_mib, "png_files": n_png, "disk_bytes": size, "source": info, "output": op,
           "geometry_changed": bool(op) and (op["width"], op["height"]) != (info["width"], info["height"]), "workdir": str(work) if keep else None}
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return row


def run_instrumented(src: pathlib.Path, w: float, max_frames: int | None, write_png: bool) -> dict:
    """B0. The upstream loop, in-process, with timers. Calls the untouched upstream functions."""
    import cv2, torch
    from torchvision.transforms.functional import normalize
    os.chdir(CF); sys.path.insert(0, str(CF))
    from basicsr.utils import imwrite, img2tensor, tensor2img
    from basicsr.utils.registry import ARCH_REGISTRY
    from basicsr.utils.video_util import VideoReader, VideoWriter
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from measure_vram import GpuSampler
    T: dict[str, float] = {k: 0.0 for k in ("decode", "read_image", "detection", "align_warp", "img2tensor_h2d", "net", "empty_cache", "tensor2img_d2h",
                                            "parsenet", "paste_back", "png_write", "video_assembly", "model_load")}
    def tick(k, t0):
        T[k] += time.perf_counter() - t0
    work = pathlib.Path(tempfile.mkdtemp(prefix="dabai-cf-instr-")); info = probe(src)
    sampler = GpuSampler(os.getpid(), 0.2); sampler.start()
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    device = torch.device("cuda")
    net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to(device)
    net.load_state_dict(torch.load(str(CF / "weights/CodeFormer/codeformer.pth"))["params_ema"]); net.eval()
    face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50", save_ext="png", use_parse=True, device=device)
    real_parse = face_helper.face_parse
    class _TimedParse:                      # wraps the ParseNet call inside paste_faces_to_input_image
        def __call__(self, x):
            t = time.perf_counter(); out = real_parse(x); torch.cuda.synchronize(); T["parsenet"] += time.perf_counter() - t; return out
    face_helper.face_parse = _TimedParse()
    torch.cuda.synchronize(); tick("model_load", t0)
    t0 = time.perf_counter()
    frames = []; vr = VideoReader(str(src)); img = vr.get_frame()
    while img is not None:
        frames.append(img); img = vr.get_frame()
        if max_frames and len(frames) >= max_frames: break
    audio = vr.get_audio(); fps = vr.get_fps(); vr.close(); tick("decode", t0)
    n_faces = 0; out_shape = None
    for i, img in enumerate(frames):
        face_helper.clean_all()
        t0 = time.perf_counter(); face_helper.read_image(img); tick("read_image", t0)
        t0 = time.perf_counter(); n = face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5); torch.cuda.synchronize(); tick("detection", t0)
        t0 = time.perf_counter(); face_helper.align_warp_face(); tick("align_warp", t0)
        for cropped_face in face_helper.cropped_faces:
            t0 = time.perf_counter()
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True); normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device); torch.cuda.synchronize(); tick("img2tensor_h2d", t0)
            t0 = time.perf_counter()
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
            torch.cuda.synchronize(); tick("net", t0)
            t0 = time.perf_counter(); restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1)); tick("tensor2img_d2h", t0)
            del output
            t0 = time.perf_counter(); torch.cuda.empty_cache(); tick("empty_cache", t0)
            face_helper.add_restored_face(restored_face.astype("uint8"), cropped_face); n_faces += 1
        t0 = time.perf_counter(); face_helper.get_inverse_affine(None); restored_img = face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=False); tick("paste_back", t0)
        T["paste_back"] -= 0.0
        out_shape = restored_img.shape
        if write_png:
            t0 = time.perf_counter(); b = str(i).zfill(6)
            for idx, (cf_, rf) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
                imwrite(cf_, str(work / "cropped_faces" / f"{b}_{idx:02d}.png")); imwrite(rf, str(work / "restored_faces" / f"{b}_{idx:02d}.png"))
            imwrite(restored_img, str(work / "final_results" / f"{b}.png")); tick("png_write", t0)
    if write_png:
        t0 = time.perf_counter()
        vf = [cv2.imread(str(p)) for p in sorted((work / "final_results").glob("*.png"))]
        h, wd = vf[0].shape[:2]; vw = VideoWriter(str(work / "out.mp4"), h, wd, fps, audio)
        for f in vf: vw.write_frame(f)
        vw.close(); tick("video_assembly", t0)
    wall = time.perf_counter() - t_all; sampler.stop()
    T["paste_back"] -= T["parsenet"]      # ParseNet time was measured inside paste_back; report it separately
    loop = wall - T["model_load"] - T["decode"] - T["video_assembly"]
    steps = {k: round(v, 3) for k, v in T.items()}
    per_frame = {k: round(v / max(len(frames), 1) * 1000, 2) for k, v in T.items()}
    other = round(wall - sum(T.values()), 3)
    disk = sum(p.stat().st_size for p in work.rglob("*") if p.is_file()); shutil.rmtree(work, ignore_errors=True)
    n = len(frames); scale = info["duration_s"] * (n / max(info["frames"], n)) if info["frames"] else info["duration_s"]
    return {"mode": "legacy_instrumented", "frames": n, "faces": n_faces, "fidelity_weight": w, "wall_seconds": round(wall, 3), "loop_seconds": round(loop, 3),
            "s_per_video_min": round(wall / scale * 60, 2), "steps_seconds": steps, "ms_per_frame": per_frame, "other_seconds": other,
            "peak_device_mib": sampler.peak_device_mib, "peak_process_mib": sampler.peak_proc_mib, "input_shape": list(frames[0].shape) if frames else None,
            "read_image_shape": list(out_shape) if out_shape is not None else None, "png_written": write_png, "disk_bytes": disk, "source": info}


def md_report(rows: dict, src: pathlib.Path) -> str:
    L = [f"# CodeFormer legacy profile — {src.name} (week 2, step 1)", "",
         f"Input: `{src}` {rows['source']['width']}x{rows['source']['height']} {rows['source']['fps']} fps {rows['source']['frames']} frames {rows['source']['duration_s']} s. "
         f"GPU {rows['gpu'].get('name')} ({rows['gpu'].get('capability')}). Script `scripts/quality/codeformer_profile.py`; raw `{rows['json']}`.", ""]
    if rows.get("cli"):
        c = rows["cli"]
        L += ["## A. Legacy CLI (`inference_codeformer.py -i video.mp4 -w 0.5 -s 1 --detection_model retinaface_resnet50`, own process)", "",
              "| wall s | s/video-min | peak VRAM MiB (device / process) | PNG files | disk MB | output geometry |", "|---:|---:|---|---:|---:|---|",
              f"| {c['seconds']} | {c['s_per_video_min']} | {c['peak_device_mib']} / {c['peak_process_mib']} | {c['png_files']} | {c['disk_bytes'] / 2**20:.0f} | "
              f"{c['output']['width']}x{c['output']['height']} (input {c['source']['width']}x{c['source']['height']}) {'CHANGED' if c['geometry_changed'] else 'preserved'} |", ""]
    if rows.get("instrumented"):
        i = rows["instrumented"]
        L += [f"## B0. Same loop in-process with timers ({i['frames']} frames, {i['faces']} faces, wall {i['wall_seconds']} s = {i['s_per_video_min']} s/video-min, "
              f"peak VRAM {i['peak_device_mib']} MiB device / {i['peak_process_mib']} MiB process)", "",
              "| step | seconds | % of wall | ms / frame |", "|---|---:|---:|---:|"]
        for k, v in sorted(i["steps_seconds"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v} | {v / i['wall_seconds'] * 100:.1f} | {i['ms_per_frame'][k]} |")
        L += [f"| other (untimed) | {i['other_seconds']} | {i['other_seconds'] / i['wall_seconds'] * 100:.1f} | |", "",
              f"read_image output shape {i['read_image_shape']} vs input {i['input_shape']} (facelib upscales the short side to 512 before detection; the CLI output keeps that size).", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--cli", action="store_true"); ap.add_argument("--instrumented", action="store_true"); ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None); ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=str(Q / "codeformer")); ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    src = pathlib.Path(a.input).resolve(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jp = out / f"codeformer_profile_{src.stem}.json"
    rows = {"input": str(src), "source": probe(src), "gpu": gpu_info(), "json": str(jp), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.cli:
        rows["cli"] = run_cli(src, a.w, a.keep); print(json.dumps({k: v for k, v in rows["cli"].items() if k != "cmd"}, indent=1))
    if a.instrumented:
        rows["instrumented"] = run_instrumented(src, a.w, a.max_frames, not a.no_png); print(json.dumps(rows["instrumented"], indent=1))
    write_json(jp, rows)
    if a.md:
        md = ROOT / "reports/quality/codeformer_baseline_profile.md"; md.write_text(md_report(rows, src)); print(f"-> {md}")
    print(f"-> {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Optimized in-memory CodeFormer stage (week 2), built next to scripts/optim/latentsync_accel.py.

Upstream flow (third_party/CodeFormer/inference_codeformer.py, per frame, from disk to disk):
  frame -> FaceRestoreHelper.read_image (upscales short side < 512: 464x848 becomes 512x936!) -> RetinaFace detection
  (resize 640) -> 5 landmarks -> estimateAffinePartial2D to the FFHQ 512 template -> warpAffine crop 512x512 -> CodeFormer
  net (batch 1) -> torch.cuda.empty_cache() -> ParseNet face-parsing mask (batch 1, two 101x101 blurs on CPU) -> inverse
  affine warp of face / square mask / parse mask over the WHOLE frame + erosion / blur / fusion / alpha paste over the whole
  frame -> 3 PNGs per frame -> PNGs re-read -> mp4.

This module keeps the restoration semantics (same template, LMEDS partial affine, same crop, the same network call
`net(x, w, adain=True)`, the same ParseNet mask recipe, the same erosion / blur / fusion / paste arithmetic and uint8
conversions; scripts/quality/codeformer_bench.py --check-reference proves max |diff| <= 1 against the upstream
FaceRestoreHelper functions fed with the same landmarks) and changes only the execution around it:
  * network and ParseNet loaded ONCE per process (`CodeFormerAccel`) and reused for every segment;
  * frames stay in RAM (numpy BGR uint8), no PNG / video round trips, frame geometry never changes (read_image's implicit
    upscale is not applied: the face is pasted back into the frame at its own resolution);
  * face alignment reuses landmarks the pipeline already has (LatentSync's insightface verify pass -> 5 points, or the
    RetinaFace router) -> no second face detector; a frame without landmarks is left untouched (counted) unless a fallback
    detector callable is supplied;
  * CodeFormer and ParseNet run on batches of faces (`batch_size`); input tensors are written into one preallocated
    (pinned) buffer, the output batch is converted to uint8 on the GPU and copied back once;
  * the parse-mask blur (2 x GaussianBlur 101x101 sigma 11 + 10 px border) runs on the GPU as a separable reflect-101
    convolution for the whole batch (max |diff| 2.4e-4 vs cv2 on the [0, 1] mask);
  * paste-back is computed only inside the ROI that the inverse-warped 512 square can touch (plus the blur margin) —
    outside it the mask is exactly zero, so the result is identical to the whole-frame upstream computation;
  * torch.cuda.empty_cache() is NOT called per face (opt-in `empty_cache_per_face=True` reproduces upstream for A/B);
  * CPU work (align warps, paste-back) runs in a thread pool (OpenCV releases the GIL) and the GPU batch k+1 is computed
    on a dedicated thread while batch k is pasted -> GPU and CPU overlap.
Every call records per-stage seconds, faces, fallbacks and peak VRAM in `self.calls` / `self.totals`.

    from codeformer_accel import CodeFormerAccel, lm5_from_insightface106
    cf = CodeFormerAccel(fidelity_weight=0.5, batch_size=8)
    out_frames, st = cf.restore_frames(frames_bgr, landmarks5_per_frame)
"""
from __future__ import annotations

import concurrent.futures as cf
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
from pilot_common import THIRD_PARTY  # noqa: E402

CF_ROOT = THIRD_PARTY / "CodeFormer"
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

FACE_SIZE = 512
# FaceRestoreHelper.face_template for det_model != dlib, template_3points=False, face_size=512, crop_ratio (1, 1)
# (facelib/utils/face_restoration_helper.py) — asserted equal at runtime by codeformer_bench.py / codeformer_landmark_check.py.
FFHQ_TEMPLATE_512 = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                              [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)
BORDER_GRAY_BGR = (135, 133, 132)
# facelib paste_faces_to_input_image: 19 ParseNet classes -> 255 = keep restored, 0 = keep original
MASK_COLORMAP = np.array([0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0], dtype=np.float32)
PARSE_BORDER = 10
BLUR_K, BLUR_SIGMA = 101, 11

# insightface landmark_2d_106 (LatentSync's verify pass) -> the 5 RetinaFace/FFHQ points: left eye, right eye, nose tip,
# left mouth corner, right mouth corner. Entries are one index or an index list (centroid). MEASURED against facexlib RetinaFace
# landmarks on 76 sampled master frames of 04/01/03 (scripts/quality/codeformer_landmark_check.py, 2026-09-29):
#   eye contour centroids 33-42 / 87-96: median 2.4 / 2.8 px from the RetinaFace eye points, nose = centroid of 77-84: 3.1 px,
#   mouth corners 52 / 61: 3.8 / 3.4 px. For scale, RetinaFace's OWN landmarks disagree with themselves by 2.1-3.2 px median
#   (router at full resolution vs the upstream CodeFormer detection at resize=640), so the derived points are within the
#   detector's noise; the 512 crops differ from RetinaFace-aligned crops by the same amount as the two RetinaFace passes differ
#   from each other (PSNR 18.2 dB vs 19.6 dB). Not assumed from documentation.
LM106_TO_5: list = [list(range(33, 43)), list(range(87, 97)), list(range(77, 85)), 52, 61]


def lm5_from_insightface106(lmk106, mapping: list | None = None) -> np.ndarray | None:
    """5 x (x, y) float32 from LatentSync's 106-point landmarks; None when unavailable."""
    if lmk106 is None:
        return None
    m = mapping or LM106_TO_5
    lm = np.asarray(lmk106, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[0] < 106:
        return None
    return np.stack([lm[idx].mean(axis=0) if isinstance(idx, (list, tuple)) else lm[idx] for idx in m]).astype(np.float32)


def is_gray_bgr(img: np.ndarray, threshold: int = 10) -> bool:
    """facelib.utils.misc.is_gray semantics (mean channel difference)."""
    img = img.astype(np.float32)
    d1 = np.mean(np.abs(img[..., 0] - img[..., 1])); d2 = np.mean(np.abs(img[..., 1] - img[..., 2]))
    return d1 <= threshold and d2 <= threshold


class CodeFormerAccel:
    def __init__(self, fidelity_weight: float = 0.5, batch_size: int = 8, use_parse: bool = True, empty_cache_per_face: bool = False,
                 device: str = "cuda", workers: int = 4, autocast: bool = False, parse_autocast: bool = False, pinned: bool = True,
                 fallback_detector=None, landmark_source: str = "insightface", mask_blur: str = "gpu", roi_paste: bool = True,
                 overlap: bool = True):
        from basicsr.archs import codeformer_arch  # noqa: F401  registers 'CodeFormer'
        from basicsr.utils.registry import ARCH_REGISTRY
        from facelib.parsing.parsenet import ParseNet
        assert batch_size >= 1 and landmark_source in ("insightface", "retinaface") and mask_blur in ("gpu", "cv2")
        self.w, self.B, self.use_parse, self.empty_cache_per_face = float(fidelity_weight), int(batch_size), use_parse, empty_cache_per_face
        self.device, self.workers, self.autocast, self.parse_autocast, self.pinned = device, max(1, int(workers)), autocast, parse_autocast, pinned
        self.fallback_detector, self.landmark_source, self.mask_blur, self.roi_paste, self.overlap = fallback_detector, landmark_source, mask_blur, roi_paste, overlap
        t0 = time.perf_counter()
        net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to(device)
        ck = torch.load(str(CF_ROOT / "weights/CodeFormer/codeformer.pth"), map_location="cpu")["params_ema"]
        net.load_state_dict(ck); net.eval(); self.net = net
        self.parse = None
        if use_parse:
            p = ParseNet(in_size=512, out_size=512, parsing_ch=19)
            p.load_state_dict(torch.load(str(CF_ROOT / "weights/facelib/parsing_parsenet.pth"), map_location="cpu"), strict=True)
            p.eval(); self.parse = p.to(device)
        self.load_s = round(time.perf_counter() - t0, 3)
        self._buf = torch.empty((self.B, 3, FACE_SIZE, FACE_SIZE), dtype=torch.float32, pin_memory=(pinned and device.startswith("cuda")))
        k = cv2.getGaussianKernel(BLUR_K, BLUR_SIGMA).astype(np.float32)
        self._kx = torch.from_numpy(k).to(device).view(1, 1, 1, BLUR_K); self._ky = torch.from_numpy(k).to(device).view(1, 1, BLUR_K, 1)
        self._cmap = torch.from_numpy(MASK_COLORMAP).to(device)
        self._pool = cf.ThreadPoolExecutor(max_workers=self.workers)
        self._gpu = cf.ThreadPoolExecutor(max_workers=1)
        self.config = {"fidelity_weight": self.w, "batch_size": self.B, "use_parse": use_parse, "empty_cache_per_face": empty_cache_per_face,
                       "autocast": autocast, "parse_autocast": parse_autocast, "pinned": pinned, "workers": self.workers, "mask_blur": mask_blur,
                       "roi_paste": roi_paste, "overlap": overlap, "template": "FFHQ 512 (facelib)", "geometry": "preserved (no read_image upscale)",
                       "landmark_source": landmark_source}
        self.calls: list[dict] = []
        self.totals = {"frames_in": 0, "faces_restored": 0, "frames_without_landmarks": 0, "fallback_detections": 0, "gray_faces": 0, "seconds": 0.0}

    # ------------------------------------------------------------------ geometry (facelib semantics, no detector)
    @staticmethod
    def affine_for(lm5: np.ndarray) -> np.ndarray:
        return cv2.estimateAffinePartial2D(np.asarray(lm5, dtype=np.float32), FFHQ_TEMPLATE_512, method=cv2.LMEDS)[0]

    @staticmethod
    def align_crop(frame_bgr: np.ndarray, affine: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(frame_bgr, affine, (FACE_SIZE, FACE_SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_GRAY_BGR)

    @staticmethod
    def paste_back(frame_bgr: np.ndarray, restored_bgr: np.ndarray, affine: np.ndarray, parse_mask: np.ndarray | None, roi: bool = True,
                   stats: dict | None = None) -> np.ndarray:
        """FaceRestoreHelper.paste_faces_to_input_image for ONE face, upscale_factor 1, no upsampler, no draw_box.
        roi=True evaluates the identical arithmetic only inside the region the warped 512 square (+ blur margin) can reach.
        `stats` (optional dict) receives exact pre-encode locality numbers: roi box, warped-square box, changed pixels and
        the box of the changed pixels."""
        h, w = frame_bgr.shape[:2]
        inverse_affine = cv2.invertAffineTransform(affine)          # upscale 1 -> no extra offset
        x0 = y0 = 0; W, H = w, h; M = inverse_affine
        if roi:
            c = np.array([[0, 0, 1], [FACE_SIZE, 0, 1], [0, FACE_SIZE, 1], [FACE_SIZE, FACE_SIZE, 1]], dtype=np.float64) @ inverse_affine.T
            # margin: the square mask blur (blur_size <= sqrt(area)//20*2 <= ~ FACE_SIZE/10) + erosion + warp interpolation
            m = int(np.ceil(max(np.linalg.norm(inverse_affine[:, :2], axis=0)) * FACE_SIZE / 10)) + 8
            x0, y0 = max(0, int(np.floor(c[:, 0].min())) - m), max(0, int(np.floor(c[:, 1].min())) - m)
            x1, y1 = min(w, int(np.ceil(c[:, 0].max())) + m), min(h, int(np.ceil(c[:, 1].max())) + m)
            if x1 <= x0 or y1 <= y0:
                return frame_bgr
            W, H = x1 - x0, y1 - y0
            M = inverse_affine.copy(); M[:, 2] -= (x0, y0)
        sub = frame_bgr[y0:y0 + H, x0:x0 + W]
        inv_restored = cv2.warpAffine(restored_bgr, M, (W, H))
        mask = np.ones((FACE_SIZE, FACE_SIZE), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, M, (W, H))
        inv_mask_erosion = cv2.erode(inv_mask, np.ones((2, 2), np.uint8))
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area ** 0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)[:, :, None]
        if parse_mask is not None:
            pm = cv2.warpAffine(parse_mask, M, (W, H), flags=3)[:, :, None]
            fuse_mask = (pm < inv_soft_mask).astype("int")
            inv_soft_mask = pm * fuse_mask + inv_soft_mask * (1 - fuse_mask)
        blended = (inv_soft_mask * pasted_face + (1 - inv_soft_mask) * sub).astype(np.uint8)   # upstream truncates, no rounding
        if stats is not None:
            ch = np.abs(blended.astype(np.int16) - sub.astype(np.int16)).max(axis=2) > 0
            ys, xs = np.nonzero(ch)
            sq = np.array([[0, 0, 1], [FACE_SIZE, 0, 1], [0, FACE_SIZE, 1], [FACE_SIZE, FACE_SIZE, 1]], dtype=np.float64) @ inverse_affine.T
            stats.update(roi=[x0, y0, x0 + W, y0 + H], square=[int(sq[:, 0].min()), int(sq[:, 1].min()), int(np.ceil(sq[:, 0].max())), int(np.ceil(sq[:, 1].max()))],
                         changed_px=int(ch.sum()), frame_px=int(h * w),
                         change_box=[int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + x0 + 1, int(ys.max()) + y0 + 1] if len(xs) else None,
                         soft_mask_px=int((inv_soft_mask[..., 0] > 0.01).sum()))
        if not roi:
            return blended
        out = frame_bgr.copy(); out[y0:y0 + H, x0:x0 + W] = blended
        return out

    @staticmethod
    def parse_mask_from_labels(labels: np.ndarray) -> np.ndarray:
        """ParseNet argmax labels (512x512) -> soft mask in [0, 1], the exact facelib recipe (two 101x101 sigma-11 blurs, 10 px border), CPU."""
        pm = MASK_COLORMAP[labels]
        pm = cv2.GaussianBlur(pm, (BLUR_K, BLUR_K), BLUR_SIGMA)
        pm = cv2.GaussianBlur(pm, (BLUR_K, BLUR_K), BLUR_SIGMA)
        pm[:PARSE_BORDER, :] = 0; pm[-PARSE_BORDER:, :] = 0; pm[:, :PARSE_BORDER] = 0; pm[:, -PARSE_BORDER:] = 0
        return pm / 255.0

    def _parse_masks_gpu(self, labels: torch.Tensor) -> torch.Tensor:
        """Same recipe on the GPU for a batch: cv2.GaussianBlur == separable convolution with BORDER_REFLECT_101 (= torch 'reflect')."""
        m = self._cmap[labels].unsqueeze(1)
        r = BLUR_K // 2
        for _ in range(2):
            m = Fnn.conv2d(Fnn.pad(m, (r, r, 0, 0), mode="reflect"), self._kx)
            m = Fnn.conv2d(Fnn.pad(m, (0, 0, r, r), mode="reflect"), self._ky)
        m[:, :, :PARSE_BORDER, :] = 0; m[:, :, -PARSE_BORDER:, :] = 0; m[:, :, :, :PARSE_BORDER] = 0; m[:, :, :, -PARSE_BORDER:] = 0
        return (m / 255.0)[:, 0]

    # ------------------------------------------------------------------ GPU
    def _to_input(self, crops_bgr: list[np.ndarray]) -> torch.Tensor:
        n = len(crops_bgr)
        buf = self._buf[:n]
        x = np.stack(crops_bgr)[..., ::-1]                             # BGR -> RGB (img2tensor bgr2rgb=True)
        buf.copy_(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))))
        buf.div_(255.0).sub_(0.5).div_(0.5)                            # normalize((0.5,)*3, (0.5,)*3)
        return buf.to(self.device, non_blocking=self.pinned)

    @staticmethod
    def _to_uint8_bgr(out: torch.Tensor) -> list[np.ndarray]:
        """tensor2img(rgb2bgr=True, min_max=(-1, 1)) for the whole batch on the GPU: clamp, [0,1], *255, round, uint8, RGB->BGR."""
        o = ((out.float().clamp(-1, 1) + 1) / 2 * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).flip(-1).contiguous().cpu().numpy()
        return list(o)

    def restore_crops(self, crops_bgr: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray] | None, dict]:
        """Batched `net(x, w, adain=True)` (+ ParseNet on the restored faces + mask blur). Returns restored BGR uint8 faces, soft parse masks, timings."""
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        x = self._to_input(crops_bgr)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.autocast):
            out = self.net(x, w=self.w, adain=True)[0]
        restored = self._to_uint8_bgr(out)
        torch.cuda.synchronize(); t["net"] = time.perf_counter() - t0
        masks = None
        if self.parse is not None:
            t0 = time.perf_counter()
            with torch.no_grad():
                pin = self._to_input(restored)                          # upstream feeds the RESTORED face (512, normalized like the net input)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.parse_autocast):
                    labels = self.parse(pin)[0].argmax(dim=1)
                if self.mask_blur == "gpu":
                    masks = list(self._parse_masks_gpu(labels).cpu().numpy())
                else:
                    masks = list(self._pool.map(self.parse_mask_from_labels, labels.cpu().numpy()))
            torch.cuda.synchronize(); t["parse"] = time.perf_counter() - t0
        del out
        if self.empty_cache_per_face:
            torch.cuda.empty_cache()
        return restored, masks, t

    # ------------------------------------------------------------------ frames
    def _align_batch(self, frames_bgr, chunk):
        affines = [self.affine_for(lm) for _, lm in chunk]
        crops = list(self._pool.map(lambda z: self.align_crop(frames_bgr[z[0]], z[1]), [(i, af) for (i, _), af in zip(chunk, affines)]))
        return affines, crops

    def _paste_batch(self, frames_bgr, out, chunk, affines, crops, restored, masks, st):
        for k, _ in enumerate(chunk):
            if is_gray_bgr(crops[k]):                                  # FaceRestoreHelper.add_restored_face gray path
                from facelib.utils.misc import adain_npy, bgr2gray
                restored[k] = adain_npy(bgr2gray(restored[k]), crops[k]); st["gray_faces"] += 1
        stats = [dict() for _ in chunk]
        outs = list(self._pool.map(lambda z: self.paste_back(frames_bgr[z[0]], z[1], z[2], z[3], self.roi_paste, z[4]),
                                   [(i, restored[k], af, masks[k] if masks else None, stats[k]) for k, ((i, _), af) in enumerate(zip(chunk, affines))]))
        for (i, _), fr, sd in zip(chunk, outs, stats):
            out[i] = fr
            if sd:
                loc = st["locality"]; loc["frames"] += 1; loc["changed_px_sum"] += sd["changed_px"]; loc["soft_mask_px_sum"] += sd["soft_mask_px"]
                loc["roi_px_sum"] += (sd["roi"][2] - sd["roi"][0]) * (sd["roi"][3] - sd["roi"][1]); loc["frame_px"] = sd["frame_px"]
                cb, sq = sd["change_box"], sd["square"]
                if cb is not None:   # pixels changed outside the warped 512 face square = the feather / parse-mask blur zone only
                    loc["change_outside_square_max_px"] = max(loc["change_outside_square_max_px"], max(0, sq[0] - cb[0]), max(0, sq[1] - cb[1]), max(0, cb[2] - sq[2]), max(0, cb[3] - sq[3]))
        st["faces_restored"] += len(chunk)

    def restore_frames(self, frames_bgr, landmarks5: list, inplace: bool = False) -> tuple[list, dict]:
        """frames_bgr: list/array of HxWx3 uint8 BGR frames (LatentSync output); landmarks5[i]: 5x2 array or None.
        Returns (frames, stats). Frames without landmarks are returned unchanged (or detected with fallback_detector)."""
        t_all = time.perf_counter(); n = len(frames_bgr)
        st = {"frames_in": n, "faces_restored": 0, "frames_without_landmarks": 0, "fallback_detections": 0, "gray_faces": 0,
              "seconds": {"landmarks": 0.0, "align": 0.0, "net": 0.0, "parse": 0.0, "paste": 0.0, "gpu_wait": 0.0}, "batches": 0,
              "locality": {"frames": 0, "changed_px_sum": 0, "soft_mask_px_sum": 0, "roi_px_sum": 0, "frame_px": 0, "change_outside_square_max_px": 0}}
        out = list(frames_bgr) if not inplace else frames_bgr
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter(); todo: list[tuple[int, np.ndarray]] = []
        for i in range(n):
            lm = landmarks5[i] if i < len(landmarks5) else None
            if lm is None and self.fallback_detector is not None:
                lm = self.fallback_detector(frames_bgr[i]); st["fallback_detections"] += 1
            if lm is None:
                st["frames_without_landmarks"] += 1; continue
            todo.append((i, np.asarray(lm, dtype=np.float32)))
        st["seconds"]["landmarks"] = time.perf_counter() - t0
        chunks = [todo[b:b + self.B] for b in range(0, len(todo), self.B)]
        pending = None                                                  # (chunk, affines, crops, future) of the batch on the GPU
        for chunk in chunks + [None]:
            if chunk is not None:
                t0 = time.perf_counter(); affines, crops = self._align_batch(frames_bgr, chunk); st["seconds"]["align"] += time.perf_counter() - t0
                fut = self._gpu.submit(self.restore_crops, crops) if self.overlap else None
            if pending is not None:
                pc, pa, pcr, pf = pending
                t0 = time.perf_counter(); restored, masks, tt = pf.result(); st["seconds"]["gpu_wait"] += time.perf_counter() - t0
                for k, v in tt.items():
                    st["seconds"][k] += v
                t0 = time.perf_counter(); self._paste_batch(frames_bgr, out, pc, pa, pcr, restored, masks, st); st["seconds"]["paste"] += time.perf_counter() - t0
                st["batches"] += 1
            if chunk is None:
                break
            if self.overlap:
                pending = (chunk, affines, crops, fut)
            else:
                restored, masks, tt = self.restore_crops(crops)
                for k, v in tt.items():
                    st["seconds"][k] += v
                t0 = time.perf_counter(); self._paste_batch(frames_bgr, out, chunk, affines, crops, restored, masks, st); st["seconds"]["paste"] += time.perf_counter() - t0
                st["batches"] += 1; pending = None
        st["seconds"]["total"] = round(time.perf_counter() - t_all, 3)
        for k in list(st["seconds"]):
            st["seconds"][k] = round(st["seconds"][k], 3)
        loc = st["locality"]
        if loc["frames"]:
            loc["changed_fraction_of_frame_mean"] = round(loc["changed_px_sum"] / loc["frames"] / max(loc["frame_px"], 1), 4)
            loc["roi_fraction_of_frame_mean"] = round(loc["roi_px_sum"] / loc["frames"] / max(loc["frame_px"], 1), 4)
            loc["changed_px_per_soft_mask_px"] = round(loc["changed_px_sum"] / max(loc["soft_mask_px_sum"], 1), 3)
        st["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
        st["faces_per_s"] = round(st["faces_restored"] / max(st["seconds"]["total"], 1e-6), 2)
        self.calls.append(st)
        for k in ("frames_in", "faces_restored", "frames_without_landmarks", "fallback_detections", "gray_faces"):
            self.totals[k] += st[k]
        self.totals["seconds"] = round(self.totals["seconds"] + st["seconds"]["total"], 3)
        return out, st

    def close(self) -> None:
        self._pool.shutdown(wait=True); self._gpu.shutdown(wait=True)

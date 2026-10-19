#!/usr/bin/env python3
"""LatentSync crop / paste geometry fixes (track A, round 2, 2026-10-17; measured in reports/experiments/lipsync-temporal/r2/STAGE1.md).

fp32_alignrestore()   LatentSync's AlignRestore defaults to float16 on CUDA: the 2x3 affine, kornia's sampling grid and the paste-back
                      inverse are float16, so translations of ~300-500 px are quantised to 0.25-0.5 px and the whole pasted face is
                      re-registered slightly differently on every frame (nose-base slide 0.12-0.27 px/frame with NO generation vs
                      0.013-0.044 px in float32). Same class, float32 default.
subpixel_anchors()    ImageProcessor.affine_transform rounds the three affine anchors to integer pixels (np.round) before the similarity
                      fit; the same function without the rounding.
Both patch the LatentSync classes in this process; call before the pipeline builds its ImageProcessor.
"""
from __future__ import annotations
import numpy as np


def fp32_alignrestore():
    import torch
    import latentsync.utils.affine_transform as atm
    if getattr(atm.AlignRestore, "_dub_fp32", False): return
    orig = atm.AlignRestore.__init__

    def init(self, align_points=3, resolution=256, device="cpu", dtype=torch.float32):
        orig(self, align_points, resolution, device, torch.float32)
    atm.AlignRestore.__init__ = init; atm.AlignRestore._dub_fp32 = True


def subpixel_anchors():
    import cv2, torch
    from einops import rearrange
    import latentsync.utils.image_processor as ipm
    if getattr(ipm.ImageProcessor, "_dub_subpixel", False): return

    def affine_transform(self, image):
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")
        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)
        landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose], np.float64)   # upstream: np.round(...)
        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix
    ipm.ImageProcessor.affine_transform = affine_transform; ipm.ImageProcessor._dub_subpixel = True


def apply(affine_precision: str = "fp16", anchor_round: str = "on") -> dict:
    if affine_precision == "fp32": fp32_alignrestore()
    if anchor_round == "off": subpixel_anchors()
    return {"affine_precision": affine_precision, "anchor_round": anchor_round}

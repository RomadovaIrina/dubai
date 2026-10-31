"""r9 speed L5c (LS_ROI=1): LatentSync's per-frame face alignment (align_warp_face) and paste-back (restore_img) on a crop around the face
instead of the whole frame (upstream uploads every full frame to the GPU and warps / erodes / blurs a full-frame mask: ~60 ms per frame at 1080p).
Same operations and dtype on a sub-window whose margin keeps every non-zero mask pixel and blur tail inside; the frame outside the crop is the
input itself (upstream: soft mask 0 -> the input pixel). Not bit-identical: kornia normalises the sampling grid by the image size in fp16.
Applied by latentsync_accel.AccelLatentSync.image_processor()."""
from __future__ import annotations
import math, types
import numpy as np, cv2, torch, kornia
from einops import rearrange


def _bbox(M, w, h, W, H, margin):
    c = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64); p = c @ M[:, :2].T + M[:, 2]
    x0, y0 = max(0, int(math.floor(p[:, 0].min())) - margin), max(0, int(math.floor(p[:, 1].min())) - margin)
    x1, y1 = min(W, int(math.ceil(p[:, 0].max())) + margin + 1), min(H, int(math.ceil(p[:, 1].max())) + margin + 1)
    return x0, y0, x1, y1


def align_warp_face(self, img, landmarks3, smooth=True):
    affine_matrix, self.p_bias = self.transformation_from_points(landmarks3, self.face_template, smooth, self.p_bias)
    A = np.asarray(affine_matrix, np.float64).reshape(2, 3); H, W = img.shape[:2]; fw, fh = self.face_size
    x0, y0, x1, y1 = _bbox(cv2.invertAffineTransform(A), fw, fh, W, H, 3)   # the source pixels the warp samples (+ bilinear neighbours)
    if x1 - x0 < 8 or y1 - y0 < 8: x0, y0, x1, y1 = 0, 0, W, H
    A2 = A.copy(); A2[:, 2] += A[:, :2] @ np.array([x0, y0], np.float64)
    crop = rearrange(torch.from_numpy(np.ascontiguousarray(img[y0:y1, x0:x1])).to(device=self.device, dtype=self.dtype), "h w c -> c h w").unsqueeze(0)
    a2 = torch.from_numpy(A2).to(device=self.device, dtype=self.dtype).unsqueeze(0)
    face = kornia.geometry.transform.warp_affine(crop, a2, (fh, fw), mode="bilinear", padding_mode="fill", fill_value=self.fill_value)
    face = rearrange(face.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
    return face, torch.from_numpy(A).to(device=self.device, dtype=self.dtype).unsqueeze(0)   # the FULL-frame matrix (restore_img inverts it)


def restore_img(self, input_img, face, affine_matrix):
    H, W, _ = input_img.shape
    if isinstance(affine_matrix, np.ndarray): affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)
    inv = kornia.geometry.transform.invert_affine_transform(affine_matrix); Mi = inv[0].float().cpu().numpy()
    mh, mw = self.mask.shape[-2:]; side = max(mh, mw) * float(np.sqrt(abs(np.linalg.det(Mi[:, :2]))))
    x0, y0, x1, y1 = _bbox(Mi, mw, mh, W, H, int(0.3 * side) + 8)   # margin > erosion radius + blur radius (both ~ face side / 10)
    inv2 = inv.clone(); inv2[0, 0, 2] -= x0; inv2[0, 1, 2] -= y0; h, w = y1 - y0, x1 - x0
    face = face.to(dtype=self.dtype).unsqueeze(0)
    inv_face = kornia.geometry.transform.warp_affine(face, inv2, (h, w), mode="bilinear", padding_mode="fill", fill_value=self.fill_value).squeeze(0)
    inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255
    crop = rearrange(torch.from_numpy(np.ascontiguousarray(input_img[y0:y1, x0:x1])).to(device=self.device, dtype=self.dtype), "h w c -> c h w")
    inv_mask = kornia.geometry.transform.warp_affine(self.mask, inv2, (h, w), padding_mode="zeros")
    inv_mask_erosion = kornia.morphology.erosion(inv_mask, torch.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype))
    pasted_face = inv_mask_erosion.squeeze(0).expand_as(inv_face) * inv_face
    w_edge = int(torch.sum(inv_mask_erosion.float()) ** 0.5) // 20; erosion_radius = w_edge * 2
    ime = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
    center = torch.from_numpy(cv2.erode(ime, np.ones((erosion_radius, erosion_radius), np.uint8))).to(device=self.device, dtype=self.dtype)[None, None, ...]
    blur = w_edge * 2 + 1; sigma = 0.3 * ((blur - 1) * 0.5 - 1) + 0.8
    soft = kornia.filters.gaussian_blur2d(center, (blur, blur), (sigma, sigma)).squeeze(0).expand_as(inv_face)
    back = rearrange(soft * pasted_face + (1 - soft) * crop, "c h w -> h w c").contiguous().to(dtype=torch.uint8).cpu().numpy()
    out = input_img.copy(); out[y0:y1, x0:x1] = back
    return out


def apply(restorer):
    restorer.align_warp_face = types.MethodType(align_warp_face, restorer); restorer.restore_img = types.MethodType(restore_img, restorer)
    restorer.roi = True

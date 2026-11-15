"""r20 (mouth): a TIME-LOCAL identity / reference frame for KeySync keyframes.
Why: the keyframe model was trained with the identity frame taken right next to the generated window (sgm/data/video_dataset_latent.py:
need_cond = one extra frame before the clip), so the reference shows the light, the pose and the mouth of that moment. At inference the authors
(and we, KS_REF_IDX) use ONE frame for the whole video. Seen on 04 (a selfie walk: frame 0 is dark and green under trees): the drawn lower face
carries the tone of frame 0 for the whole video (the band above the lip), and a closed-mouth reference gives the model no teeth -> a flat black cavity.
Only the keyframe model sees the identity frame (the interpolation model is conditioned on the two keyframes), so a per-batch reference costs nothing.
KS_LREF=<plan.json> (ks_fast.py: a non-numeric 4th job column):
  {"refs": [r0, r1, ...]}   the reference frame (index in the job clip) of keyframe batch b = frames [169 b, 169 b + 168]; the last entry repeats
  "slot0": 1                 the anchor slot of every batch (the authors put frame 0 there: add_zero_flag) = the batch reference instead of frame 0
  "kf_refs": [[14 frames] per batch]   a reference per keyframe slot (ks_lrefpick --kf-yaw: pose-matched frames on head turns)
No plan = the pipeline untouched. KS_KF_DUMP=<dir>: the decoded keyframes as jpg (debug)."""
import json, os
import torch

ST = {"plan": None, "emb": None, "vid": None, "aud": None, "lmk": None, "b": 0, "kw": {}, "sig": []}


def guide_hook(g):
    """KS_KF_MODE (keyframe guider only; the 3 rows are x0 predictions: uncond | ref | audio + ref; outside the latent mask they are equal):
      sched:S:AR:RR      below sigma S the audio / ref ratios drop to AR / RR (full guidance sets the mouth shape at high noise; the late steps only
                         add contrast: the over-shoot that crushes the open mouth to flat black)
      rescale:PHI        the guided prediction is rescaled toward the spread of the conditional one inside the mask (CFG rescale, PHI 0..1)"""
    cls = type(g); call0 = cls.__call__
    if getattr(cls, "_ks_r20", False): return
    def call(self, x, sigma):
        mode = os.environ.get("KS_KF_MODE", "")
        if self is not g or not mode: return call0(self, x, sigma)
        m = mode.split(":"); e_uc, e_ref, c = x.chunk(3); ar0, rr0 = self.audio_ratio, self.ref_ratio; ar, rr = ar0, rr0; sv = float(sigma.flatten()[0])
        if sv not in ST["sig"]: ST["sig"].append(sv)
        if m[0] == "sched" and sv < float(m[1]): ar, rr = float(m[2]), float(m[3])
        if m[0] == "hi" and sv > float(m[1]): ar, rr = float(m[2]), float(m[3])   # hi:S:AR:RR = the ratios ABOVE sigma S (the first steps)
        self.audio_ratio, self.ref_ratio = ar, rr   # the per-step ratios go through the attributes and the PREVIOUS __call__ (the authors formula or another wrapper,
        try: e = call0(self, x, sigma)              # e.g. r18-quality ks_quiet.py: per-row audio ratio in silence), so the wrappers compose in either order
        finally: self.audio_ratio, self.ref_ratio = ar0, rr0
        mm = dict(zip(m[0::2], m[1::2])) if m[0] in ("mix",) else {}   # mix:floor:K:reflp:K2:audlp:K3 = several rules at once
        if m[0] == "reflp" or "reflp" in mm:   # reflp:K = the LOW frequencies (tone, light) of what the reference adds are not amplified (K = 1: ratio 1 for them); its detail keeps the ref ratio. The band of 04 = the tone of frame 0 doubled
            k = float(mm.get("reflp", m[1] if m[0] == "reflp" else 0)); dr = e_ref - e_uc
            lo = torch.nn.functional.avg_pool2d(dr, 7, 1, 3, count_include_pad=False); e = e - (rr - 1) * k * lo
        if m[0] == "audlp" or "audlp" in mm:   # audlp:K = the same for the audio term: the low frequencies (the big dark blob of an open mouth) at ratio 1 + (ar - 1)(1 - K)
            k = float(mm.get("audlp", m[1] if m[0] == "audlp" else 0)); da = c - e_ref
            lo = torch.nn.functional.avg_pool2d(da, 5, 1, 2, count_include_pad=False); e = e - (ar - 1) * k * lo
        if "floor" in mm: m = ["floor", mm["floor"]]
        if m[0] == "floor":   # floor:K[:T[:W]] = in the darks (the conditional prediction darker than T, soft over W) the guidance may not push the LUMA below the conditional prediction (K = how much of that over-shoot is removed); shape, chroma and the lit areas keep the full guidance
            k = float(m[1]); T = float(m[2]) if len(m) > 2 else -0.3; Wd = float(m[3]) if len(m) > 3 else 0.3
            u = e.new_tensor([0.277, 0.416, 0.050, -0.299]).view(1, 4, 1, 1)   # luma of the SD latent (the usual latent -> RGB linear fit, BT.601 weights)
            Ye, Yc = (e * u).sum(1, keepdim=True), (c * u).sum(1, keepdim=True)
            e = e + u * ((Yc - Ye).clamp_min(0) * ((T - Yc) / Wd).clamp(0, 1) * k) / float((u * u).sum())
        if m[0] == "floorv" or "floorv" in mm:   # floorv:K[:T[:W]] = as floor, but the WHOLE over-shoot (all latent channels, so the colour too: luma alone leaves a navy cavity) is pulled back to the conditional prediction where that one is dark and the guidance made it darker
            mv = m if m[0] == "floorv" else ["floorv"] + mm["floorv"].split(",")
            k = float(mv[1]); T = float(mv[2]) if len(mv) > 2 else -0.3; Wd = float(mv[3]) if len(mv) > 3 else 0.3
            u = e.new_tensor([0.277, 0.416, 0.050, -0.299]).view(1, 4, 1, 1); Ye, Yc = (e * u).sum(1, keepdim=True), (c * u).sum(1, keepdim=True)
            w = ((T - Yc) / Wd).clamp(0, 1) * ((Yc - Ye) / 0.1).clamp(0, 1) * k; e = e - w.clamp(0, 1) * (e - c)
            # the K2 gate (ks_gate.py) keys on black pixels of the decoded keyframes; the floored cells are the pixels that WOULD have been black: recorded per sampling run
            if ST.get("fw_sig") is None or sv > ST["fw_sig"] * 1.0001: ST.setdefault("fw_runs", []).append(torch.zeros_like(w[:, 0]))
            ST["fw_sig"] = sv
            if sv <= float(os.environ.get("KS_FLOOR_GSIG", "2")): ST["fw_runs"][-1] = torch.maximum(ST["fw_runs"][-1], (Ye[:, 0] < float(os.environ.get("KS_FLOOR_GY", "-0.8"))).to(w.dtype))   # would-be-black: the guided luma BEFORE the pull-back under the gate luma (0.18 of 0..1 = -0.64)
        if m[0] == "rescale":
            w = ((c - e_uc).abs().sum(1, keepdim=True) > 1e-4).to(e.dtype); n = w.sum((1, 2, 3), keepdim=True).clamp_min(1.0) * e.shape[1]
            mu_c = (c * w).sum((1, 2, 3), keepdim=True) / n; mu_e = (e * w).sum((1, 2, 3), keepdim=True) / n
            sd_c = (((c - mu_c) ** 2 * w).sum((1, 2, 3), keepdim=True) / n).sqrt(); sd_e = (((e - mu_e) ** 2 * w).sum((1, 2, 3), keepdim=True) / n).sqrt().clamp_min(1e-6)
            r = ((e - mu_e) * (sd_c / sd_e) + mu_e); phi = float(m[1]); e = torch.where(w > 0, phi * r + (1 - phi) * e, e)
        return e
    cls.__call__ = call; cls._ks_r20 = True


def install(d):
    cpi0, sk0 = d.create_pipeline_inputs, d.sample_keyframes

    def create_pipeline_inputs(video, audio, audio_interpolation, num_frames, video_emb, landmarks, **k):
        p = os.environ.get("KS_LREF", ""); ST["plan"] = None; ST["b"] = 0; ST["fw_last"] = None; ST["fw_chunk"] = None
        if p:
            ST["plan"] = json.load(open(p)); ST.update(emb=video_emb, vid=video, aud=audio, lmk=landmarks, kw=k)
            print("KS_LREF", p, "refs", ST["plan"]["refs"], "slot0", ST["plan"].get("slot0", 0), flush=True)
        return cpi0(video, audio, audio_interpolation, num_frames, video_emb, landmarks, **k)

    def dump(x, b):
        dd = os.environ.get("KS_KF_DUMP", "")
        if not dd: return
        dd = os.path.join(dd, os.environ.get("KS_KF_TAG", ""))
        import cv2, numpy as np
        os.makedirs(dd, exist_ok=True); g = ((x.float() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        for j in range(g.shape[0]):
            cv2.imwrite(os.path.join(dd, "b%03d_s%02d_f%05d.jpg" % (b, j, 13 * (13 * b + j - 1) if j else 0)), (g[j][..., ::-1] * 255).astype(np.uint8))

    def sample_keyframes(model_keyframes, audio_list, gt_list, masks_list, condition, num_frames, fps_id, cond_aug, device, embbedings, *a, **k):
        P = ST["plan"]; zs, xs = [], []; ST["fw_runs"] = []; ST["fw_sig"] = None; ST["fw_chunk"] = None
        g = model_keyframes.sampler.guider   # (tests) per-job keyframe guidance: KS_KF_AR / KS_KF_RR / KS_KF_NORM read at every call when KS_KF_LIVE=1
        if os.environ.get("KS_KF_MODE") is not None: guide_hook(g)
        if os.environ.get("KS_KF_LIVE") == "1":
            if "ar0" not in ST: ST["ar0"], ST["rr0"] = g.audio_ratio, g.ref_ratio
            g.audio_ratio = float(os.environ.get("KS_KF_AR") or ST["ar0"]); g.ref_ratio = float(os.environ.get("KS_KF_RR") or ST["rr0"]); g.use_normalized = os.environ.get("KS_KF_NORM") == "1"
        if not P and not os.environ.get("KS_KF_DUMP") and not os.environ.get("KS_KF_ONLY"):
            r_ = sk0(model_keyframes, audio_list, gt_list, masks_list, condition, num_frames, fps_id, cond_aug, device, embbedings, *a, **k)
            ST["fw_chunk"] = torch.cat(ST["fw_runs"]) if ST["fw_runs"] else None; return r_
        for i in range(audio_list.shape[0]):
            au, gt, mk, cond, emb = audio_list[i:i + 1], gt_list[i:i + 1], masks_list[i:i + 1], condition, embbedings
            if P:
                refs = P["refs"]; r = int(refs[min(ST["b"], len(refs) - 1)]); r = max(0, min(r, len(ST["emb"]) - 1))
                cond = ST["vid"][r].unsqueeze(0).to(condition); emb = ST["emb"][r].unsqueeze(0).to(embbedings)
                if P.get("slot0"):
                    au, gt, mk = au.clone(), gt.clone(), mk.clone()
                    gt[0, 0] = ST["emb"][r].to(gt); au[0, 0] = ST["aud"][r].to(au)
                    mk[0, 0] = d.load_landmarks(ST["lmk"][r:r + 1], ST["vid"].shape[-2:], r, target_size=(64, 64), is_dub=True,
                                                what_mask=ST["kw"].get("what_mask", "box"), nose_index=ST["kw"].get("nose_index", 28))[0].to(mk)
            KR = (P.get("kf_refs") or [None])[min(ST["b"], len(P["kf_refs"]) - 1)] if P and P.get("kf_refs") else None
            cn = model_keyframes.conditioner; g0 = cn.get_unconditional_conditioning
            if KR and len(set(KR)) > 1:   # a reference PER KEYFRAME: the identity latent is concatenated per frame (DubbingWrapper takes (t c) channels), so each keyframe can carry its own pose-matched frame
                def guc(batch, batch_uc=None, force_uc_zero_embeddings=None, **kw):
                    c_, uc_ = g0(batch, batch_uc=batch_uc, force_uc_zero_embeddings=force_uc_zero_embeddings, **kw); cats = {}
                    for r_ in set(KR):
                        b2 = dict(batch); rr_ = max(0, min(int(r_), len(ST["emb"]) - 1)); b2["cond_frames"] = ST["emb"][rr_].unsqueeze(0).to(batch["cond_frames"])
                        if "cond_frames_without_noise" in b2: b2["cond_frames_without_noise"] = ST["vid"][rr_].unsqueeze(0).to(batch["cond_frames_without_noise"])
                        cats[r_] = g0(b2, batch_uc=batch_uc, force_uc_zero_embeddings=force_uc_zero_embeddings, **kw)[0]["concat"]
                    c_["concat"] = torch.cat([cats[r_] for r_ in KR[:num_frames]], dim=1); uc_["concat"] = torch.cat([uc_["concat"]] * num_frames, dim=1)
                    return c_, uc_
                cn.get_unconditional_conditioning = guc
            try: z, x = sk0(model_keyframes, au, gt, mk, cond, num_frames, fps_id, cond_aug, device, emb, *a, **k)
            finally:
                if "get_unconditional_conditioning" in cn.__dict__: del cn.__dict__["get_unconditional_conditioning"]
            dump(x, ST["b"]); zs.append(z); xs.append(x); ST["b"] += 1
        ST["fw_chunk"] = torch.cat(ST["fw_runs"]) if ST["fw_runs"] else None
        if os.environ.get("KS_KF_DUMP") and ST["fw_chunk"] is not None:
            import numpy as np; dd_ = os.path.join(os.environ["KS_KF_DUMP"], os.environ.get("KS_KF_TAG", "")); os.makedirs(dd_, exist_ok=True); np.save(os.path.join(dd_, "fw_b%03d.npy" % (ST["b"] - len(ST["fw_runs"]))), ST["fw_chunk"].float().cpu().numpy())
        return torch.cat(zs), torch.cat(xs)

    si0 = d.sample_interpolation
    def sample_interpolation(model, samples_z, samples_x, audio_interpolation_list, *a, **k):   # KS_KF_ONLY=1 (tests): keyframes only, black frames instead of the interpolation
        gm = __import__("sys").modules.get("ks_gate"); fwc = ST.get("fw_chunk")
        if gm is not None:
            gm.ST["extra"] = None
            if fwc is not None:
                seq = ([ST["fw_last"]] if ST.get("fw_last") is not None and samples_x.shape[0] == fwc.shape[0] + 1 else []) + list(fwc)
                if len(seq) == samples_x.shape[0]:
                    tr = k.get("to_remove") or []; kept = [m for i, m in zip(tr, seq) if not i] if len(tr) else seq
                    gm.ST["extra"] = torch.stack(kept).float().cpu().numpy() if kept else None
                ST["fw_last"] = seq[-1] if seq else None
        if os.environ.get("KS_KF_ONLY") != "1": return si0(model, samples_z, samples_x, audio_interpolation_list, *a, **k)
        import numpy as np
        return np.zeros((13 * len(audio_interpolation_list) + 1, 3, samples_x.shape[-2], samples_x.shape[-1]), np.uint8)
    d.create_pipeline_inputs = create_pipeline_inputs; d.sample_keyframes = sample_keyframes; d.sample_interpolation = sample_interpolation

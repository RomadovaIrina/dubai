"""Chatterbox fp16 speech-token dtype guard (pipeline-level wrapper; no upstream / third-party file is edited).

Bug (found in the first_res intelligibility review, reports/quality/first_res_tts_intelligibility_diagnosis.md): the production runner casts
Chatterbox to fp16. chatterbox S3Token2Mel.forward (s3gen.py) then casts EVERY ref_dict tensor, including the int64 `prompt_token`, to fp16, and
`torch.concat([prompt_token, token])` in flow.inference promotes the GENERATED speech-token ids to fp16 as well. Integers > 2048 are not
representable in fp16 (> 4096: multiples of 4): ~44 % of the 6561 ids change by up to +-2 -> mumbled / swallowed / "aaa" speech
(unit CTC CER 0.395 -> 0.083 and Whisper WER 0.745 -> 0.084 once the ids stay integer; quality equals fp32, speed stays fp16). Log symptom:
"6560.0>6561 out-of-range special tokens found in flow" (fp16(6561) == 6560).

Fix: keep an int64 copy of the prompt tokens taken right after `prepare_conditionals` (or from the built-in conditionals) and hand THAT to flow.inference,
so the concat stays integer. fp32 models are unaffected (the wrapper is then a no-op with identical values).

    from chatterbox_fp16 import install_speech_token_guard
    guard = install_speech_token_guard(m)      # right after ChatterboxMultilingualTTS.from_local(), BEFORE the first generate / fp16 cast
    ...; m.prepare_conditionals(ref)           # captured automatically
    ...; m.conds = builtin.to(device="cuda"); guard.use_builtin()
"""
from __future__ import annotations

import torch


class SpeechTokenGuard:
    def __init__(self, m):
        self.m = m
        self.prompt = None      # (prompt_token int64, prompt_token_len int64) handed to flow.inference
        self.builtin = None
        self.calls = 0
        self.violations = 0
        self.capture_builtin()

    @staticmethod
    def _int_copy(gen: dict):
        pt, ptl = gen["prompt_token"], gen["prompt_token_len"]
        if torch.is_floating_point(pt):   # already cast (a generate() ran on these conds): ids may be rounded - refuse to use them silently
            raise RuntimeError("prompt_token is already floating point; capture the conditionals before the first generate()")
        return pt.detach().clone().long(), ptl.detach().clone().long()

    def capture_builtin(self):
        conds = getattr(self.m, "conds", None)
        if conds is not None:
            self.builtin = self._int_copy(conds.gen)
            self.prompt = self.builtin

    def capture_current(self):
        self.prompt = self._int_copy(self.m.conds.gen)

    def use_builtin(self):
        if self.builtin is None:
            raise RuntimeError("no built-in conditionals were captured")
        self.prompt = self.builtin


def install_speech_token_guard(m) -> SpeechTokenGuard:
    flow = m.s3gen.flow
    if getattr(flow, "_dabai_token_guard", None) is not None:
        return flow._dabai_token_guard
    g = SpeechTokenGuard(m)
    orig_inference = flow.inference
    orig_prepare = m.prepare_conditionals

    def guarded_inference(*a, **k):
        if g.prompt is not None:
            k["prompt_token"], k["prompt_token_len"] = g.prompt
        if torch.is_floating_point(k["token"]) or torch.is_floating_point(k["prompt_token"]):
            g.violations += 1
            raise RuntimeError("speech-token ids reached flow.inference as floating point (fp16 corruption)")
        g.calls += 1
        return orig_inference(*a, **k)

    def prepare(*a, **k):
        r = orig_prepare(*a, **k)
        g.capture_current()
        return r

    flow.inference = guarded_inference
    m.prepare_conditionals = prepare
    flow._dabai_token_guard = g
    return g

cd /workspace/dub/dubai-speed; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai-speed DUB_ALLOW_NC=1
B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a; R=/workspace/dub/r9/face; mkdir -p $R
# r9-a, worktree experiment/r9-speed-face. b0 = r9 + fix (E2E_FX look), then the stack, then half-rate / VAE compile on the stack; each gated vs b0
V=_b0 bash scripts/r9/face_only.sh m03
S="GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast CFD_EXTRA=--threads=12 FAST_MPREUSE=1 LS_DEC_ROW0=15 FC_EXTRA=--pre-threads=12"
for spec in "_S:" "_Sh:LS_HALF_RATE=1" "_Svc:LS_VAE_COMPILE=max-autotune-no-cudagraphs" "_Svw:LS_VAE_COMPILE=max-autotune-no-cudagraphs"; do
  v=${spec%%:*}; x=${spec#*:}
  env $S $x V=$v bash scripts/r9/face_only.sh m03
  python scripts/r9/gate_cmp.py --base $R/m03_b0 --cand $R/m03$v --md $R/gate_rows.md | head -1
done
echo QA1_DONE
# the stack fails stick +2 (from composite / fallback): drop the mouth-only decode (largest pixel change, smallest gain)
S2="GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast CFD_EXTRA=--threads=12 FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12"
env $S2 V=_S2 bash scripts/r9/face_only.sh m03
python scripts/r9/gate_cmp.py --base $R/m03_b0 --cand $R/m03_S2 --md $R/gate_rows.md | head -1
echo QA1B_DONE

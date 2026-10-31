cd /workspace/dub/dubai-speed; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai-speed DUB_ALLOW_NC=1
B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a; R=/workspace/dub/r9/face
while pgrep -f "bash qa1.sh" > /dev/null; do sleep 20; done   # qa1 runs from its own shell (no ssh launcher left with this text)
g() { python scripts/r9/gate_cmp.py --base $R/$1 --cand $R/$2 --md $R/gate_rows.md > $R/$2/gate.txt 2>&1; head -1 $R/$2/gate.txt; }
S2="GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast CFD_EXTRA=--threads=12 FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12"
env $S2 LS_SKIP_SILENT=1 V=_S2k bash scripts/r9/face_only.sh m03; g m03_b0 m03_S2k
V=_b0 bash scripts/r9/face_only.sh m19
for spec in "_S2h:LS_HALF_RATE=1"; do v=${spec%%:*}; x=${spec#*:}; env $S2 $x V=$v bash scripts/r9/face_only.sh m19; g m19_b0 m19$v; done
# wrap-up: m01 (1080p60) dropped, it cannot finish before the servers go
echo QA2_DONE

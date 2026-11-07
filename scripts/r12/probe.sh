cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 LS_CMS_READ=1
flock /workspace/dub/gpu.lock python scripts/r12/probe_seg.py old03 3 7 /workspace/dub/outbox/r14_f/old03_probe_3-7s_q2_nd_fuse8_05x.mp4 "q2=dc" "nodc=nodc" "fuse8=nodc,stride8,fuse" "cont8=dc,stride8"
echo PROBE_DONE

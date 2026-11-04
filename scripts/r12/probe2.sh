cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 LS_CMS_READ=1
PROBE_SAVE=/workspace/dub/r14/probe2 python scripts/r12/probe_seg.py old03 4.5 8 /workspace/dub/outbox/r14_f/old03_probe_4.5-8s_sticky_ref_cfg_05x.mp4 "q2=dc" "ref+32=dc,shift32" "cfg3=dc,g3.0" "ref32+cfg3=dc,shift32,g3.0"
echo PROBE2_DONE

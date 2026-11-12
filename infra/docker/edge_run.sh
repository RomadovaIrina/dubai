#!/bin/bash
# r17 TZ 12.3 edge set on the VM: generate the 17 cases (scripts/r13/gen_edge.sh, ffmpeg) from our clips inside the image, then run
# scripts/r13/edge_suite.py (the CURRENT scripts/job interface) over them in ONE container (the worker must survive every case).
#   bash infra/docker/edge_run.sh <image> <raw dir: 01.MP4 02.mp4 03.mp4 04.mp4 05.mp4> <mamakris dir> <out dir> [cases, default: all but long_10min] [code dir]
set -uo pipefail
IMG=${1:?image}; RAW=$(readlink -f "${2:?raw 01-05}"); MK=$(readlink -f "${3:?mamakris}"); O=$(mkdir -p "${4:?out}" && readlink -f "$4"); C=${6:-}; VOL=${VOL:-dabai-models}
CASES=${5:-short_3s,no_audio,silence,music_only,fps60_conv,fps60_real,odd_fps_29.58_real,vfr,rotated_meta90,horizontal_1920x1080,square_720,res4k_vertical,hdr_hlg_10bit,three_faces,face_half_out,very_dark}
MNT=(); [ -n "$C" ] && MNT=(-v $C/scripts:/workspace/dub/dubai/scripts:ro -v $C/infra:/workspace/dub/dubai/infra:ro)
mkdir -p $O/in $O/out
docker run --rm "${MNT[@]}" -v $RAW:/raw:ro -v $MK:/mk:ro -v $O/in:/edge_in -e EDGE_T=/raw -e EDGE_M=/mk -e EDGE_O=/edge_in --entrypoint bash $IMG \
  -c 'cd /workspace/dub/dubai && sed "/^# 10+ min/,\$d" scripts/r13/gen_edge.sh > /tmp/gen.sh && bash /tmp/gen.sh' > $O/gen.log 2>&1   # (long_10min: done 12.11, skipped)
# (odd_fps_29.58_real: a real phone clip r13 had; gen_edge.sh does not make it -> reported missing, not failed)
echo "generated: $(ls $O/in | tr '\n' ' ')"; grep -i failed $O/gen.log | head -3
HAVE=$(for c in ${CASES//,/ }; do ls $O/in/$c.* > /dev/null 2>&1 && echo -n "$c,"; done); HAVE=${HAVE%,}
echo "running: $HAVE"
docker run --rm --gpus all --shm-size 16g "${MNT[@]}" -v $VOL:/workspace/dub/dubai/models -v $O/in:/edge/in:ro -v $O/out:/edge/out --entrypoint bash $IMG \
  -c "cd /workspace/dub/dubai && export DUB_PROFILE=clean DUB_ALLOW_NC=0 DUB_ALLOW_NC_BED=0 && bash infra/docker/entrypoint.sh true && /venv/api/bin/python scripts/r13/edge_suite.py --in /edge/in --out /edge/out --cases $HAVE" > $O/suite.log 2>&1
grep -E "^== |EDGE_SUITE" $O/suite.log

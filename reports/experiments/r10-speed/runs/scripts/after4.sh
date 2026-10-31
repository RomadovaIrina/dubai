#!/bin/bash
# r10: resident generator worker: exactness (m03 x2 through the worker = the cold-process md5), then the 4-video queue, then the UNet compile re-test
until grep -q M01_DONE /workspace/dub/r10/after2.log; do sleep 30; done
cd /workspace/dub/dubai; SP=/workspace/dub/r10/gw_test
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
rm -rf $SP; bash scripts/r10/gen_worker.sh start $SP
for v in _gw1 _gw2; do R10_GEN_SPOOL=$SP R10_GEN_PY=scripts/r10/gen_client.py METRICS=off V=$v bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing"
  f=$(ls /workspace/dub/artifacts/r4m/r9g_m03$v/work_*/*_25fps_ls_raw.mp4); echo "EXACT $v $(ffmpeg -loglevel error -i $f -map 0:v -f md5 - 2>/dev/null) (cold process: MD5=3330d733ba2ee20b93c78eacdd80fe89)"; done
bash scripts/r10/gen_worker.sh stop $SP; grep "gen_worker\]" $SP/worker.log
echo GW_EXACT_DONE
unset GEN_COMP FAST_SRCPREP FAST_FB_BG FAST_ENC FAST_MPREUSE FC_EXTRA LS_VAE_COMPILE CFD_EXTRA
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q4.txt --out /workspace/dub/r10/q4_i2_gw --inflight 2 --gen-worker --env E2E_METRICS=off 2>&1 | grep -E "QUEUE_DONE|done "
echo GWQ_DONE
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
LS_UNET_COMPILE=default METRICS=off V=_uc1 bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing"
LS_UNET_COMPILE=default V=_uc2 bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing\|metrics done"
echo UC2_DONE

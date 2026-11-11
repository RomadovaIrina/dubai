#!/bin/bash
# r16 TZ: r16-q software = the IMAGE recipe (infra/docker/install_env.sh venvs / latentsync / api / llama_cuda / check + ks_install.sh) +
# setup_spleeter.sh, with every pinned file prefetched by wheel_fetch.py (aria2c) and handed to uv as UV_FIND_LINKS (one PyPI connection here
# = 0.06-0.2 MB/s). Unpinned closures (KeySync extras, spleeter) resolved with `uv pip compile` (metadata only) first.
set -u
R=/workspace/dub/dubai-tz; H=$R/infra/docker; W=/workspace/dub/wheels; WL=/workspace/dub/wheels_llama; D=/workspace/dub/wheels_req; L=/workspace/dub/setup2.log
E=/workspace/dub/dlenv/bin/python; PT=https://download.pytorch.org/whl/cu128
log(){ echo "[$(date +%T)] $*" >> $L; }
mkdir -p $W $WL $D; cd $D; log "== setup2 start"
cp $H/requirements.dabai.lock.txt $H/requirements.mediapipe.lock.txt $H/requirements.main.lock.txt $H/requirements.api.lock.txt $R/scripts/keysync/req_ks.txt $D/
printf "torch==2.7.1\ntorchvision==0.22.1\ntorchaudio==2.7.1\nomegaconf==2.3.0\nantlr4-python3-runtime==4.9.3\nface-alignment==1.4.1\ntorchao==0.11.0\nhf_xet\ntensorrt-cu12==10.16.1.11\nonnx==1.18.0\n" > ks_extra.in
timeout 1200 uv pip compile -q --python-version 3.11 --python-platform x86_64-manylinux_2_28 --index-url $PT --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match ks_extra.in -o ks_extra.txt >> $L 2>&1 && log "ks_extra $(grep -c == ks_extra.txt) pins" || log "ks_extra compile failed (network fallback)"
printf 'spleeter==2.3.2\ntensorflow==2.9.3\nnumpy==1.22.4\nnumba==0.55.2\nllvmlite==0.38.1\nwrapt<1.15\nsetuptools<70\nsoundfile\n' > spleeter.in
timeout 1200 uv pip compile -q --python-version 3.10 --python-platform x86_64-manylinux_2_28 spleeter.in -o spleeter.txt >> $L 2>&1 && log "spleeter $(grep -c == spleeter.txt) pins" || log "spleeter compile failed"
S="--set 3.11:requirements.dabai.lock.txt --set 3.11:requirements.mediapipe.lock.txt --set 3.11:requirements.main.lock.txt --set 3.11:requirements.api.lock.txt --set 3.11:req_ks.txt"
[ -s ks_extra.txt ] && S="$S --set 3.11:ks_extra.txt"; [ -s spleeter.txt ] && S="$S --set 3.10:spleeter.txt"
$E $R/scripts/keysync/r16/tz/wheel_fetch.py --out $W --llama-out $WL $S >> $L 2>&1; log "WHEEL_FETCH rc $?"
# spleeter 2stems weights (GitHub release, MIT; sha256 = setup_spleeter.sh) with 8 connections, so the script finds them
SM=$R/models/spleeter/pretrained_models; SHA=$(grep -o "^SHA=[0-9a-f]*" $R/scripts/setup_spleeter.sh | cut -d= -f2)
if [ ! -f $SM/2stems/model.index ]; then T=$(mktemp -d); aria2c -q -x 8 -s 8 -d $T -o 2stems.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz \
  && echo "$SHA  $T/2stems.tar.gz" | sha256sum -c - >> $L 2>&1 && mkdir -p $SM/2stems && tar -xzf $T/2stems.tar.gz -C $SM/2stems && log "spleeter weights ok"; rm -rf $T; fi
export UV_FIND_LINKS=$W UV_HTTP_TIMEOUT=600
rm -rf /venv/dabai   # my own partial setup_dabai venv (stage 0 + part of 1, created 19:22) -> the image recipe from scratch
( cd $R && DUB_ROOT=$R bash $H/install_env.sh venvs ) >> $L.venvs 2>&1 && log VENVS_DONE || log VENVS_FAILED
( cd $R && DUB_ROOT=$R bash $H/install_env.sh latentsync api ) >> $L.venvs 2>&1 && log LS_API_DONE || log LS_API_FAILED
( cd $R && UV_FIND_LINKS=$WL DUB_ROOT=$R JOBS=64 bash $H/install_env.sh llama_cuda ) > $L.llama 2>&1 && log LLAMA_CUDA_DONE || log LLAMA_CUDA_FAILED &
( KS_NO_WEIGHTS=1 bash $R/scripts/keysync/r16/ks_install.sh ) > $L.ks 2>&1 && log KS_INSTALL_DONE || log KS_INSTALL_FAILED &
( cd $R && bash scripts/setup_spleeter.sh ) > $L.spleeter 2>&1 && log SPLEETER_DONE || log SPLEETER_FAILED &
wait
( cd $R && DUB_ROOT=$R bash $H/install_env.sh check ) >> $L 2>&1 && log CHECK_DONE || log CHECK_FAILED
/venv/dabai/bin/python $R/scripts/voice_env_check.py --python /venv/dabai/bin/python >> $L 2>&1 && log VOICE_ENV_OK || log VOICE_ENV_BAD
log "uv network downloads (slow path): $(cat $L.* | grep -c '^Downloading\| Downloading ')"
log "== setup2 end"

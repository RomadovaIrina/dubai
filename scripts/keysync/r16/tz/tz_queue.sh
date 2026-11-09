#!/bin/bash
# r16 TZ overnight queue on r16-q, one phase at a time after the fault suite (every job through dabai_job.run_job / job_api, the clean env tzenv.sh):
#   o23   O1-O3: forced real failures (weights / voice env / GPU lock -> ERROR codes; probe conf / check offset / route / dissolve -> PARTIAL + warning)
#   rc    e2e.sh exit codes for the 4 ERROR paths (direct e2e runs)
#   edge  12.3 edge set on KeySync     long  12.3 the 10.5-min video (RAM / disk / time)     subs  9.1-9.3 q01 subs + watermark, zh / ar / hi
#   load  12.4 the real job_api under a burst of 6 jobs
set -u; Q=/workspace/dub/r16tz; IN=$Q/in; L=$Q/queue.log; E="bash /workspace/dub/tzenv.sh"; S="$E /venv/dabai/bin/python scripts/keysync/r16/tz_suite.py"
log(){ echo "[$(date +%T)] $*" >> $L; }
phase(){ [ -f $Q/$1.done ] && { log "skip $1"; return 1; }; log "start $1"; return 0; }
while pgrep -f "[k]s_fault_suite.py" > /dev/null; do sleep 20; done; log "fault suite finished: $(tail -1 $Q/fault/suite.log | cut -c1-200)"
DUB=$(ls -d /workspace/dub/artifacts/r4a/*/ksf_k_ok 2>/dev/null | head -1); log "dub for reuse: $DUB"
if phase kwarm; then   # the fault suite's clean job ran into the 1800 s job timeout on the COLD worker (TRT build + VAE inductor compile): again, warm;
  # the dub-reuse jobs again too: their route found no manifest (the reused <old id>_r4b.manifest.json; fixed in e2e.sh) -> a spurious face_route_failed
  $E /venv/dabai/bin/python scripts/keysync/r16/ks_fault_suite.py --single $IN/m18_20s.mp4 --shots $IN/two_shots_cut.mp4 --out $Q/fault_warm --jobs k_ok,k_crash,k_few,k_many \
     > $Q/fault_warm.log 2>&1; touch $Q/kwarm.done; log "kwarm: $(grep -E '^(PASS|FAIL)' $Q/fault_warm.log | cut -c1-300)"
fi
M18=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; M03=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV
[ -f $IN/m03_20s.mp4 ] || ffmpeg -loglevel error -y -i $M03 -t 20 -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 160k $IN/m03_20s.mp4
[ -f $IN/old03_20s.mp4 ] || ffmpeg -loglevel error -y -i /workspace/dub/inputs/test_videos_raw/03.mp4 -t 20 -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 160k $IN/old03_20s.mp4
if phase o23; then
  mkdir -p $Q/o23/empty_hf/hub $Q/o23/fake_nr; echo 'raise ImportError("r16 TZ test: noisereduce hidden from the voice env check")' > $Q/o23/fake_nr/noisereduce.py
  $S --out $Q/o23 --job o_w_hubert=$IN/m18_20s.mp4,E.HF_HOME=$Q/o23/empty_hf --job o_w_lora=$IN/m18_20s.mp4,E.KS_LORA=/nonexistent/lora_pick.pt \
     --job o_voice=$IN/m18_20s.mp4,E.PYTHONPATH=$Q/o23/fake_nr --job o_gpulock=$IN/m18_20s.mp4,E.R10_HEAVY_LOCK=/nonexistent_tz/gpu.lock,E.GPU_LOCK=/nonexistent_tz/gpu.lock \
     --job o1_lowconf=$IN/m18_20s.mp4,E.KS_PROBE_MIN_CONF=99,E.DUB_AUDIO_FROM=$DUB --job o1_checkoff=$IN/m18_20s.mp4,E.KS_DELAY=200,E.DUB_AUDIO_FROM=$DUB \
     --job o1_route=$IN/m18_20s.mp4,E.DUB_ROUTE_DEVICE=bogus,E.DUB_AUDIO_FROM=$DUB --job o1_dis_black=$IN/dissolve_fadeblack.mp4 --job o1_dis_fade=$IN/dissolve_fade.mp4 \
     > $Q/o23/suite.log 2>&1 && touch $Q/o23.done; log "o23: $(tail -1 $Q/o23/suite.log)"
fi
if phase rc; then
  O=$Q/o23/rc; mkdir -p $O
  for c in "w_hubert HF_HOME=$Q/o23/empty_hf" "w_lora KS_LORA=/nonexistent/lora_pick.pt" "voice PYTHONPATH=$Q/o23/fake_nr" \
           "gpulock R10_HEAVY_LOCK=/nonexistent_tz/gpu.lock GPU_LOCK=/nonexistent_tz/gpu.lock"; do
    set -- $c; n=$1; shift; rm -rf $O/$n; mkdir -p $O/$n
    env "$@" ID=tzrc_$n E2E_METRICS=off E2E_FRAME_GATE=off $E bash scripts/r9/e2e.sh $IN/m18_20s.mp4 $O/$n > $O/$n.log 2>&1; rc=$?
    echo "$n e2e_rc=$rc manifest: $(python3 -c "import json; m=json.load(open('$O/$n/manifest.json')); print(m['status'], m.get('errors'))" 2>&1 | cut -c1-300)" >> $O/rc.txt
  done; touch $Q/rc.done; log "rc: $(tr '\n' '|' < $O/rc.txt)"
fi
if phase edge; then
  J=""; for n in no_face tiny_face two_faces two_shots_cut res4k_vertical silence music_only no_audio short_3s vfr rotated_meta90 hdr_hlg_10bit fps60_real \
                 face_half_out very_dark horizontal_1920x1080 square_720 odd_fps_29.58_real; do J="$J --job $n=$IN/$n.mp4"; done
  mkdir -p $Q/edge; $S --out $Q/edge $J > $Q/edge/suite.log 2>&1 || true; touch $Q/edge.done; log "edge: $(tail -1 $Q/edge/suite.log)"
fi
if phase long; then mkdir -p $Q/long; $S --out $Q/long --job long10=$IN/long_10min.mp4 > $Q/long/suite.log 2>&1; touch $Q/long.done; log "long: $(tail -1 $Q/long/suite.log)"; fi
if phase subs; then
  mkdir -p $Q/subs; $S --out $Q/subs --job q01_subs=/workspace/dub/r16in/day1/HANDOFF_Q/clips/q01_amir.mp4,subs=yellow,wm=DabAI,keep=1 \
     --job zh=$IN/m18_20s.mp4,lang=zh,subs=yellow,keep=1 --job ar=$IN/m18_20s.mp4,lang=ar,subs=yellow,keep=1 --job hi=$IN/m18_20s.mp4,lang=hi,subs=yellow,keep=1 \
     > $Q/subs/suite.log 2>&1; touch $Q/subs.done; log "subs: $(tail -1 $Q/subs/suite.log)"
fi
if phase load; then
  mkdir -p $Q/load; $E python3 /workspace/dub/tz_load.py --out $Q/load $IN/m18_20s.mp4 $IN/m03_20s.mp4 $IN/old03_20s.mp4 $IN/two_shots_cut.mp4 $IN/dissolve_fade.mp4 $IN/tiny_face.mp4 \
     > $Q/load/load.log 2>&1; touch $Q/load.done; log "load: $(tail -1 $Q/load/load.log | cut -c1-400)"
fi
log "QUEUE_DONE"

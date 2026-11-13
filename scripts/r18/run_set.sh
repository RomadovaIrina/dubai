#!/bin/bash
# r18: restore 08.11 + the seam candidates for a list of videos. Detached on the VM:
#   systemd-run --unit r18-run bash /root/r18/code/scripts/r18/run_set.sh m03 old03 long5
# Per video: dub (the accepted 08.11 mix, stream-copied; the voice KeySync hears) -> ONE e2e job on the image (the GPU part) -> CPU chain in the
# background while the next job runs: re-pastes (oval = self-test of the re-paste path, a, b, c, c5) -> r18_check (audio md5 / colr / outside the
# box vs the 08.11 file) -> Q's detectors vs the source. Progress: log/r18.log; per video out/<id>/{restored,oval,a,b,c,c5}.mp4 + *.check.json + det/.
# The job itself runs with KS_MASK=oval (restored = the 08.11 paste). R18_STEPS (default "dub job post") limits the steps; a finished job is kept unless R18_REDO=1.
set -uo pipefail
B=${R18_ROOT:-/root/r18}; CODE=${R18_CODE:-$B/code}; S=$CODE/scripts/r18; STEPS=${R18_STEPS:-dub job post}; RT=reports/experiments/r16-qa/routing
log(){ echo "[$(date -u +%T)] $*" | tee -a $B/log/r18.log; }
declare -A INP REF VOX RTE
row(){ INP[$1]=$2; REF[$1]=$3; VOX[$1]=$4; RTE[$1]=$5; }
#   id     input (in/)                                    the 08.11 file (ref/)      accepted voice dir     route
row m03   mamakris/story-1774249919482-367803614.MOV      f0811_m03.mp4              $B/vox/m03             $RT/m03.json
row m18   mamakris/story-1774250817106-616905094.mp4      full_m18.mp4               ""                     off
row old03 03.mp4                                          full_old03.mp4             ""                     off
row long5 long5_stories.mp4                               long5.mp4                  ""                     ${R18_LONG5_ROUTE:-off}
row q01   clips/q01_amir.mp4                              f0811_q01.mp4              ""                     $RT/q01_amir.json
row q02   clips/q02_vera.mp4                              f0811_q02.mp4              ""                     $RT/q02_vera.json
row q03   clips/q03_priest.mp4                            q03_e2e_alpha_full.mp4     $B/vox/q03             $RT/q03_priest.json
row q04   clips/q04_farikh.mp4                            f0811_q04.mp4              ""                     $RT/q04_farikh.json
row q05   clips/q05_rozhkov.mp4                           q05_e2e_full.mp4           $B/vox/q05             $RT/q05_rozhkov.json
row q06   clips/q06_suvorov.mp4                           f0811_q06.mp4              ""                     $RT/q06_suvorov.json
A2="--mask-tsmooth 2 --neck-follow 0.15 --mask-chin-down 0.06"
long5_src(){   # the 08.11 recipe (r16d_render.sh): 8 mamakris stories -> 720x1280 25 fps, 300 s
  [ -s $B/in/long5_stories.mp4 ] && return 0; log "long5_stories.mp4: building"
  bash $S/r18.sh sh 'set -e; mkdir -p /r18/in/long5 && cd /r18/in/long5 && rm -f list.txt; i=0
    for s in story-1773579769284-320619146.MP4 story-1774249930391-60765207.MOV story-1774250573471-666527943.MP4 story-1774250795589-441186215.mp4 story-1774250832175-746471562.mp4 story-1774250884789-692894647.mp4 story-1774250904597-834383232.mp4 story-1773579769284-320619146.MP4; do
      ffmpeg -v error -y -i /r18/in/mamakris/$s -vf "fps=25,scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1" -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p -c:a aac -ar 48000 -ac 2 -b:a 192k p$i.mp4; echo "file p$i.mp4" >> list.txt; i=$((i+1)); done
    ffmpeg -v error -y -f concat -safe 0 -i list.txt -t 300 -c copy /r18/in/long5_stories.mp4' > $B/log/long5_src.log 2>&1 || { log "long5 source build FAILED"; return 1; }; }
post(){ local id=$1 O=$B/out/$1 W=/data/jobs/$1/e2e/work; mkdir -p $O/det
  [ -s $B/data/jobs/$id/$id.mp4 ] || { log "POST $id: no job output"; return 1; }
  cp $B/data/jobs/$id/$id.mp4 $O/restored.mp4; cp $B/data/jobs/$id/status.json $O/status.json 2> /dev/null
  bash $S/r18.sh repaste $id oval KS_MASK=oval KS_TAIL_FADE=off   # (KS_MASK is always explicit: the code / image default is hybrid since Irina's pick 13.11)
  bash $S/r18.sh repaste $id a KS_MASK=oval "KS_PLAIN_EXTRA=--mask-tsmooth 2"
  bash $S/r18.sh repaste $id b KS_MASK=oval "KS_PLAIN_EXTRA=--mask tight --mask-chin 1 $A2"
  bash $S/r18.sh repaste $id c KS_MASK=hybrid
  bash $S/r18.sh repaste $id c5 KS_MASK=oval "KS_PLAIN_EXTRA=--mask hybrid --hybrid-at 0.5 $A2"
  log "SELFTEST $id restored vs oval re-paste: $(bash $S/r18.sh sh "/venv/dabai/bin/python scripts/keysync/r16/tz/out_compare.py --a /r18/out/$id/restored.mp4 --b /r18/out/$id/oval.mp4 --src /r18/in/${INP[$id]} --json /r18/out/$id/selftest.json" 2>&1 | grep OUT_COMPARE | cut -c1-300)"
  for v in restored a b c c5; do [ -s $O/$v.mp4 ] || continue
    log "CHECK $id $v: $(bash $S/r18.sh sh "/venv/dabai/bin/python scripts/r18/r18_check.py --new /r18/out/$id/$v.mp4 --ref /r18/ref/${REF[$id]} --work $W --out /r18/out/$id/$v.check.json" 2>&1 | grep R18_CHECK | cut -c1-420)"; done
  for v in ref restored a b c c5; do F=/r18/out/$id/$v.mp4; [ $v = ref ] && F=/r18/ref/${REF[$id]}; [ $v = ref ] || [ -s $O/$v.mp4 ] || continue   # Q's detectors (CPU), 3 at a time
    ( bash $S/r18.sh sh "nice -n 10 /venv/mediapipe/bin/python scripts/qa16/r16q_defects.py --src /r18/in/${INP[$id]} --ours $F --out /r18/out/$id/det/$v.json --npz /r18/out/$id/det/$v.det.npz --label ${id}_$v" > $O/det/$v.log 2>&1
      log "DET $id $v: $(tail -1 $O/det/$v.log | cut -c1-400)" ) &
    while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 5; done; done; wait
  log "POST_DONE $id"; }
BG=()
for id in "$@"; do [ -n "${INP[$id]:-}" ] || { log "unknown video $id"; continue; }
  [ $id = long5 ] && { long5_src || continue; }
  if [[ " $STEPS " == *" dub "* ]] && [ ! -s $B/dubs/$id/dubbed_16k.wav ]; then bash $S/r18.sh dub $id ${REF[$id]} ${VOX[$id]} || continue; fi
  if [[ " $STEPS " == *" job "* ]] && { [ "${R18_REDO:-0}" = 1 ] || [ ! -s $B/data/jobs/$id/$id.mp4 ]; }; then bash $S/r18.sh job $id ${INP[$id]} ${RTE[$id]} KS_MASK=oval KS_TAIL_FADE=off; fi
  if [[ " $STEPS " == *" post "* ]]; then post $id & BG+=($!); fi
done
for p in "${BG[@]:-}"; do [ -n "$p" ] && wait $p; done
log "R18_SET_DONE $*"

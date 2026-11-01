cd /workspace/dub/dubai; R=/workspace/dub/r10/runs; mkdir -p $R
( while true; do for p in $(pgrep -f "quality/fps_composite.py"); do h=$(grep VmHWM /proc/$p/status 2>/dev/null | awk "{print \$2}"); [ -n "$h" ] && echo "$(date +%T) $p $h" >> $R/m01_composite_rss.log; done; sleep 5; done ) & SAMP=$!
rm -rf $R/m01; E2E_FX=1 E2E_METRICS=off ID=m01 bash scripts/r9/e2e.sh /workspace/dub/inputs/mamakris/story-1773579769284-320619146.MP4 $R/m01 > $R/m01.out 2>&1
kill $SAMP; grep R9_E2E_DONE $R/m01.out; echo "peak composite VmHWM KB: $(sort -k3 -n $R/m01_composite_rss.log | tail -n 1)"; echo M01_DONE

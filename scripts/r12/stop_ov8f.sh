# stop the full-video fuse arm of gen1.sh before it starts (user rule: no ceiling runs); the short probe replaces it
until [ -f /workspace/dub/artifacts/r4m/r9g_old03ov8f/README.txt ] || grep -q GEN1_DONE /workspace/dub/r14/logs/gen1.out; do sleep 5; done
for p in $(pgrep -f "[l]s_arm.sh old03 ov8f"); do pkill -TERM -P $p; kill $p; done; pkill -f "[r]7_e2e.sh old03" ; pkill -f "[c]andidate_video_stage.*r9g_old03ov8f"
echo "STOPPED ov8f $(date +%T)"

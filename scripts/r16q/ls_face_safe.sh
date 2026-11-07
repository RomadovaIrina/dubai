# r16 Q: each id in its own session (setsid); after it ends (or is killed) every leftover in that process group is killed,
# so no orphan (e.g. r4_video.sh nvidia-smi -lms 500) can keep an inherited gpu.lock fd. Never kill -9 the runner: kill the group.
IDS="${*:-q01_amir q03_priest q04_farikh}"; for id in $IDS; do
  setsid bash /workspace/dub/r16q/ls_face.sh $id & P=$!; wait $P; pkill -g $P 2>/dev/null
done; echo LS_FACE_ALL_DONE

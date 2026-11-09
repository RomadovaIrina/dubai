#!/bin/bash
# r16 TZ 11.2 / 11.3 CPU test: collect.py turns the KeySync substituted-frame counts into video_status + the 2 % rule
set -u; R=/workspace/dub/dubai-tz; T=$(mktemp -d); cd $R; ok=1
printf "audio\t1000.0\nface_keysync\t1010.0\nend\t1020.0\n" > $T/stages.tsv; echo '{"probe": {"duration_s": 20.0}}' > $T/precheck.json
run(){   # run <name> <subst json> <expected status> <expected flag or ->
  echo "$2" > $T/subst.json; mkdir -p $T/$1
  python3 scripts/r9/collect.py --id t --out $T/$1 --status SUCCESS --flags "face_keysync" --errors "" --stages $T/stages.tsv --nvsmi /nonexistent --precheck $T/precheck.json --ks-subst $T/subst.json > $T/$1.log 2>&1
  python3 - $T/$1/manifest.json "$3" "$4" "$1" <<'PY' || ok=0
import json, sys
m = json.load(open(sys.argv[1])); want_s, want_f, name = sys.argv[2], sys.argv[3], sys.argv[4]; vs = m["spec"]["video_status"] or {}
good = m["status"] == want_s and (want_f == "-" or want_f in m["flags"]) and (want_f != "-" or "frames_substituted_over_2pct" not in m["flags"])
print(("PASS " if good else "FAIL ") + f"{name}: status {m['status']} flags {m['flags']} video_status {json.dumps({k: vs.get(k) for k in ('eligible_lipsync_frames', 'substituted_frames', 'degraded_ratio', 'status')})}")
sys.exit(0 if good else 1)
PY
}
run few  '{"eligible_frames": 1000, "guard_replaced": 10, "failed_frames": 0}' SUCCESS -
run many '{"eligible_frames": 1000, "guard_replaced": 50, "failed_frames": 0}' PARTIAL_SUCCESS frames_substituted_over_2pct
run shot '{"eligible_frames": 1000, "guard_replaced": 5, "failed_frames": 300, "failed_shots": 1}' PARTIAL_SUCCESS frames_substituted_over_2pct
run zero '{"eligible_frames": 1000, "guard_replaced": 0, "failed_frames": 0}' SUCCESS -
echo "" > $T/subst.json; mkdir -p $T/none
python3 scripts/r9/collect.py --id t --out $T/none --status SUCCESS --flags "face_keysync" --errors "" --stages $T/stages.tsv --nvsmi /nonexistent --precheck $T/precheck.json --ks-subst /nonexistent > /dev/null 2>&1 \
  && python3 -c "import json; m=json.load(open('$T/none/manifest.json')); print('PASS none: no subst file -> status', m['status'], 'video_status', m['spec']['video_status'])" || { echo "FAIL none"; ok=0; }
rm -rf $T; [ $ok = 1 ] && echo KS_SUBST_TESTS_OK || echo KS_SUBST_TESTS_FAILED

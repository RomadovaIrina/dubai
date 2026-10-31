cd /workspace/dub/r10/eq
cmp_() {   # decoded-frame comparison of two lossless outputs
/venv/dabai/bin/python - "$1" "$2" <<'PY'
import sys, numpy as np, av
ca, cb = av.open(sys.argv[1]), av.open(sys.argv[2]); n = 0; d = 0; mx = 0
for fa, fb in zip(ca.decode(video=0), cb.decode(video=0)):
    x = fa.to_ndarray(format="bgr24"); y = fb.to_ndarray(format="bgr24"); n += 1
    if not np.array_equal(x, y): d += 1; mx = max(mx, int(np.abs(x.astype(np.int16) - y).max()))
print(f"   decoded frames compared {n}, differing {d}, max diff {mx}")
PY
}
for spec in "m03 crf=0" "old03 crf=0" "m03 crf=0,reg_smooth=4"; do set -- $spec; id=$1; S_=$2
  echo "== $id $S_"; EQ_V="fps_composite fps_composite_stream" bash /workspace/dub/dubai/scripts/r10/run_eq_test.sh $id "$S_" 2>&1 | grep -v "^  \(fallback\|sanitize\|sharpen\|reg-smooth\|paste\)"
  t=${S_//[=,]/_}; cmp_ ${id}_${t}_fps_composite.mp4 ${id}_${t}_fps_composite_stream.mp4
done
echo EQ_ALL_DONE

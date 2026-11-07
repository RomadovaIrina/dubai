"""r13 colour-managed ffmpeg I/O for the KeySync paste chain (KS_CMS=1; off = the old commands).
ffmpeg 6.1 converts YUV <-> RGB with its BT.601 default and ignores the BT.709 tags, and every decode -> rgb -> x264 round trip of our
intermediates lost ~1.7 L (measured on the old03 master, 50 frames, even at -qp 0); the KeySync box goes through two more round trips than the
rest of the frame, so it came out ~3-4 L darker than its surroundings. With KS_CMS=1 every RGB conversion is explicit BT.709 tv range with
accurate rounding and the outputs are tagged BT.709: 0.0 L drift per round trip."""
import os
ON = os.environ.get("KS_CMS", "0") == "1"
_F = "flags=accurate_rnd+full_chroma_int"
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
def dec(vf=None, pix="bgr24"):   # the -vf of a decode to raw rgb
    if not ON: return ["-vf", vf] if vf else []
    return ["-vf", (vf + "," if vf else "") + f"scale=in_color_matrix=bt709:in_range=tv:{_F},format={pix}"]
def enc():                       # replaces "-pix_fmt yuv420p" of an rgb -> x264 encode
    if not ON: return ["-pix_fmt", "yuv420p"]
    return ["-vf", f"scale=out_color_matrix=bt709:out_range=tv:{_F},format=yuv420p"] + TAGS

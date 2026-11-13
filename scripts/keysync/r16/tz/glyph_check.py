#!/usr/bin/env python3
"""r16 TZ 9.x (11.11; r18 13.11): automatic subtitle glyph check for the scripts Irina can't judge by eye (zh / ar / hi, + ru / en controls), inside the image.

r18 finding (13.11): the r17 run said "hi FAIL, shaping diff 0.0". Cause: this check burned with ffmpeg's `subtitles=` filter, and that filter
sets libass to the SIMPLE shaper (its private `shaping` field is 0 = ASS_SHAPING_SIMPLE; only the `ass=` filter has the option, default -1 =
libass's own default, COMPLEX with HarfBuzz). The simple shaper draws Devanagari without conjuncts (visible halants), so the plain and the
ZWNJ-isolated renders were the same picture. Arabic passed anyway: the simple shaper still joins Arabic through FriBidi + the presentation
forms in DejaVu Sans. Production (`scripts/r7/r7_subs.py burn`) uses `ass=` -> complex shaping -> correct Hindi.
Now every render here goes through the PRODUCTION code: r7_subs.make_ass (the real style, size, wrapping, karaoke) + r7_subs.burn.

  1. coverage: for every codepoint, the font fontconfig / libass falls back to for the style font ("DejaVu Sans" + the char) must hold a real
     glyph for it (fontTools cmap); the font per language is logged; a missing glyph = tofu = FAIL
  2. shaping: (a) hi: the production render must differ from the same ASS burned with `ass=...:shaping=simple` (the conjuncts; the check
     that would have caught the r17 test bug); (b) ar / hi: it must differ from a render with a ZWNJ between every letter (blocks joining /
     conjuncts); (c) Arabic must also differ from the codepoint-reversed string (bidi / RTL)
  3. the production burn uses `ass=` and never `subtitles=` (source check of r7_subs.burn)
  4. OCR round-trip (optional, --ocr): tesseract on the inverted, 2x frame of the production render (and of the simple render, for contrast),
     CER vs the source text (needs tesseract + the chi_sim / ara / hin data)
    python3 glyph_check.py --out DIR [--style yellow] [--ocr]      -> DIR/glyph_check.json + the PNG renders; exit 1 on any FAIL"""
import argparse, glob, inspect, json, os, pathlib, shutil, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "r7"))
import r7_subs  # noqa: E402  (the production subtitle code)

TEXT = {  # short subtitle-like lines; the ar / hi ones contain joining letters / conjuncts
    "ru": "Привет, как у тебя дела сегодня?",
    "en": "Hello, how are you doing today?",
    "zh": "你好，你今天过得怎么样？我们明天见。",
    "ar": "مرحبا، كيف حالك اليوم؟ نراك غدا.",
    "hi": "नमस्ते, आज आप कैसे हैं? क्षमा कीजिए, स्त्री और द्वार।",
}
LANG_FC = {"ru": "ru", "en": "en", "zh": "zh-cn", "ar": "ar", "hi": "hi"}
OCR_LANG = {"ru": "rus", "en": "eng", "zh": "chi_sim", "ar": "ara", "hi": "hin"}
ZWNJ = "‌"
W, H = 1280, 720


def fc_font(font, ch, lang):
    """the file fontconfig picks for this char (what libass does for a glyph missing in the style font)"""
    pat = f"{font}:charset={ord(ch):x}:lang={lang}"
    return subprocess.run(["fc-match", "-f", "%{file}", pat], capture_output=True, text=True).stdout.strip()


def has_glyph(path, ch, cache={}):
    from fontTools.ttLib import TTFont, TTCollection
    if path not in cache:
        cm = set()
        try:
            fonts = TTCollection(path).fonts if path.lower().endswith((".ttc", ".otc")) else [TTFont(path, lazy=True)]
            for f in fonts: cm |= set((f.getBestCmap() or {}).keys())
        except Exception as e: print("font read", path, e)
        cache[path] = cm
    return ord(ch) in cache[path]


def make_clip(out: pathlib.Path) -> str:
    clip = out / "black_2s.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=25:d=2",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(clip)], check=True)
    return str(clip)


def write_ass(text, path: pathlib.Path, style):
    """the production ASS: one cue 0-2 s, words spread over it (r7_subs.make_ass: style, size, wrapping, karaoke as in a job)"""
    words = text.split(); n = len(words)
    path.write_text(r7_subs.make_ass([(w, 0.05 + 1.9 * i / n, 0.05 + 1.9 * (i + 1) / n) for i, w in enumerate(words)], W, H, style, karaoke=True),
                    encoding="utf-8")
    return path


def burn_prod(ass: pathlib.Path, png: pathlib.Path, clip: str):
    """the production burn (r7_subs.burn, preview mode = one frame at t; t = 1.0 s: mid-karaoke)"""
    png.unlink(missing_ok=True)
    try: r7_subs.burn(clip, str(png), str(ass), preview_t=1.0)
    except subprocess.CalledProcessError: return False
    return png.exists()


def burn_simple(ass: pathlib.Path, png: pathlib.Path, clip: str):
    """the same ASS with libass's SIMPLE shaper (= what ffmpeg's `subtitles=` filter does)"""
    png.unlink(missing_ok=True)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", clip, "-filter_complex",
                        f"[0:v]ass='{r7_subs._esc(str(ass))}':shaping=simple,format=yuv420p,trim=start=1.0,setpts=PTS-STARTPTS[p]",
                        "-map", "[p]", "-frames:v", "1", str(png)], capture_output=True, text=True)
    return r.returncode == 0 and png.exists()


def diff(a, b):
    import numpy as np
    from PIL import Image
    x = np.asarray(Image.open(a).convert("L"), dtype=float); y = np.asarray(Image.open(b).convert("L"), dtype=float)
    return float(np.abs(x - y).mean())


def cer(ref, hyp):
    ref = "".join(ref.lower().split()); hyp = "".join(hyp.lower().split())   # case-free: the trendy style upper-cases
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        p, d[0] = d[0], i
        for j, h in enumerate(hyp, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def ocr(png: pathlib.Path, lang: str) -> str:
    """tesseract wants dark text on light: the lower half (where the cue is), every bright fill pixel (both karaoke colours) -> black,
    the black outline / background -> white, 2x"""
    from PIL import Image
    im = Image.open(png).convert("L"); im = im.crop((0, im.height // 2, im.width, im.height)).point(lambda v: 0 if v > 110 else 255)
    im = im.resize((im.width * 2, im.height * 2)); p = png.with_name(png.stem + "_ocr.png"); im.save(p)
    return subprocess.run(["tesseract", str(p), "-", "-l", lang, "--psm", "6"], capture_output=True, text=True).stdout.strip()


def cue_at(ass: pathlib.Path, t: float = 1.0) -> str:
    """the text of the cue on screen at t (make_ass may split a line into cues at a sentence end): tags and \\N removed"""
    import re
    for line in ass.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"): continue
        f = line.split(",", 9); h0, m0, s0 = f[1].split(":"); h1, m1, s1 = f[2].split(":")
        if int(h0) * 3600 + int(m0) * 60 + float(s0) <= t < int(h1) * 3600 + int(m1) * 60 + float(s1):
            return re.sub(r"\{[^}]*\}", "", f[9]).replace("\\N", " ")
    return ""


def cer_any_order(ref, hyp):
    """tesseract may print an RTL line in visual (reversed word) order: take the better of the two word orders"""
    return min(cer(ref, hyp), cer(ref, " ".join(hyp.split()[::-1])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--style", default="yellow", choices=list(r7_subs.STYLES))
    ap.add_argument("--ocr", action="store_true")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); clip = make_clip(out)
    font = r7_subs.STYLES[a.style]["font"]
    libass = (glob.glob("/usr/lib/x86_64-linux-gnu/libass.so*") or [""])[0]
    ldd = subprocess.run(["ldd", libass], capture_output=True, text=True).stdout if libass else ""
    src = inspect.getsource(r7_subs.burn); prod_filter_ok = "ass=" in src and "subtitles=" not in src
    ffv = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    res = {"font": font, "style": a.style, "ffmpeg": ffv, "libass": libass, "harfbuzz": "libharfbuzz" in ldd, "fribidi": "libfribidi" in ldd,
           "prod_burn_uses_ass_filter": prod_filter_ok, "langs": {}}
    ok_all = res["harfbuzz"] and res["fribidi"] and prod_filter_ok
    for lg, t in TEXT.items():
        chars = sorted({c for c in t if not c.isspace()}); fonts = {}; missing = []
        for c in chars:
            f = fc_font(font, c, LANG_FC[lg]); fonts.setdefault(os.path.basename(f), 0); fonts[os.path.basename(f)] += 1
            if not f or not has_glyph(f, c): missing.append(f"U+{ord(c):04X}")
        r = {"fonts": fonts, "missing_glyphs": missing, "coverage_ok": not missing}
        ass = write_ass(t, out / f"{lg}.ass", a.style); base = out / f"{lg}.png"; r["rendered"] = burn_prod(ass, base, clip)
        if lg in ("ar", "hi"):
            simp = out / f"{lg}_simple.png"; burn_simple(ass, simp, clip); r["complex_vs_simple_diff"] = round(diff(base, simp), 3)
            iso = out / f"{lg}_zwnj.png"; burn_prod(write_ass(ZWNJ.join(t), out / f"{lg}_zwnj.ass", a.style), iso, clip)
            r["zwnj_diff"] = round(diff(base, iso), 3); r["shaped"] = r["zwnj_diff"] > 0.5
            if lg == "hi": r["shaped"] = r["shaped"] and r["complex_vs_simple_diff"] > 0.5   # conjuncts exist only in the complex shaper
        if lg == "ar":
            rev = out / f"{lg}_reversed.png"; burn_prod(write_ass(t[::-1], out / f"{lg}_reversed.ass", a.style), rev, clip)
            r["rtl_diff"] = round(diff(base, rev), 3); r["rtl_ok"] = r["rtl_diff"] > 0.5
        if a.ocr and shutil.which("tesseract"):
            ref = cue_at(ass); r["ocr_ref"] = ref
            hyp = ocr(base, OCR_LANG[lg]); r["ocr"] = hyp; r["cer"] = round(cer_any_order(ref, hyp), 3)
            if lg in ("ar", "hi"):
                hs = ocr(out / f"{lg}_simple.png", OCR_LANG[lg]); r["ocr_simple"] = hs; r["cer_simple"] = round(cer_any_order(ref, hs), 3)
        r["pass"] = r["coverage_ok"] and r["rendered"] and r.get("shaped", True) and r.get("rtl_ok", True)
        ok_all &= r["pass"]; res["langs"][lg] = r
        print(f"{lg}: {'PASS' if r['pass'] else 'FAIL'} fonts {fonts} missing {missing[:8]} shaped {r.get('shaped', '-')} "
              f"(zwnj {r.get('zwnj_diff', '-')}, vs simple {r.get('complex_vs_simple_diff', '-')}) rtl {r.get('rtl_ok', '-')} "
              f"cer {r.get('cer', '-')} (simple {r.get('cer_simple', '-')})")
    res["pass"] = ok_all; (out / "glyph_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(ffv); print("libass", libass, "harfbuzz", res["harfbuzz"], "fribidi", res["fribidi"], "prod burn = ass filter", prod_filter_ok)
    print("GLYPH_CHECK", "PASS" if ok_all else "FAIL"); sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

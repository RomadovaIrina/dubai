#!/usr/bin/env python3
"""r16 TZ 9.x (11.11): automatic subtitle glyph check for the scripts Irina can't judge by eye (zh / ar / hi, + ru / en controls), inside the image.
  1. coverage: for every codepoint of every test string, the font fontconfig / libass falls back to for the style font ("DejaVu Sans" + the char)
     must hold a real glyph for it (fontTools cmap); the font per language is logged; a missing glyph = tofu = FAIL
  2. shaping (libass must be built with HarfBuzz): the line is burned with ffmpeg's subtitles filter twice, once as is and once with a ZWNJ
     between every letter (which blocks joining / conjuncts). Arabic and Devanagari renders MUST differ (shaped); Arabic must also differ from
     the codepoint-reversed string (bidi / RTL). The ldd of libass is logged (harfbuzz / fribidi linked)
  3. OCR round-trip (optional, --ocr): tesseract on the burned frame, CER vs the source text (needs tesseract + the chi_sim / ara / hin data)
    python3 glyph_check.py --out DIR [--font "DejaVu Sans"] [--ocr]      -> DIR/glyph_check.json + the PNG renders; exit 1 on any FAIL"""
import argparse, glob, json, os, pathlib, shutil, subprocess, sys

TEXT = {  # short subtitle-like lines; the ar / hi ones contain joining letters / conjuncts
    "ru": "Привет, как у тебя дела сегодня?",
    "en": "Hello, how are you doing today?",
    "zh": "你好，你今天过得怎么样？我们明天见。",
    "ar": "مرحبا، كيف حالك اليوم؟ نراك غدا.",
    "hi": "नमस्ते, आज आप कैसे हैं? क्षमा कीजिए, स्त्री और द्वार।",
}
LANG_FC = {"ru": "ru", "en": "en", "zh": "zh-cn", "ar": "ar", "hi": "hi"}
ZWNJ = "‌"


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


def burn(text, png, font, w=1280, h=200):
    ass = png.with_suffix(".ass")
    ass.write_text("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
                   "BackColour, Bold, Alignment, MarginV, BorderStyle, Outline, Shadow\nStyle: S,%s,56,&H00FFFFFF,&H00000000,&H00000000,0,5,10,1,2,0\n\n[Events]\n"
                   "Format: Layer, Start, End, Style, Text\nDialogue: 0,0:00:00.00,0:00:05.00,S,%s\n" % (w, h, font, text), encoding="utf-8")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:d=1", "-vf", f"subtitles={ass}",
                        "-frames:v", "1", str(png)], capture_output=True, text=True)
    return r.returncode == 0 and png.exists()


def diff(a, b):
    import numpy as np
    from PIL import Image
    x = np.asarray(Image.open(a).convert("L"), dtype=float); y = np.asarray(Image.open(b).convert("L"), dtype=float)
    return float(np.abs(x - y).mean())


def cer(ref, hyp):
    ref = "".join(ref.split()); hyp = "".join(hyp.split())
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        p, d[0] = d[0], i
        for j, h in enumerate(hyp, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--font", default="DejaVu Sans"); ap.add_argument("--ocr", action="store_true")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    libass = (glob.glob("/usr/lib/x86_64-linux-gnu/libass.so*") or [""])[0]
    ldd = subprocess.run(["ldd", libass], capture_output=True, text=True).stdout if libass else ""
    res = {"font": a.font, "libass": libass, "harfbuzz": "libharfbuzz" in ldd, "fribidi": "libfribidi" in ldd, "langs": {}}
    ok_all = res["harfbuzz"] and res["fribidi"]
    for lg, t in TEXT.items():
        chars = sorted({c for c in t if not c.isspace()}); fonts = {}; missing = []
        for c in chars:
            f = fc_font(a.font, c, LANG_FC[lg]); fonts.setdefault(os.path.basename(f), 0); fonts[os.path.basename(f)] += 1
            if not f or not has_glyph(f, c): missing.append(f"U+{ord(c):04X}")
        r = {"fonts": fonts, "missing_glyphs": missing, "coverage_ok": not missing}
        base = out / f"{lg}.png"; r["rendered"] = burn(t, base, a.font)
        if lg in ("ar", "hi"):
            iso = out / f"{lg}_zwnj.png"; burn(ZWNJ.join(t), iso, a.font); r["shaping_diff"] = round(diff(base, iso), 3); r["shaped"] = r["shaping_diff"] > 0.5
        if lg == "ar":
            rev = out / f"{lg}_reversed.png"; burn(t[::-1], rev, a.font); r["rtl_diff"] = round(diff(base, rev), 3); r["rtl_ok"] = r["rtl_diff"] > 0.5
        if a.ocr and shutil.which("tesseract"):
            tl = {"ru": "rus", "en": "eng", "zh": "chi_sim", "ar": "ara", "hi": "hin"}[lg]
            hyp = subprocess.run(["tesseract", str(base), "-", "-l", tl, "--psm", "7"], capture_output=True, text=True).stdout.strip()
            r["ocr"] = hyp; r["cer"] = round(cer(t, hyp), 3)
        r["pass"] = r["coverage_ok"] and r["rendered"] and r.get("shaped", True) and r.get("rtl_ok", True)
        ok_all &= r["pass"]; res["langs"][lg] = r
        print(f"{lg}: {'PASS' if r['pass'] else 'FAIL'} fonts {fonts} missing {missing[:8]} shaped {r.get('shaped', '-')} ({r.get('shaping_diff', '-')}) "
              f"rtl {r.get('rtl_ok', '-')} cer {r.get('cer', '-')}")
    res["pass"] = ok_all; (out / "glyph_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("libass", libass, "harfbuzz", res["harfbuzz"], "fribidi", res["fribidi"]); print("GLYPH_CHECK", "PASS" if ok_all else "FAIL"); sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

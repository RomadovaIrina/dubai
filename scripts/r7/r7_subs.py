#!/usr/bin/env python3
"""Spec block 9: subtitles (ASS + karaoke, 4 presets) + watermark + burn-in. CPU only (ffmpeg/libass).

Timing source (9.1/9.2):
  dub: manifest units[].alignment.placements[].placed (where the English phrase really plays after the fit/atempo)
       + units[].tts.word_spans (CTC forced-align of the TTS, relative to the unit's TTS raw clip).
       A word at t inside phrase clip_span [a,b] is mapped linearly onto placed [p0,p1] (covers atempo).
       If word spans are missing, the phrase is split proportionally by characters.
  src: asr.segments[].words (whisper word timestamps of the source).

  python scripts/r7/r7_subs.py ass  --manifest M --lang dub --style karaoke_yellow --out x.ass
  python scripts/r7/r7_subs.py burn --video in.mp4 --manifest M [--lang dub] [--style S] [--watermark wm.png --wm-corner br --wm-opacity 0.6] --out out.mp4
"""
import argparse, json, os, re, subprocess, sys

THREADS = "4"
# ASS colours are &HAABBGGRR. Primary = sung/highlighted colour for \kf, Secondary = not yet sung.
STYLES = {
    "white":   dict(font="DejaVu Sans", size=0.050, bold=0, primary="&H0000FFFF", secondary="&H00FFFFFF",
                    outline_c="&H00000000", back="&H80000000", border=1, outline=2.0, shadow=1.0),
    "yellow":  dict(font="DejaVu Sans", size=0.050, bold=1, primary="&H00FFFFFF", secondary="&H0000E5FF",
                    outline_c="&H00000000", back="&H80000000", border=1, outline=2.5, shadow=0.0),
    "trendy":  dict(font="DejaVu Sans", size=0.072, bold=1, primary="&H0000FF7F", secondary="&H00FFFFFF",
                    outline_c="&H00000000", back="&H00000000", border=1, outline=4.0, shadow=2.0, upper=True),
    "outline": dict(font="DejaVu Sans", size=0.052, bold=0, primary="&H0000C8FF", secondary="&H00FFFFFF",
                    outline_c="&H00202020", back="&H96000000", border=3, outline=1.0, shadow=0.0),
}
MAX_CHARS, MAX_LINES = 42, 2


def ts(t):
    t = max(0.0, t); cs = int(round(t * 100)); h, cs = divmod(cs, 360000); m, cs = divmod(cs, 6000); s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _norm(w):
    return re.sub(r"[^\w']", "", w.lower())


def dub_words(man):
    """[(word_text, start, end)] on the final timeline."""
    out = []
    for u in man["units"]:
        al = u.get("alignment") or {}; tparts = u.get("tts_parts") or []; spans = (u.get("tts") or {}).get("word_spans") or []
        for pl in al.get("placements", []):
            p0, p1 = pl["placed"]
            for pi in pl.get("phrases", []):
                if pi >= len(tparts): continue
                tp = tparts[pi]; a, b = tp["clip_span"]
                words = tp["text"].split()
                if not words: continue
                sp = [s for s in spans if a - 0.02 <= s["start"] < b]
                if len(pl.get("phrases", [])) > 1:  # several phrases share one placement: split it by clip length
                    tot = sum(tparts[j]["clip_span"][1] - tparts[j]["clip_span"][0] for j in pl["phrases"])
                    off = sum(tparts[j]["clip_span"][1] - tparts[j]["clip_span"][0] for j in pl["phrases"] if j < pi)
                    q0 = p0 + (p1 - p0) * off / tot; q1 = q0 + (p1 - p0) * (b - a) / tot
                else:
                    q0, q1 = p0, p1
                k = (q1 - q0) / max(b - a, 1e-3)
                if len(sp) == len(words):
                    for w, s in zip(words, sp):
                        out.append((w, q0 + (s["start"] - a) * k, q0 + (min(s["end"], b) - a) * k))
                else:  # proportional split by characters
                    tot = sum(len(w) + 1 for w in words); t = q0
                    for w in words:
                        d = (q1 - q0) * (len(w) + 1) / tot; out.append((w, t, t + d)); t += d
    out.sort(key=lambda x: x[1])
    return out


def src_words(man):
    out = []
    for s in man["asr"]["segments"]:
        for w in s.get("words", []):
            out.append((w["word"].strip(), w["start"], w["end"]))
    return [w for w in out if w[0]]


def group_events(words, max_chars=MAX_CHARS, max_gap=0.7, max_dur=6.0):
    """Cue = <= 2 lines x 42 chars; break on long pauses, sentence ends, or too long."""
    cues, cur = [], []
    def flush():
        if cur: cues.append(list(cur)); cur.clear()
    for w in words:
        if cur:
            text = " ".join(x[0] for x in cur + [w])
            if (w[1] - cur[-1][2] > max_gap or len(text) > max_chars * MAX_LINES - 2 or w[2] - cur[0][1] > max_dur
                    or (re.search(r"[.!?]$", cur[-1][0]) and len(" ".join(x[0] for x in cur)) > 20)):
                flush()
        cur.append(w)
    flush()
    return cues


def split_lines(ws, max_chars=MAX_CHARS):
    """Index after which to put \\N (balanced 2 lines), or None for 1 line."""
    text = " ".join(w[0] for w in ws)
    if len(text) <= max_chars: return None
    best, bi, acc = 1e9, None, 0
    for i, w in enumerate(ws[:-1]):
        acc += len(w[0]) + (1 if i else 0)
        cost = abs(acc - (len(text) - acc - 1)) - (6 if re.search(r"[,;:]$", w[0]) else 0)
        if cost < best: best, bi = cost, i
    return bi


def make_ass(words, w, h, style, karaoke=True):
    st = STYLES[style]
    fs = max(14, int(round(min(w, h) * st["size"] * (0.9 if h > w else 1.0))))
    # chars per line: <= 42, and what fits the usable width (avg glyph ~0.6 em, bold/caps ~0.72 em)
    em = 0.72 if (st["bold"] or st.get("upper")) else 0.6
    max_chars = max(12, min(MAX_CHARS, int(w * 0.9 / (fs * em))))
    mv = int(h * (0.12 if h > w else 0.07)); ml = int(w * 0.05)
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{st['font']},{fs},{st['primary'] if karaoke else st['secondary']},{st['secondary']},{st['outline_c']},{st['back']},{-1 if st['bold'] else 0},0,0,0,100,100,0,0,{st['border']},{st['outline']},{st['shadow']},2,{ml},{ml},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    cues = group_events(words, max_chars)
    for ci, cue in enumerate(cues):
        nxt = cues[ci + 1][0][1] if ci + 1 < len(cues) else 1e9  # no overlap: never 2 cues on screen
        brk = split_lines(cue, max_chars); t0 = cue[0][1]; t1 = min(cue[-1][2] + 0.25, nxt - 0.01); parts = []; prev = t0
        for i, (wt, a, b) in enumerate(cue):
            if st.get("upper"): wt = wt.upper()
            wt = wt.replace("{", "(").replace("}", ")")
            if karaoke:
                gap = int(round((a - prev) * 100))
                if gap > 0: parts.append(f"{{\\k{gap}}}")
                parts.append(f"{{\\kf{max(1, int(round((b - a) * 100)))}}}{wt}"); prev = b
            else:
                parts.append(wt)
            parts.append("\\N" if i == brk else " ")
        lines.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},Sub,,0,0,0,,{''.join(parts).rstrip()}")
    return hdr + "\n".join(lines) + "\n"


def probe(video):
    r = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height,r_frame_rate", "-of", "json", video]))["streams"][0]
    return r["width"], r["height"], r["r_frame_rate"]


def _esc(p):
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn(video, out, ass=None, wm=None, corner="br", opacity=0.6, crf=18, preview_t=None):
    w, h, fps = probe(video)
    chain = "[0:v]" + (f"ass='{_esc(ass)}'" if ass else "null") + "[s]"
    inputs = ["-i", video]; last = "[s]"
    if wm:
        inputs += ["-i", wm]; ww = max(16, int(min(w, h) * 0.18)); m = int(min(w, h) * 0.03)
        pos = {"tl": (f"{m}", f"{m}"), "tr": (f"W-w-{m}", f"{m}"), "bl": (f"{m}", f"H-h-{m}"), "br": (f"W-w-{m}", f"H-h-{m}")}[corner]
        chain += (f";[1:v]format=rgba,scale={ww}:-1,colorchannelmixer=aa={opacity}[wm];"
                  f"[s][wm]overlay={pos[0]}:{pos[1]}:format=auto[o]"); last = "[o]"
    chain += f";{last}format=yuv420p[v]"
    if preview_t is not None:
        # -ss before -i resets timestamps; ass needs absolute time -> seek after decode instead
        cmd = ["ffmpeg", "-y", "-v", "error", "-threads", THREADS, *inputs, "-filter_complex", chain + f";[v]trim=start={preview_t},setpts=PTS-STARTPTS[p]",
               "-map", "[p]", "-frames:v", "1", out]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-threads", THREADS, *inputs, "-filter_complex", chain, "-filter_complex_threads", THREADS,
               "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-x264-params", f"threads={THREADS}",
               "-r", fps, "-c:a", "copy", "-movflags", "+faststart", out]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["ass", "burn", "preview"])
    ap.add_argument("--manifest"); ap.add_argument("--video"); ap.add_argument("--out", required=True)
    ap.add_argument("--lang", choices=["src", "dub"], default="dub"); ap.add_argument("--style", choices=list(STYLES), default="yellow")
    ap.add_argument("--karaoke", choices=["on", "off"], default="on")
    ap.add_argument("--watermark"); ap.add_argument("--wm-corner", choices=["tl", "tr", "bl", "br"], default="br")
    ap.add_argument("--wm-opacity", type=float, default=0.6); ap.add_argument("--t", type=float, default=15.0)
    ap.add_argument("--no-subs", action="store_true")
    a = ap.parse_args()
    ass = None
    if not a.no_subs:
        man = json.load(open(a.manifest)); words = dub_words(man) if a.lang == "dub" else src_words(man)
        if a.video: w, h, _ = probe(a.video)
        else: w, h = man["source"]["video"]["width"], man["source"]["video"]["height"]
        ass = a.out + ".ass" if a.cmd != "ass" else a.out
        open(ass, "w").write(make_ass(words, w, h, a.style, a.karaoke == "on"))
        print(f"ass {ass}: {len(words)} words", file=sys.stderr)
    if a.cmd == "burn": burn(a.video, a.out, ass, a.watermark, a.wm_corner, a.wm_opacity)
    elif a.cmd == "preview": burn(a.video, a.out, ass, a.watermark, a.wm_corner, a.wm_opacity, preview_t=a.t)


if __name__ == "__main__":
    main()

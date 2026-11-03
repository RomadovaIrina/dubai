#!/usr/bin/env python3
"""r13 spec 6.5 / 7.4 / 7.5: the external path for languages the local models handle badly.

  7.5  OpenAITTSDonor   OpenAI TTS (the ONLY allowed donor engine) over plain HTTPS (POST /v1/audio/speech, no SDK dependency):
                        a process-wide token-bucket rate limiter (requests / min) + exponential backoff with full jitter on 429 / 5xx /
                        network errors, honouring Retry-After; after max_retries it raises DonorUnavailable, which the caller turns into a
                        fallback + a flag: a 429 never crashes the worker and never loses the job.
  7.4  VoiceConverter   ChatterboxVC (Chatterbox S3Gen, the same s3gen.pt the TTS uses, MIT): the donor track's speech tokens re-voiced
                        with the user's timbre (the speaker reference the pipeline already cuts, 7.1).
  6.5  Router           per target language: MT engine (qwen | openai) and TTS path (chatterbox | donor_vc), from lang_routes.json
                        (the table proposed from the local measurements, reports/experiments/r13-spec/LANGS.md).
  mock                  MockOpenAI: a local HTTP server with the same endpoint; 429 / 500 on a schedule; the audio it returns is a stand-in
                        voice (Chatterbox multilingual with its built-in voice, NOT the user's). Used until the key arrives:
                        DABAI_OPENAI_BASE=http://127.0.0.1:<port>/v1, OPENAI_API_KEY=mock.
The key is read from the environment (OPENAI_API_KEY) and never logged."""
from __future__ import annotations
import io, json, os, pathlib, random, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[2]


class DonorUnavailable(RuntimeError):
    """the donor could not deliver after every retry (rate limit / outage): the caller falls back, the job goes on"""


class RateLimiter:
    """token bucket shared by all threads of the process: at most `rpm` requests per minute, bursts up to `burst`"""
    def __init__(self, rpm: float, burst: int = 1):
        self.rate = rpm / 60.0; self.cap = float(max(1, burst)); self.tokens = self.cap; self.t = time.monotonic(); self.lock = threading.Lock()

    def acquire(self) -> float:
        waited = 0.0
        while True:
            with self.lock:
                now = time.monotonic(); self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate); self.t = now
                if self.tokens >= 1: self.tokens -= 1; return waited
                need = (1 - self.tokens) / self.rate
            time.sleep(need); waited += need

    def penalise(self, seconds: float) -> None:
        """a 429 says the server-side budget is spent: drain the bucket so the other threads back off too"""
        with self.lock: self.tokens = min(self.tokens, -seconds * self.rate)


_LIMITERS: dict = {}


class OpenAITTSDonor:
    RETRY = {429, 500, 502, 503, 504}

    def __init__(self, model: str = "gpt-4o-mini-tts", voice: str = "alloy", rpm: float | None = None, max_retries: int = 6,
                 base_delay: float = 1.0, max_delay: float = 60.0, timeout: float = 60.0, base_url: str | None = None, seed: int | None = None):
        self.base = (base_url or os.environ.get("DABAI_OPENAI_BASE") or "https://api.openai.com/v1").rstrip("/")
        self.model, self.voice, self.max_retries, self.base_delay, self.max_delay, self.timeout = model, voice, max_retries, base_delay, max_delay, timeout
        rpm = rpm or float(os.environ.get("DABAI_OPENAI_RPM", "50"))
        self.limiter = _LIMITERS.setdefault((self.base, rpm), RateLimiter(rpm, burst=max(1, int(rpm // 20))))
        self.rng = random.Random(seed); self.stats = {"requests": 0, "retries": 0, "429": 0, "5xx": 0, "net": 0, "limiter_wait_s": 0.0, "backoff_s": 0.0}

    def _key(self) -> str:
        k = os.environ.get("OPENAI_API_KEY", "")
        if not k: raise DonorUnavailable("OPENAI_API_KEY is not set")
        return k

    def _request(self, path: str, body: dict):
        """POST <base>/<path> with the limiter, exponential backoff + full jitter, Retry-After; returns the 200 response or raises DonorUnavailable"""
        import requests
        last = ""
        for attempt in range(self.max_retries + 1):
            self.stats["limiter_wait_s"] += self.limiter.acquire(); self.stats["requests"] += 1; retry_after = None
            try:
                r = requests.post(f"{self.base}/{path}", json=body, timeout=self.timeout, headers={"Authorization": f"Bearer {self._key()}"})
                if r.status_code == 200 and r.content: return r
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in self.RETRY: raise DonorUnavailable(f"donor refused the request ({last})")   # 400 / 401 / 403: retrying will not help
                self.stats["429" if r.status_code == 429 else "5xx"] += 1
                ra = r.headers.get("Retry-After") or r.headers.get("retry-after")
                if ra:
                    try: retry_after = float(ra)
                    except ValueError: retry_after = None
                if r.status_code == 429: self.limiter.penalise(retry_after or self.base_delay)
            except requests.RequestException as e:
                self.stats["net"] += 1; last = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt == self.max_retries: break
            d = min(self.max_delay, self.base_delay * 2 ** attempt); d = self.rng.uniform(0, d)   # exponential backoff, full jitter
            if retry_after: d = max(d, retry_after)
            self.stats["retries"] += 1; self.stats["backoff_s"] += d; time.sleep(d)
        raise DonorUnavailable(f"donor unavailable after {self.max_retries + 1} attempts: {last}")

    def synth(self, text: str, out_wav: pathlib.Path, instructions: str | None = None) -> pathlib.Path:
        body = {"model": self.model, "voice": self.voice, "input": text, "response_format": "wav"}
        if instructions: body["instructions"] = instructions
        out_wav = pathlib.Path(out_wav); out_wav.write_bytes(self._request("audio/speech", body).content); self.stats["tts_chars"] = self.stats.get("tts_chars", 0) + len(text)
        return out_wav

    def translate(self, text: str, src_name: str, tgt_name: str, hint: str = "") -> str:
        """6.5: the external translation (chat completions, DABAI_OPENAI_MT_MODEL, default gpt-4o-mini), the production prompt; same limiter / retries"""
        body = {"model": os.environ.get("DABAI_OPENAI_MT_MODEL", "gpt-4o-mini"), "temperature": 0,
                "messages": [{"role": "system", "content": "You are a professional translator. Return only the translation, with no explanation."},
                             {"role": "user", "content": f"Translate from {src_name} to {tgt_name}{' for dubbing. ' + hint if hint else ''}:\n{text}"}]}
        j = self._request("chat/completions", body).json(); u = j.get("usage") or {}
        self.stats["mt_tokens_in"] = self.stats.get("mt_tokens_in", 0) + int(u.get("prompt_tokens", 0)); self.stats["mt_tokens_out"] = self.stats.get("mt_tokens_out", 0) + int(u.get("completion_tokens", 0))
        return " ".join(j["choices"][0]["message"]["content"].split()).strip().strip("\"«»“”")


class VoiceConverter:
    """ChatterboxVC built from the local s3gen.pt (the TTS checkpoint; no s3gen.safetensors download)"""
    def __init__(self, device: str = "cuda"):
        import torch
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.vc import ChatterboxVC
        s3 = S3Gen(); s3.load_state_dict(torch.load(ROOT / "models" / "chatterbox" / "s3gen.pt", map_location="cpu"), strict=False); s3.to(device).eval()
        self.vc = ChatterboxVC(s3, device); self.sr = self.vc.sr

    def convert(self, donor_wav: pathlib.Path, user_ref_wav: pathlib.Path, out_wav: pathlib.Path) -> pathlib.Path:
        import soundfile as sf
        w = self.vc.generate(str(donor_wav), target_voice_path=str(user_ref_wav)).squeeze(0).numpy(); sf.write(out_wav, w, self.sr); return pathlib.Path(out_wav)


class Router:
    """6.5: per target language -> {"mt": qwen|openai, "tts": chatterbox|donor_vc}; unknown language -> external for both (safe default)"""
    def __init__(self, table: str | pathlib.Path | None = None):
        p = pathlib.Path(table or os.environ.get("DABAI_LANG_ROUTES") or pathlib.Path(__file__).with_name("lang_routes.json"))
        self.table = json.loads(p.read_text()) if p.exists() else {}

    def route(self, lang: str, key_available: bool | None = None) -> dict:
        r = dict(self.table.get("langs", {}).get(lang) or {"mt": "openai", "tts": "donor_vc", "note": "not measured"})
        key = bool(os.environ.get("OPENAI_API_KEY")) if key_available is None else key_available
        if not key and (r["mt"] == "openai" or r["tts"] == "donor_vc"):   # no key yet: the local path, flagged (the user is told the quality may be lower)
            r.update(mt_planned=r["mt"], tts_planned=r["tts"], mt="qwen", tts="chatterbox", degraded="no_external_key")
        return r


def synth_with_fallback(text: str, lang: str, out_wav: pathlib.Path, user_ref: pathlib.Path, donor: OpenAITTSDonor, vc: VoiceConverter, local_tts) -> dict:
    """7.4 / 7.5 for one phrase: donor TTS -> VC to the user's voice; DonorUnavailable -> local Chatterbox (local_tts(text, lang, out_wav)) + flag"""
    rec = {"path": None, "engine": None, "flag": None}
    try:
        d = donor.synth(text, pathlib.Path(str(out_wav) + ".donor.wav")); vc.convert(d, user_ref, out_wav); rec.update(path=str(out_wav), engine="openai_tts+chatterbox_vc")
    except DonorUnavailable as e:
        local_tts(text, lang, out_wav); rec.update(path=str(out_wav), engine="chatterbox_fallback", flag="donor_unavailable_fallback", error=str(e)[:300])
    return rec


# ------------------------------------------------------------------------------------------------------------------------------ mock
def _wav_bytes(y, sr) -> bytes:
    import numpy as np
    b = io.BytesIO(); w = wave.open(b, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()); w.close(); return b.getvalue()


class MockOpenAI:
    """local stand-in for POST /v1/audio/speech. schedule: list of status codes for the first requests (e.g. [429, 429, 500]),
    then `p429` random 429s; audio = stand_in(text) -> (float32 numpy, sr) (default: 1 s of silence)."""
    def __init__(self, schedule=(), p429: float = 0.0, retry_after: float | None = 1.0, stand_in=None, seed: int = 0, rpm_cap: float | None = None, stand_in_text=None):
        self.schedule = list(schedule); self.p429 = p429; self.stand_in_text = stand_in_text; self.retry_after = retry_after; self.stand_in = stand_in; self.rng = random.Random(seed)
        self.log = []; self.lock = threading.Lock(); self.rpm_cap = rpm_cap; self.times = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with outer.lock:
                    now = time.monotonic(); outer.times.append(now)
                    over = outer.rpm_cap and sum(1 for t in outer.times if now - t < 60) > outer.rpm_cap
                    code = outer.schedule.pop(0) if outer.schedule else (429 if over or outer.rng.random() < outer.p429 else 200)
                    outer.log.append({"t": round(now, 3), "code": code, "auth": self.headers.get("Authorization", "").startswith("Bearer "), "chars": len(body.get("input", ""))})
                if code == 200 and self.path == "/v1/chat/completions":
                    msg = body["messages"][-1]["content"]; txt = outer.stand_in_text(msg) if outer.stand_in_text else "[mock] " + msg.split("\n", 1)[-1]
                    data = json.dumps({"choices": [{"message": {"content": txt}}], "usage": {"prompt_tokens": len(msg) // 4, "completion_tokens": len(txt) // 4}}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
                if self.path not in ("/v1/audio/speech", "/v1/chat/completions"): code = 404
                if code != 200:
                    self.send_response(code)
                    if code == 429 and outer.retry_after: self.send_header("Retry-After", str(outer.retry_after))
                    self.end_headers(); self.wfile.write(json.dumps({"error": {"code": code}}).encode()); return
                import numpy as np
                y, sr = outer.stand_in(body["input"]) if outer.stand_in else (np.zeros(24000, np.float32), 24000)
                data = _wav_bytes(y, sr); self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H); self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def base(self) -> str: return f"http://127.0.0.1:{self.port}/v1"

    def close(self): self.srv.shutdown()

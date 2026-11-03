# OpenAI readiness (6.5 / 7.4 / 7.5): what it takes to switch the mock to the real service

*r13, 03.11. One page for the decision; the user deferred the key on 03.11.*

## Who needs it at all
From the local measurements (LANGS.md), **13 of 22 languages need nothing external**: en, de, es, fr, it, nl, zh, ja, pt, sv, no, ms, hi. The external path is used only for:

| Group | Languages | External part |
|---|---|---|
| translation only | sw | OpenAI chat translation |
| voice only | da | OpenAI TTS donor + local ChatterboxVC (the user's timbre) |
| both | he, ar, ko, el, fi | translation + donor voice |
| odd sentence | tr, pl (any language) | a sentence still mixed after the local guard → that sentence goes to the external translation (`polyglot.py`); without a key it is kept and the job is flagged `mixed_language_sentence_kept` |

## What is ready (code, tested against the mock)
- `scripts/r13/donor.py`:
  - **`OpenAITTSDonor.synth()`**: `POST /v1/audio/speech`, model `gpt-4o-mini-tts`, voice `alloy`, WAV.
  - **`.translate()`**: `POST /v1/chat/completions`, model `gpt-4o-mini` by default, the production prompt.
  - Both share one process-wide rate limiter and retry with exponential backoff (full jitter, Retry-After honoured). After the retries they raise `DonorUnavailable`, and the job falls back to the local path with a flag. **A 429 never crashes the worker or loses the job** (mock tests: transient 429/500, persistent 429, 16 parallel requests at the cap, 400, no key).
- **`VoiceConverter`**: ChatterboxVC from the local s3gen.pt, so the user's voice never leaves our GPU. Mock test: similarity to the user 0.55–0.63 → 0.78–0.87.
- **`Router` + `scripts/r13/lang_routes.json`**: per language `mt: qwen|openai`, `tts: chatterbox|donor_vc`. Without a key it routes to the local path and marks `degraded: no_external_key`.
- **`scripts/r13/polyglot.py`** (every non-English dub) follows the router per language. Flags: `external_mt_unavailable_fallback`, `donor_unavailable_fallback`. The user message is in `dabai_job.py`.

## To switch on (no code change)
```bash
# /workspace/dub/.env (mode 600), next to HF_TOKEN:
OPENAI_API_KEY=sk-...                 # an organisation key, restricted to the Audio + Chat endpoints
DABAI_OPENAI_RPM=50                   # the account's requests-per-minute for the TTS model (Settings -> Limits), shared by all jobs of the process
DABAI_OPENAI_MT_MODEL=gpt-4o-mini     # optional; the translation model
# DABAI_OPENAI_BASE stays unset (https://api.openai.com/v1); it points at the mock only in tests
```
Then one real check:
- `python scripts/r13/donor_test.py gpu --ref <a user reference wav> --langs he,ar,ko,el,fi`, after switching its server from the mock to the real one: a 5-line change in the test;
- one Hebrew E2E: `DUB_POLYGLOT='{"mode":"all","lang":"he"}'`.

## Cost estimate (to confirm on the pricing page before signing)
- **TTS `gpt-4o-mini-tts`:** $0.60 / 1M text input tokens + $12 / 1M audio output tokens ≈ **$0.015 per minute of generated speech** (third-party estimates from the token prices). `tts-1`: $15 / 1M characters ≈ $0.0135 per minute (~900 characters / min).
  - Our dub speech is ~60–75 % of the video, one request per phrase, plus a rare shorter retry.
  - So it's **≈ $0.010–0.015 per video-minute** for the "voice" languages.
- **Translation `gpt-4o-mini`:** ~150 words / min of speech ≈ 400 input + 250 output tokens per video-minute ≈ **$0.0002**. Negligible.
- **For comparison:** our GPU cost per video-minute is in WALLCLOCK.md (at $0.72 / h). The external part adds ~10–15 % only for the 7 external languages, and 0 for the other 15.

## Rate limits
- They depend on the account's usage tier and grow with spend. The real numbers are in the account's Limits page.
- **New accounts:** TTS limits can be very low (single-digit RPM have been reported at the entry tier).
- **Our load:** one video-minute ≈ 10–15 phrases = 10–15 TTS requests. 10 parallel 1-minute jobs ≈ 150 requests within a few minutes. **Plan a tier with ≥ 150–200 RPM on the TTS model before launch**, or keep `DABAI_OPENAI_RPM` at the real limit: then jobs wait instead of failing.

## Before production (not code)
- **Privacy (12.5):** the translated TEXT goes to OpenAI (translation and TTS input); the user's AUDIO and VOICE do not (VC is local). This must be in the privacy policy / data-processing terms. Consider OpenAI's zero-data-retention option for the organisation.
- **Monitoring (block 10):** `status.json` and the manifest carry the external call counts (`polyglot.external_stats`: requests, retries, 429s, characters, tokens). Infra can sum them per day for the bill.

Sources (prices / limits, 03.11): [OpenAI gpt-4o-mini-tts model page](https://developers.openai.com/api/docs/models/gpt-4o-mini-tts), [OpenAI rate-limits guide](https://developers.openai.com/api/docs/guides/rate-limits/index.html), [community thread on TTS pricing](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616), [TTS cost estimate](https://www.s-anand.net/blog/openai-tts-cost/), [tts-1 RPM by tier thread](https://community.openai.com/t/tts-1-tts-1-hd-api-rpm-and-rpd-based-on-chosen-tier/783207).

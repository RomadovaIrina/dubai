# r8: LTX-2.3 LipDub (DubIt) licence check (2026-10-29)

Checked against primary sources: HF model cards and API, plus the LICENSE files in github.com/Lightricks/LTX-2 at commit e98f6d9 (2026-10-18). This is not legal advice.
**Renamed:** `LTX-2.3-22b-IC-LoRA-LipDub` now redirects to **`Lightricks/LTX-2.3-22b-IC-LoRA-DubIt`**, file `ltx-2.3-22b-ic-lora-dubit-0.9.safetensors`. The pipeline is `ltx_pipelines/dubit.py`.

| # | Component (what inference needs) | Source / file / size | Licence (primary) | Commercial? | Flag |
|---|---|---|---|---|---|
| 1 | DubIt IC-LoRA | HF Lightricks/LTX-2.3-22b-IC-LoRA-DubIt, 2,466,665,072 B, **gated (auto)** | card: "ltx-2-community-license" | yes, < $10M revenue | GATED: HF account must accept. Training data is only "a lip-dubbing dataset" (undisclosed). |
| 2 | Base LTX-2.3 22B distilled, fp8 (bundles transformer + video VAE + audio VAE + text projection) | HF Lightricks/LTX-2.3-fp8 `ltx-2.3-22b-distilled-fp8.safetensors`, 29,531,884,062 B, not gated | `license_link: .../LICENSE-2` (LTX-2 Community Licence, 2026-01-05; covers LTX-2.3) | yes, < $10M; ≥ $10M needs a paid licence | see 6–8 |
| 3 | Video VAE / audio VAE (+ vocoder) | inside #2 (single-file checkpoint) | as #2 | as #2 | — |
| 4 | Spatial upscaler x2 (stage 2 of DubIt) | HF Lightricks/LTX-2.3 `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, 995,743,560 B | as #2 | as #2 | — |
| 5 | Text encoder Gemma 3 12B (QAT q4_0 unquantised, bf16) | HF google/gemma-3-12b-it-qat-q4_0-unquantized, ~24.4 GB, **gated (manual click)** | Gemma Terms of Use (last modified 2026-04-01) + Gemma Prohibited Use Policy | yes: no NC clause, no revenue cap | GATED. Pass-through: use restrictions + Notice file if we redistribute. Google may restrict remotely. Outputs not claimed. |
| 6 | LTX-2 Licence: grant + revenue | LICENSE-2 §2 | "for any purpose", but Entities with revenue ≥ **$10,000,000** need a paid Commercial Use Agreement (liquidated damages otherwise) | conditional | **WATCH:** revenue is aggregated across affiliates (≥ 50 % control). |
| 7 | LTX-2 Attachment A (use restrictions; also bind Outputs) | LICENSE-2 Att. A | #5: content must be **disclosed as machine-generated**. #7: no impersonation/deepfakes **without consent**. #20: no use in a product that **"directly competes with Licensor's commercial products"** without a separate licence | conditional | **FLAG #20:** Lightricks sells LTX Studio/API, so a dubbing *product* could be read as competing; ask legal. #5/#7: need an "AI-dubbed" label + speaker consent. |
| 8 | Inference code (ltx-core, ltx-pipelines) | github Lightricks/LTX-2 | **no OSS code licence:** the repo LICENSE is the same Community Licence ("inference-enabling code" is part of "LTX-2"); the packages carry no other licence field | as #2 | **Not a "clean" OSS licence** for the code; same terms as the weights. Third-party deps: torch (BSD), transformers (Apache-2.0), av (BSD). Optional videoseal/audioseal are not installed. |
| 9 | Not needed / not downloaded | dev checkpoints, distilled LoRA, temporal upscaler, LTX-2.5 (Gemma 4) | — | — | DubIt 2.3 does not use them. LTX-2.5 weights are not compatible with the 2.3 LoRA. |
| 10 | **Verdict** | — | Nothing is strictly non-commercial. Usable for a < $10M company with: AI-generated disclosure, speaker consent, and a legal read of Att. A #20 (competing product) | **grey-green** | Blocker for the probe: two HF gates (#1, #5) must be accepted by the user on the RomadovaIrina account. |

Planned download on r8-a: 29.53 + 2.47 + 1.00 + ~24.4 GB ≈ **57.4 GB** (bf16 distilled 46 GB would break the 60 GB cap, so we use the fp8 checkpoint).

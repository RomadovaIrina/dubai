"""pyannote speaker-diarization-3.1 on the demo speech file (needs HF_TOKEN + accepted model licenses)."""
from smoke_common import *
import torch, pyannote.audio
print("pyannote.audio", pyannote.audio.__version__)
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not tok:
    print("SKIP pyannote: HF_TOKEN not set (put HF_TOKEN=hf_... into /workspace/.env); import OK"); sys.exit(0)
from pyannote.audio import Pipeline
with Timer() as tl:
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=tok)
if pipe is None: fail("pipeline is None: accept licenses for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0")
pipe.to(torch.device("cuda"))
with Timer() as t:
    dia = pipe(str(DEMO_WAV))
turns = list(dia.itertracks(yield_label=True))
print(f"loaded {tl.s:.1f}s, diarized {t.s:.1f}s, {len(turns)} turns, speakers={sorted(dia.labels())}")
for seg, _, spk in turns[:5]: print(f"  {seg.start:6.2f}-{seg.end:6.2f} {spk}")
if not turns: fail("no speaker turns")
print("PASS pyannote", gpu_mem())

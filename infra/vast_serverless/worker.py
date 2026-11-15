"""DabAI on Vast Serverless (TZ 10.1, Irina 11.11: Vast, not RunPod): the Vast PyWorker in front of the worker image's job API.

The image runs the API (+ the resident worker, DABAI_RESIDENT=1) on 127.0.0.1:8000; this PyWorker exposes ONE route, POST /jobs/sync = one whole
dubbing job per request (job_api.submit_sync: blocks until the job ends, uploads the video to the request's presigned output_put_url, returns
the final status). One job at a time per worker (allow_parallel_requests=False): the 32 GB card fits one pipeline. The autoscaler's load unit =
input video minutes (the client sends "video_minutes"; a missing value counts as 1).
    PYWORKER_REPO=<a public repo with this file + requirements.txt at its root>   (the Serverless template runs `python worker.py`)
Log readiness: /data/resident/start.log gets "RESIDENT_READY" once the KeySync models are loaded + parked and the audio worker is up."""
import math, os
from dataclasses import dataclass, field
from typing import Any, Dict
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig
from vastai.serverless.server.lib.data_types import ApiPayload, JsonDataException   # vastai==1.11.0 (requirements.txt)

BENCH_CLIP = os.environ.get("BENCH_CLIP", "/data/bench_5s.mp4")   # start.sh: a 5 s synthetic clip (no face: a fast round trip)
ROUTING_ONLY = ("video_minutes",)   # read here for the autoscaler, never sent on: the job API rejects unknown fields (422)


def bench_payload() -> dict:
    import uuid
    return {"job_id": "bench_" + uuid.uuid4().hex[:8], "input": BENCH_CLIP, "target_lang": "en", "video_minutes": 5 / 60}


@dataclass
class DabaiPayload(ApiPayload):
    """r20: the generic payload forwards the request body as is, so "video_minutes" reached POST /jobs/sync -> 422 "unknown field:
    video_minutes" -> every benchmark failed -> Vast destroyed each worker ("No successful responses from benchmark")."""
    input: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_test(cls) -> "DabaiPayload":
        return cls(input=bench_payload())

    @classmethod
    def from_dict(cls, input: Dict[str, Any]) -> "DabaiPayload":
        return cls(input=input)

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "DabaiPayload":
        if not isinstance(json_msg, dict): raise JsonDataException({"data": "payload must be a dictionary"})
        return cls(input=json_msg)

    def generate_payload_json(self) -> Dict[str, Any]:
        return {k: v for k, v in self.input.items() if k not in ROUTING_ONLY}

    def count_workload(self) -> float:
        try: w = float(self.input.get("video_minutes") or 1.0)
        except (TypeError, ValueError): return 1.0
        return w if math.isfinite(w) and w > 0 else 1.0


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8000,
    model_log_file="/data/resident/start.log",
    handlers=[
        HandlerConfig(
            route="/jobs/sync",
            allow_parallel_requests=False,           # FIFO, one job in flight (the API queue is the GPU lock of the node)
            max_queue_time=3 * 3600.0,               # a queued job waits, it is not refused (a 5-min video takes ~15-20 min of wall)
            payload_class=DabaiPayload,              # strips video_minutes before the job API; counts it as the workload
            benchmark_config=BenchmarkConfig(generator=bench_payload, runs=1, concurrency=1),   # one real 10 s job: also warms the TRT / compile caches
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["RESIDENT_READY"],
        on_error=["RESIDENT: "],                     # resident.sh prints "RESIDENT: <half> failed" on a failed start
        on_info=["KS_SERVE start", "audio worker up"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()

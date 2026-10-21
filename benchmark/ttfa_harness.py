"""
TTFA (Time To First Audio) harness for speech-to-speech systems.

    pip install silero-vad soundfile numpy torch

Input files: 16 kHz, mono WAV containing one user utterance.

TTFA is measured from the *reference* end of speech (offline VAD on the input file)
to the first audio the system sends back. The file is streamed in real time, followed
by silence, exactly like a live microphone.

Usage:
    python ttfa_harness.py --adapter example_adapters:FakeS2SAdapter -n 3 a.wav b.wav
"""
import argparse
import asyncio
import importlib
import json
import time
from functools import lru_cache

import numpy as np
import soundfile as sf

SR = 16000
CHUNK_MS = 20
CHUNK = SR * CHUNK_MS // 1000   # samples per chunk sent to the system
TIMEOUT_S = 15.0                # give up if nothing audible arrives this long after end of speech
SILENCE_RMS = 500               # int16 RMS above this counts as "audible"


class S2SAdapter:
    """Implement one of these per system (API or local). The harness only sees this interface."""

    async def connect(self): ...

    async def send_audio(self, pcm_bytes: bytes): ...   # 16 kHz mono int16 PCM, CHUNK samples

    async def receive_audio(self):                       # async generator yielding int16 PCM bytes
        raise NotImplementedError
        yield

    async def close(self): ...


@lru_cache(maxsize=1)
def _vad():
    # Loaded lazily so importing this module (e.g. from an adapter) stays cheap.
    from silero_vad import load_silero_vad
    return load_silero_vad()


def reference_eos_seconds(pcm_int16: np.ndarray) -> float:
    """Where the user actually stops talking in the file, in seconds."""
    import torch
    from silero_vad import get_speech_timestamps

    audio = torch.from_numpy(pcm_int16.astype(np.float32) / 32768)
    # speech_pad_ms=0: the default 30 ms padding would push the reference EOS late.
    segments = get_speech_timestamps(audio, _vad(), sampling_rate=SR, speech_pad_ms=0)
    if not segments:
        raise ValueError("no speech detected in input file")
    return segments[-1]["end"] / SR


def load_input(path: str) -> np.ndarray:
    pcm, sr = sf.read(path, dtype="int16")
    if sr != SR or pcm.ndim != 1:
        raise ValueError(f"{path}: expected {SR} Hz mono audio, got {sr} Hz with shape {pcm.shape}")
    return pcm


def is_audible(chunk: bytes) -> bool:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return samples.size > 0 and np.sqrt(np.mean(samples ** 2)) > SILENCE_RMS


async def run_once(adapter: S2SAdapter, path: str, timeout_s: float = TIMEOUT_S) -> dict:
    pcm = load_input(path)
    eos_offset = reference_eos_seconds(pcm)

    first_byte = None
    first_audible = None

    async def sender(t0: float):
        # Stream the file in real time, then keep sending silence like a live mic would.
        silence = np.zeros(CHUNK, dtype=np.int16)
        n = 0
        while True:
            start = n * CHUNK
            chunk = pcm[start:start + CHUNK] if start < len(pcm) else silence
            await adapter.send_audio(chunk.tobytes())
            n += 1
            # Pace against the absolute clock so sleep errors don't accumulate.
            await asyncio.sleep(max(0.0, t0 + n * CHUNK / SR - time.perf_counter()))

    async def receiver():
        nonlocal first_byte, first_audible
        async for chunk in adapter.receive_audio():
            now = time.perf_counter()
            if first_byte is None:
                first_byte = now
            if is_audible(chunk):
                first_audible = now
                return

    await adapter.connect()
    t0 = time.perf_counter()   # connection setup is excluded from the measurement
    send_task = asyncio.create_task(sender(t0))
    recv_task = asyncio.create_task(receiver())
    try:
        # The deadline counts from the reference end of speech, not from the start of
        # streaming, so long utterances don't eat into the reply budget.
        done, _ = await asyncio.wait(
            {send_task, recv_task},
            timeout=eos_offset + timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if send_task in done:
            send_task.result()   # the sender only stops on error: surface it instead of waiting
        if recv_task in done:
            recv_task.result()
    finally:
        for task in (send_task, recv_task):
            task.cancel()
        await asyncio.gather(send_task, recv_task, return_exceptions=True)
        await adapter.close()

    t_eos = t0 + eos_offset

    def ms(t):
        return None if t is None else (t - t_eos) * 1000

    if first_audible is not None:
        status = "ok"
    elif recv_task.done() and not recv_task.cancelled():
        status = "no_audible"   # reply stream ended without anything above SILENCE_RMS
    else:
        status = "timeout"

    return {
        "file": path,
        "status": status,
        "eos_s": eos_offset,
        "ttfa_first_byte_ms": ms(first_byte),
        "ttfa_audible_ms": ms(first_audible),
    }


async def benchmark(make_adapter, files: list[str], repetitions: int = 5, timeout_s: float = TIMEOUT_S) -> list[dict]:
    """Runs sequentially (concurrency = 1). make_adapter() returns a fresh adapter per run."""
    results = []
    for _ in range(repetitions):
        for path in files:
            result = await run_once(make_adapter(), path, timeout_s)
            print(json.dumps(result))
            results.append(result)
    return results


def summarize(results: list[dict], key: str = "ttfa_audible_ms") -> dict:
    """Percentiles cover replies that started at or after the end of speech; replies that
    started earlier are counted in early_starts instead of dragging the percentiles down."""
    vals = np.array([r[key] for r in results if r[key] is not None])
    on_time = vals[vals >= 0]

    def pct(q):
        return float(np.percentile(on_time, q)) if on_time.size else None

    return {
        "metric": key,
        "runs": len(results),
        "measured": int(on_time.size),
        "early_starts": int((vals < 0).sum()),   # system replied before the user finished
        "missing": len(results) - int(vals.size),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
    }


def load_factory(spec: str):
    """'module:attr' -> callable returning a fresh adapter (a class or a factory function)."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--adapter must look like module:ClassName, got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="16 kHz mono WAV files, one utterance each")
    parser.add_argument("--adapter", required=True, help="adapter factory as module:ClassName")
    parser.add_argument("-n", "--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S,
                        help=f"seconds after end of speech to wait for audio (default {TIMEOUT_S})")
    parser.add_argument("--out", help="write per-run results and summaries to this JSON file")
    args = parser.parse_args()

    results = asyncio.run(benchmark(load_factory(args.adapter), args.files, args.repetitions, args.timeout))
    summaries = [summarize(results, key) for key in ("ttfa_first_byte_ms", "ttfa_audible_ms")]
    for s in summaries:
        print(json.dumps(s, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "summaries": summaries}, f, indent=2)


if __name__ == "__main__":
    main()

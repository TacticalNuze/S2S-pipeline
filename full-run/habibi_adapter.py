"""
Cascaded speech-to-speech on CPU: silero VAD -> moulsot ASR -> LLM -> Habibi TTS.

The LLM is any OpenAI-compatible endpoint, configured like pipecat-quickstart's bot.py
(environment or a .env file in this folder):

    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...      # optional, for non-OpenAI providers
    OPENAI_MODEL=gpt-4.1     # optional
    HABIBI_DIALECT=MAR       # optional, see DIALECTS in tts/habibi_synth.py

One utterance, saving the reply:
    uv run full-run/habibi_adapter.py recordings/question.wav -o recordings/reply.wav

TTFA benchmark (CPU TTS is slow, so raise the timeout):
    uv run benchmark/ttfa_harness.py --adapter full-run/habibi_adapter.py:HabibiS2SAdapter -n 1 --timeout 300 recordings/question.wav

The reply is spoken sentence by sentence as the LLM streams it, so the first sentence
starts synthesizing before the LLM has finished.
"""
import argparse
import asyncio
import os
import re
import time
from functools import lru_cache

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from benchmark.example_adapters import LocalModelAdapter
from benchmark.ttfa_harness import CHUNK, SR, load_input
from llm.llm_chat import SYSTEM_PROMPT
from tts.habibi_synth import HabibiTTS

load_dotenv()

ENDPOINT_MS = 500      # silence after speech that ends the user's turn
VAD_WINDOW = 512       # samples per silero VAD call at 16 kHz
SENTENCE_END = re.compile(r"[.!?؟\n]")


# Models are loaded once per process and shared: the harness builds a fresh adapter per run.
@lru_cache(maxsize=1)
def _asr():
    from qwen_asr import Qwen3ASRModel
    return Qwen3ASRModel.from_pretrained("atlasia/moulsot.v0.3", dtype="float32", device_map="cpu")


@lru_cache(maxsize=1)
def _tts():
    return HabibiTTS(dialect=os.getenv("HABIBI_DIALECT", "MAR"))


@lru_cache(maxsize=1)
def _llm():
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)


@lru_cache(maxsize=1)
def _vad_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def load_models():
    """Load the models up front so the first run isn't charged for it (the LLM client is cheap)."""
    _asr(), _tts(), _vad_model()


class HabibiS2SAdapter(LocalModelAdapter):
    def __init__(self, system_prompt: str = SYSTEM_PROMPT, verbose: bool = True):
        super().__init__()
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.transcript = None
        self.reply = None

    async def connect(self):
        await asyncio.to_thread(load_models)   # excluded from the TTFA measurement
        await super().connect()

    def _log(self, t0: float, msg: str):
        if self.verbose:
            print(f"[{(time.perf_counter() - t0) * 1000:7.0f} ms after end of turn] {msg}", flush=True)

    async def run(self, inbox, outbox):
        audio = await self._listen(inbox)
        t0 = time.perf_counter()

        result = await asyncio.to_thread(_asr().transcribe, audio=(audio, SR), language="Arabic")
        self.transcript = result[0].text
        self._log(t0, f"ASR: {self.transcript}")

        reply = []
        async for sentence in self._llm_sentences(self.transcript):
            self._log(t0, f"LLM: {sentence}")
            reply.append(sentence)
            pcm = await asyncio.to_thread(_tts().synthesize_pcm, sentence)
            self._log(t0, f"TTS: {len(pcm) // 2 / SR:.1f} s of audio")
            for i in range(0, len(pcm), CHUNK * 2):
                await outbox.put(pcm[i:i + CHUNK * 2])
        self.reply = " ".join(reply)
        await outbox.put(None)

    async def _listen(self, inbox) -> np.ndarray:
        """Collects audio until ENDPOINT_MS of silence follows speech; returns float32."""
        import torch
        from silero_vad import VADIterator

        vad = VADIterator(_vad_model(), sampling_rate=SR, min_silence_duration_ms=ENDPOINT_MS)
        chunks, pending, in_speech = [], np.zeros(0, dtype=np.float32), False
        while True:
            samples = np.frombuffer(await inbox.get(), dtype=np.int16).astype(np.float32) / 32768
            chunks.append(samples)
            pending = np.concatenate([pending, samples])
            while len(pending) >= VAD_WINDOW:
                event = vad(torch.from_numpy(pending[:VAD_WINDOW]))
                pending = pending[VAD_WINDOW:]
                if event and "start" in event:
                    in_speech = True
                elif event and "end" in event and in_speech:
                    return np.concatenate(chunks)

    async def _llm_sentences(self, text: str):
        """Streams the LLM reply and yields it one sentence at a time."""
        stream = await _llm().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
            stream=True,
        )
        buffer = ""
        async for event in stream:
            if not event.choices:
                continue
            buffer += event.choices[0].delta.content or ""
            while match := SENTENCE_END.search(buffer):
                sentence, buffer = buffer[:match.end()].strip(), buffer[match.end():]
                if sentence:
                    yield sentence
        if buffer.strip():
            yield buffer.strip()


async def _reply_to_file(in_path: str, out_path: str):
    """Feeds a WAV file in real time, like the harness, and saves the spoken reply."""
    pcm = load_input(in_path)
    adapter = HabibiS2SAdapter()
    await adapter.connect()

    async def feed():
        silence = np.zeros(CHUNK, dtype=np.int16)
        for start in range(0, len(pcm), CHUNK):
            await adapter.send_audio(pcm[start:start + CHUNK].tobytes())
        while True:   # trailing silence so the VAD can see the end of the turn
            await adapter.send_audio(silence.tobytes())
            await asyncio.sleep(CHUNK / SR)

    feeder = asyncio.create_task(feed())
    try:
        out = b"".join([chunk async for chunk in adapter.receive_audio()])
    finally:
        feeder.cancel()
        await adapter.close()
    sf.write(out_path, np.frombuffer(out, dtype=np.int16), SR)
    print(f"\nuser:  {adapter.transcript}\nreply: {adapter.reply}\n-> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="16 kHz mono WAV with one user utterance")
    parser.add_argument("-o", "--out", default="reply.wav", help="where to save the spoken reply (16 kHz)")
    args = parser.parse_args()
    asyncio.run(_reply_to_file(args.file, args.out))


if __name__ == "__main__":
    main()

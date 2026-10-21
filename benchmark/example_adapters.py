"""
Example adapters for ttfa_harness.

- LocalModelAdapter: base class for a model running in this process.
- FakeS2SAdapter:    a simulated system with a known delay, to sanity-check the harness.
- WebSocketPCMAdapter: a local model served over a websocket that speaks raw PCM.
"""
import asyncio
import os

import numpy as np

from benchmark.ttfa_harness import CHUNK, SR, S2SAdapter


class LocalModelAdapter(S2SAdapter):
    """Base class for in-process models. Subclass it and implement run().

    run() reads 16 kHz int16 PCM chunks from `inbox` and puts reply PCM onto `outbox`
    (put None to end the reply stream). Wrap blocking inference in
    `await asyncio.to_thread(...)`: blocking the event loop would stall the harness's
    real-time sender and the timestamps would be wrong.
    """

    def __init__(self):
        self._inbox: asyncio.Queue[bytes] = asyncio.Queue()
        self._outbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def run(self, inbox: asyncio.Queue, outbox: asyncio.Queue):
        raise NotImplementedError

    async def connect(self):
        self._task = asyncio.create_task(self.run(self._inbox, self._outbox))

    async def send_audio(self, pcm_bytes: bytes):
        if self._task.done():
            self._task.result()   # re-raise a crash in run() instead of feeding a dead model
        await self._inbox.put(pcm_bytes)

    async def receive_audio(self):
        while (chunk := await self._outbox.get()) is not None:
            yield chunk

    async def close(self):
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class FakeS2SAdapter(LocalModelAdapter):
    """Pretends to be a system: energy-based endpointing, then a fixed processing delay,
    then 100 ms of silence followed by a tone. Expected TTFA ~= endpoint_ms + delay_ms
    (plus up to one chunk), so a run that reports something else means the harness is off.
    """

    def __init__(self, endpoint_ms: int = 500, delay_ms: int = 300, rms_threshold: float = 300):
        super().__init__()
        self.endpoint_chunks = endpoint_ms * SR // 1000 // CHUNK
        self.delay_s = delay_ms / 1000
        self.rms_threshold = rms_threshold

    async def run(self, inbox, outbox):
        heard_speech = False
        quiet_chunks = 0
        while True:
            samples = np.frombuffer(await inbox.get(), dtype=np.int16).astype(np.float32)
            loud = np.sqrt(np.mean(samples ** 2)) > self.rms_threshold
            heard_speech |= loud
            quiet_chunks = 0 if loud else quiet_chunks + 1
            if heard_speech and quiet_chunks >= self.endpoint_chunks:
                break

        await asyncio.sleep(self.delay_s)
        await outbox.put(np.zeros(SR // 10, dtype=np.int16).tobytes())
        t = np.arange(SR // 2) / SR
        await outbox.put((8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes())
        await outbox.put(None)


class WebSocketPCMAdapter(S2SAdapter):
    """A local server that takes binary frames of 16 kHz int16 PCM and streams back binary
    int16 PCM. Text frames (transcripts, events) are ignored. Set TTFA_WS_URL to override."""

    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("TTFA_WS_URL", "ws://localhost:8765")
        self._ws = None

    async def connect(self):
        import websockets
        self._ws = await websockets.connect(self.url, max_size=None)

    async def send_audio(self, pcm_bytes: bytes):
        await self._ws.send(pcm_bytes)

    async def receive_audio(self):
        async for message in self._ws:
            if isinstance(message, bytes):
                yield message

    async def close(self):
        if self._ws:
            await self._ws.close()

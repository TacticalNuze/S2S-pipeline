"""
Live voice bot on the GPU: talk to it from your browser, it answers out loud.

    browser mic ──WebRTC──► silero VAD ─► moulsot ASR ─► LLM ─► Habibi TTS ──WebRTC──► browser speaker

Same setup as pipecat-quickstart/server/bot.py, but with the local moulsot (speech-to-text)
and Habibi (text-to-speech) models instead of Deepgram and Cartesia. The LLM is any
OpenAI-compatible endpoint, configured in .env:

    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...      # optional, for non-OpenAI providers (Ollama, LM Studio, ...)
    OPENAI_MODEL=gpt-4.1     # optional
    HABIBI_DIALECT=MAR       # optional, see DIALECTS in tts/habibi_synth.py
    HABIBI_NFE_STEP=32       # optional, fewer steps = faster TTS, lower quality
    S2S_DEVICE=cuda          # optional, "cpu" to run without a GPU (slow)

Run, then open http://localhost:7860 and click Connect:

    uv run full-run/bot_habibi.py

The models are loaded once at startup (the first run downloads them) and shared by every
browser session.
"""
import asyncio
import os
from collections.abc import AsyncGenerator
from functools import lru_cache

import numpy as np
import torch
import truststore
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import ErrorFrame, Frame, LLMRunFrame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

from llm.llm_chat import SYSTEM_PROMPT
from tts.habibi_synth import SR as HABIBI_SR
from tts.habibi_synth import HabibiTTS

# Verify HTTPS with the Windows certificate store (corporate proxy breaks model downloads otherwise).
truststore.inject_into_ssl()
load_dotenv(override=True)

ASR_MODEL = "atlasia/moulsot.v0.3"
DEVICE = os.getenv("S2S_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Loaded once per process: the runner calls bot() again for every browser session.
@lru_cache(maxsize=1)
def _asr():
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
    return Qwen3ASRModel.from_pretrained(ASR_MODEL, dtype=dtype, device_map=DEVICE)


@lru_cache(maxsize=1)
def _tts():
    return HabibiTTS(
        dialect=os.getenv("HABIBI_DIALECT", "MAR"),
        device=DEVICE,
        nfe_step=int(os.getenv("HABIBI_NFE_STEP", "32")),
    )


def load_models():
    logger.info(f"Loading moulsot and Habibi on {DEVICE}...")
    _asr(), _tts()
    logger.info("Models loaded")


class MoulsotSTTService(SegmentedSTTService):
    """Transcribes each VAD speech segment with moulsot."""

    def __init__(self, **kwargs):
        super().__init__(settings=STTSettings(model=ASR_MODEL, language=None), **kwargs)

    def wants_wav_segments(self) -> bool:
        return False   # run_stt gets the raw 16-bit PCM

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_processing_metrics()
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
            result = await asyncio.to_thread(
                _asr().transcribe, audio=(samples, self.sample_rate), language="Arabic"
            )
            await self.stop_processing_metrics()
            text = result[0].text.strip()
            logger.info(f"ASR: {text}")
            if text:
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601())
        except Exception as e:
            yield ErrorFrame(error=f"moulsot ASR failed: {e}")


class HabibiTTSService(TTSService):
    """Speaks the LLM reply with Habibi, one sentence at a time (pipecat splits the text)."""

    def __init__(self, **kwargs):
        tts = _tts()
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="SWivid/Habibi-TTS", voice=None, language=None),
            **kwargs,
        )
        self._habibi = tts
        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)
            logger.info(f"TTS: {text}")
            wave = await asyncio.to_thread(self._habibi.synthesize, text)
            await self.stop_ttfb_metrics()
            pcm = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            yield TTSAudioRawFrame(
                audio=await self._resampler.resample(pcm, HABIBI_SR, self.sample_rate),
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
        except Exception as e:
            yield ErrorFrame(error=f"Habibi TTS failed: {e}")
        finally:
            await self.stop_ttfb_metrics()


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    logger.info("Starting bot")

    stt = MoulsotSTTService()
    tts = HabibiTTSService()
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        settings=OpenAILLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[],
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation with a short greeting. Sent as a user message: a second
        # system message is rejected by local models' chat templates (Qwen3.8 27B wants the
        # system message first and at least one user message; Atlas-Chat has no system role).
        context.add_message({"role": "user", "content": "Greet me in one short sentence."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point, called by the pipecat runner for every session."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    load_models()   # before the server starts, so the first session doesn't wait on it
    main()

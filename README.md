# S2S pipeline: moulsot ASR + Habibi TTS

A speech-to-speech pipeline for Arabic (Moroccan Darija by default). It runs in two ways:

- **Live voice bot on the GPU** (`full-run/bot_habibi.py`): talk to it from your browser
  and it answers out loud, like `pipecat-quickstart`. See [Live voice bot](#live-voice-bot-gpu).
- **From a WAV file on the CPU** (`full-run/habibi_adapter.py`): record a question and get
  the spoken answer back as a WAV. See steps 3 to 6.

```
your voice ──► silero VAD ──► moulsot ASR ──► LLM ──► Habibi TTS ──► spoken reply
               (end of turn)   (speech→text)  (reply)  (text→speech)
```

- **ASR**: [`atlasia/moulsot.v0.3`](https://huggingface.co/atlasia/moulsot.v0.3), loaded with `qwen-asr`.
- **LLM**: any OpenAI-compatible endpoint (OpenAI, or a local Ollama / LM Studio server, etc.).
- **TTS**: [Habibi-TTS](https://github.com/SWivid/Habibi-TTS), the Unified model, which supports several Arabic dialects.

The LLM reply is streamed and spoken one sentence at a time, so the first sentence is
synthesized before the LLM has finished.

## Project layout

Run every command from the `S2S-pipeline/` folder.

| Folder | Script | What it does |
| --- | --- | --- |
| `asr/` | `moulsot_asr.py` | Speech-to-text only: prints the transcript of one or more WAV files. |
| `tts/` | `habibi_synth.py` | Text-to-speech only: turns a line of text into a WAV. Also used by the full pipeline. |
| `llm/` | `llm_chat.py` | Text chat with the LLM, to check your `.env`. Holds `SYSTEM_PROMPT`, which the full pipeline uses. |
| `llm/servers/` | `serve_*.ps1` | Start a local LLM server (llama.cpp) for one model, with its tuned settings. See [Local LLM server](#local-llm-server-llamacpp). |
| `full-run/` | `bot_habibi.py` | **Live voice bot** on the GPU, in your browser (Pipecat + WebRTC). |
| `full-run/` | `habibi_adapter.py` | **The full pipeline from a file**, on CPU: WAV in, spoken reply WAV out. |
| `benchmark/` | `ttfa_harness.py` | Measures time to first audio (TTFA) of the pipeline. |
| `benchmark/` | `example_adapters.py` | Base adapter class plus a fake adapter for testing the harness. |
| `tools/` | `record_wav.py` | Records a 16 kHz mono WAV from your microphone. |
| `recordings/` | | Sample and recorded WAV files. |

`asr`, `tts`, `llm` and `benchmark` are installed as packages of this project by `uv sync`
(see `[tool.hatch.build.targets.wheel]` in `pyproject.toml`), so any script can use, for
example, `from tts.habibi_synth import HabibiTTS`.

## 1. Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.13 (uv installs it if it's missing).

```bash
cd S2S-pipeline
uv sync
```

This installs everything in `pyproject.toml` into `.venv`, including the CUDA 12.8 build
of PyTorch (from the `pytorch-cu128` index configured there), which works on RTX 50xx and
older NVIDIA GPUs. To check that the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On the first run the models are downloaded from Hugging Face: moulsot (3.9 GB),
Habibi-TTS (1.3 GB) and the vocos vocoder. Later runs use the local cache.

> **Behind a corporate proxy?** The scripts call `truststore.inject_into_ssl()`, so HTTPS
> is verified against the Windows certificate store. Without it, downloads behind a
> TLS-inspecting proxy fail with `SSL: CERTIFICATE_VERIFY_FAILED`.

## 2. Configure the LLM

Create a `.env` file in `S2S-pipeline/`. A `.env` in a parent folder also works, because
the scripts look upward until they find one.

```ini
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1        # optional, default gpt-4.1
OPENAI_BASE_URL=            # optional, for providers other than OpenAI
HABIBI_DIALECT=MAR          # optional, default MAR (Moroccan)
```

To use a local model instead of OpenAI, point `OPENAI_BASE_URL` at it. The API key can be
any non-empty string:

```ini
# Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b

# LM Studio
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=<model id shown in LM Studio>
```

Check the settings with a text chat (no audio involved):

```bash
uv run llm/llm_chat.py "السلام، شنو سميتك؟"   # one question
uv run llm/llm_chat.py                      # interactive, empty line to quit
```

The system prompt, `SYSTEM_PROMPT` in `llm/llm_chat.py`, asks for one or two short
sentences in Darija written in Arabic script. Both full-run scripts use it. Edit it if you
change `HABIBI_DIALECT`.

## Local LLM server (llama.cpp)

Instead of a cloud model, the pipeline can use an LLM running on your own GPU with
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, which speaks the same
OpenAI API.

### Install llama.cpp

From a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases), download the two
Windows CUDA zips for your driver (here CUDA 13.4, x64) and extract **both into the same
folder**, by default `C:\Users\<you>\tools\llama.cpp`:

- `llama-<build>-bin-win-cuda-13.4-x64.zip`: llama.cpp itself (`llama-server.exe` and its DLLs).
- `cudart-llama-bin-win-cuda-13.4-x64.zip`: the NVIDIA CUDA runtime it needs, so you don't
  have to install the CUDA Toolkit.

Avoid the winget package: it is the Vulkan build, which is several times slower on NVIDIA.
Check that the GPU is found:

```powershell
& "$HOME\tools\llama.cpp\llama-server.exe" --list-devices   # should list CUDA0: NVIDIA ...
```

### Pick a model

There is one script per model in `llm/servers/`, with the settings that tested best on an
RTX 5060 Ti 16 GB. Full results: [benchmark/results/llm_compare_2026-10-21.md](benchmark/results/llm_compare_2026-10-21.md).

| Script | Model | VRAM | First sentence | Fits next to the voice bot? | Use it for |
| --- | --- | --- | --- | --- | --- |
| `serve_atlas_chat_9b.ps1` | Atlas-Chat 9B (Darija), Q4_K_M | ~6.9 GB | ~0.44 s | **Yes** | **The live voice bot.** Most natural Darija. |
| `serve_qwen35_9b.ps1` | Qwen3.5 9B, Q4_K_M + MTP | ~6 GB | ~0.6 s | Yes | A second small option; weaker Darija. |
| `serve_qwen38_27b.ps1` | Swift-1.5 Qwen3.8 27B, IQ3_S + MTP | ~12.7 GB | ~1.6 s | **No** | Comparing the LLM alone against a cloud model. |

The voice bot's ASR and TTS use about 6 GB of VRAM, so on a 16 GB card only the 9B models
fit beside them. None of the models is reliable on facts (they got places and travel times
wrong in the test), so don't rely on them for factual answers.

Download the 9B models (the 27B comes from
[ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)):

```bash
uv run hf download mradermacher/Atlas-Chat-9B-GGUF Atlas-Chat-9B.Q4_K_M.gguf
uv run hf download unsloth/Qwen3.5-9B-MTP-GGUF Qwen3.5-9B-Q4_K_M.gguf
```

### Start it

```powershell
.\llm\servers\serve_atlas_chat_9b.ps1
```

It prints the `.env` settings to use, then runs until you press `Ctrl+C`:

```ini
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=local
OPENAI_MODEL=atlas-chat-9b
```

Put those in `.env`, check with `uv run llm/llm_chat.py "السلام"`, then start the bot as
usual in another terminal. Opening http://127.0.0.1:8080 in a browser also gives you
llama.cpp's own chat page.

Options, the same for every script:

```powershell
.\llm\servers\serve_qwen38_27b.ps1 -Port 8081 -Context 32768   # another port or context size
.\llm\servers\serve_atlas_chat_9b.ps1 -Model D:\models\atlas.gguf   # a model file elsewhere
.\llm\servers\serve_qwen35_9b.ps1 -LlamaDir D:\llama.cpp        # llama.cpp elsewhere (or set LLAMA_CPP_DIR)
.\llm\servers\serve_atlas_chat_9b.ps1 --threads 4 --no-webui    # anything else goes to llama-server
```

What the settings do:

- **Common to every model** (in `llm/servers/_common.ps1`): the whole model on the GPU
  (`--n-gpu-layers all`), flash attention, an 8-bit KV cache, and one conversation at a time.
- **Qwen models**: `--spec-type draft-mtp --spec-draft-n-max 2` uses the model's built-in
  multi-token prediction layer to draft 2 tokens ahead. That's about 1.7x faster on long
  answers, and 2 beat 1 and 3 on this GPU. `--reasoning off` turns thinking off, or the voice
  bot would wait for hidden reasoning before speaking.
- **Atlas-Chat**: no thinking or MTP, and an 8192-token maximum context. Its chat template
  rejects system messages, but llama.cpp merges the system prompt into the first user
  message, so `SYSTEM_PROMPT` still works.

## Live voice bot (GPU)

`full-run/bot_habibi.py` is the same kind of bot as `pipecat-quickstart/server/bot.py`,
but it uses the local moulsot and Habibi models instead of Deepgram and Cartesia. Pipecat
handles the browser connection (WebRTC), voice activity detection, turn-taking and
interruptions.

1. Configure the LLM in `.env` (step 2).
2. Start the bot:

   ```bash
   uv run full-run/bot_habibi.py
   ```

   It loads both models onto the GPU (about 20 s once they're downloaded), then starts a
   server. Wait for `Open: http://localhost:7860`.
3. Open **http://localhost:7860** in your browser, click **Connect** and allow the
   microphone. The bot greets you, and then you can talk to it. You can also interrupt it
   while it's speaking.
4. Press `Ctrl+C` in the terminal to stop the bot.

The terminal shows what moulsot heard (`ASR: ...`) and each sentence being spoken
(`TTS: ...`). Measured on an RTX 5060 Ti after warm-up: ASR takes about 0.2 s per
utterance, and Habibi produces 2.6 s of speech in about 1 s.

Optional settings in `.env`, in addition to the LLM ones:

```ini
HABIBI_DIALECT=MAR      # voice/dialect, see the list in step 4
HABIBI_NFE_STEP=32      # lower (e.g. 16) for faster but slightly lower-quality speech
S2S_DEVICE=cuda         # "cpu" to run without a GPU (much slower)
```

Habibi speaks each sentence only after the whole sentence is generated, so short
sentences start playing sooner.

## 3. Record a question

```bash
uv run tools/record_wav.py -o recordings/question.wav        # press Enter to stop
uv run tools/record_wav.py -o recordings/question.wav -s 5   # or record exactly 5 seconds
uv run tools/record_wav.py --list-devices                    # if the recording is silent, pick a mic with -d <index>
```

The file pipeline needs **16 kHz mono** WAV. `record_wav.py` always writes that format. To
use another file, convert it first, for example with
`ffmpeg -i in.mp3 -ar 16000 -ac 1 recordings/question.wav`.

There are already sample recordings in `recordings/`.

## 4. Test each stage on its own (optional, recommended)

Running each stage on its own makes problems easier to find before you run the whole
pipeline. The LLM check is in step 2.

**ASR** prints what moulsot hears:

```bash
uv run asr/moulsot_asr.py recordings/question.wav
```

**TTS** speaks a line of text:

```bash
uv run tts/habibi_synth.py "السلام عليكم، كيداير؟" -o recordings/tts_test.wav
uv run tts/habibi_synth.py --dialect EGY "إزيك عامل إيه؟" -o recordings/tts_egy.wav
uv run tts/habibi_synth.py --nfe-step 16 "..."    # about twice as fast on CPU, slightly lower quality
```

Options for `habibi_synth.py`:

- `--dialect`: one of `MAR, MSA, SAU, UAE, ALG, IRQ, EGY, OMN, TUN, LEV, SDN, LBY, UNK`.
- `--ref-audio me.wav --ref-text "exact transcript"`: clone another voice. Habibi copies
  the voice of a short reference clip (only the first 12 s are used) and needs its exact
  transcript. If you leave these out, the sample bundled with habibi-tts for that dialect
  is used. OMN, TUN, LEV, SDN, LBY and UNK have no bundled sample and use the MSA one.
  A reference in the same dialect as the text gives the best results.
- `--speed`: speaking rate, default `1.0`.

## 5. Run the full pipeline from a file

```bash
uv run full-run/habibi_adapter.py recordings/question.wav -o recordings/reply.wav
```

What happens:

1. The WAV is fed in at real-time speed, followed by silence, as a live microphone would be.
2. Silero VAD ends your turn after 500 ms of silence.
3. moulsot transcribes it.
4. The LLM streams a reply, which is split into sentences.
5. Habibi speaks each sentence, and the audio is saved to `recordings/reply.wav` (16 kHz).

Progress is logged with timings measured from the end of your turn. The numbers below are
only an example and depend on your CPU and LLM:

```
[   2140 ms after end of turn] ASR: <your transcript>
[   2900 ms after end of turn] LLM: <first sentence>
[  14500 ms after end of turn] TTS: 2.8 s of audio
...
user:  <your transcript>
reply: <full reply>
-> recordings/reply.wav
```

Play `recordings/reply.wav` to hear the answer.

## 6. Measure latency (TTFA)

`benchmark/ttfa_harness.py` streams each file in real time and measures the time from the
end of your speech to the first audible reply audio. On CPU, Habibi is slow, so raise the
timeout (default 15 s):

```bash
uv run benchmark/ttfa_harness.py --adapter full-run/habibi_adapter.py:HabibiS2SAdapter -n 1 --timeout 300 recordings/question.wav
```

- `--adapter`: either `path/to/file.py:ClassName` or `package.module:ClassName`.
- `-n`: how many times each file is run. Models are loaded once and kept for later runs,
  and loading is not counted in the measurement.
- `--out results.json`: save each run and the p50/p95/p99 summary.

To check the harness itself, run it on the fake adapter. It should report about 800 ms:

```bash
uv run benchmark/ttfa_harness.py --adapter benchmark.example_adapters:FakeS2SAdapter -n 3 recordings/question.wav
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'tts'` (or `llm`, `benchmark`) | Run `uv sync` so the project's own folders are installed, and start scripts with `uv run`. |
| `expected 16000 Hz mono audio` | Convert the file to 16 kHz mono (see step 3). |
| `no speech detected in input file` | The recording is too quiet or empty. Re-record, or choose another mic with `-d`. |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Make sure you run through `uv run`, so `truststore` is installed and used. |
| `Missing credentials` or an LLM connection error | Check `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `OPENAI_MODEL` in `.env`, and that the file is saved. Test with `uv run llm/llm_chat.py`. |
| `UnicodeEncodeError: 'charmap' codec` | `tts/habibi_synth.py` and `llm/llm_chat.py` switch the console to UTF-8. Update them if you copied an older version. |
| Live bot: no sound or no reply | Allow microphone access in the browser and check the terminal for `ASR:` lines. If there are none, the mic isn't reaching the bot. |
| Live bot runs on CPU | `torch.cuda.is_available()` is `False`. Run `uv sync` again so the CUDA build of torch is installed. |
| TTS is very slow | `habibi_adapter.py` runs on CPU; use `bot_habibi.py` for the GPU. Lower `nfe_step` (`--nfe-step 16` in `habibi_synth.py`, `HABIBI_NFE_STEP=16` for the live bot, or `HabibiTTS(nfe_step=16)` in `habibi_adapter.py`'s `_tts()`). |
| Local LLM: `Port 8080 is already in use` | Another server is still running. Stop it (`Get-Process llama-server \| Stop-Process`), or start this one with `-Port`. |
| Local LLM is slow (under 20 tok/s) | Check the server log for `CUDA0`. Without it you're on the CPU or Vulkan build: use the CUDA zips plus the cudart zip, in the same folder. |
| Local LLM: out of memory with the voice bot running | Use a 9B model; the 27B doesn't fit next to ASR and TTS on 16 GB. |
| Reply is in the wrong dialect | Make `HABIBI_DIALECT` and `SYSTEM_PROMPT` ask for the same dialect. |

## About `pipecat-quickstart/`

`pipecat-quickstart/` is the Pipecat web voice bot that `full-run/bot_habibi.py` is based
on. It uses cloud services (Deepgram STT, Cartesia TTS) and has its own README and
`uv sync`. `bot_habibi.py` does not need it: Pipecat is installed in this project.

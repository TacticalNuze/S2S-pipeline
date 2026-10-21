<#
.SYNOPSIS
Serves Atlas-Chat 9B (MBZUAI, Moroccan Darija, Q4_K_M, 5.8 GB) with llama.cpp on the GPU.

.DESCRIPTION
The recommended LLM for the live voice bot: the most natural Darija of the local models
tested, the fastest first sentence, and it fits next to moulsot ASR and Habibi TTS (the GPU
peaked at 12.8 GB of 16.3 GB with all three loaded).

Based on Gemma 2: no thinking mode, no MTP layer, and a maximum context of 8192 tokens.
The chat template embedded in the GGUF rejects system messages and non-alternating roles,
which breaks the live bot (HTTP 400 "System role not supported"). This script replaces it
with templates/atlas-chat.jinja: the same Gemma format, with the system messages merged
into the first user turn.

Measured on an RTX 5060 Ti 16 GB (llama.cpp b11669): first token ~0.08 s,
first sentence ~0.44 s, ~51 tok/s.

.EXAMPLE
.\llm\servers\serve_atlas_chat_9b.ps1
#>
param(
    [string]$Model,                       # default: the file in the Hugging Face cache
    [ValidateRange(512, 8192)] [int]$Context = 8192,   # the model's maximum
    [int]$Port = 8080,
    [string]$BindHost = "127.0.0.1",
    [string]$LlamaDir,                    # default: $env:LLAMA_CPP_DIR or ~\tools\llama.cpp
    [Parameter(ValueFromRemainingArguments)] [string[]]$ExtraArgs   # passed through to llama-server
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_common.ps1")

$server = Resolve-LlamaServer $LlamaDir
$modelPath = Resolve-Model $Model `
    (Join-Path $HOME ".cache\huggingface\hub\models--mradermacher--Atlas-Chat-9B-GGUF\snapshots\*\Atlas-Chat-9B.Q4_K_M.gguf") `
    "Download it with: uv run hf download mradermacher/Atlas-Chat-9B-GGUF Atlas-Chat-9B.Q4_K_M.gguf"

Start-LlamaServer -Server $server -Model $modelPath -Alias "atlas-chat-9b" -BindHost $BindHost -Port $Port -ExtraArgs $ExtraArgs -ModelArgs @(
    "-c", $Context,
    "--chat-template-file", (Join-Path $PSScriptRoot "templates\atlas-chat.jinja"),
    "--temp", "0.7", "--top-p", "0.9", "--min-p", "0.05", "--repeat-penalty", "1.1"
)
